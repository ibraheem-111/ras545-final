import threading
import traceback
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String as StringMsg, Bool
from final_interfaces.action import FindObject, MoveRobot
from final_interfaces.msg import Target
from google import genai
from google.genai import types as gt
import time

MODEL_ID = "gemini-2.5-pro"

# --- System Prompt (UPDATED) ---

SYSTEM_INSTRUCTION = """
You are a precise Robot Orchestrator. 

YOUR GOAL:
1. Find coordinates for every object mentioned.
2. Execute the physical movement plan.

WORKSPACE DESCRIPTION:
- The robot operates on a flat surface with a top-down camera view.
- The blocks closest to you have relatively lower X values.
- The blocks furthest from you have relatively higher X values.
- The blocks to your left have higher Y values.
- The blocks to your right have lower Y values.

STRICT OPERATIONAL RULES:
- Phase 1 (Scouting): Identify ALL objects referenced in the user's prompt. 
    - Call `find_object` **EXACTLY ONCE**. 
    - The query argument MUST be a single string describing ALL items (e.g., "blue blocks and red blocks"). 
    - Do not include any directive language (e.g., "left", "right") in the query.
    - **DO NOT** make multiple function calls for detection.
- Phase 2 (Decision): 
    - Review the labels returned by `find_object`.
    - If the user asked for a singular object (e.g., "pick the blue block") but the tool returned MULTIPLE variations (e.g., "blue_block_1" and "blue_block_2"), **DO NOT EXECUTE**.
    - Use spatial reasoning based upon coordinate values and workspace description to determine which object to pick (e.g., closest to robot, leftmost, etc.).
    - If clarification is needed (e.g., multiple objects found but only one requested), ASK the user a direct question specifying the options.
- Phase 3 (Execution): 
    - If the object is unique or specific (e.g., "blue_block_1"), execute the trajectory immediately.
    - **DO NOT** output text summaries like "I have found the objects" if you are proceeding to execution. Only speak if you need clarification.

CONSTANTS:
- Z_Movement = 0 
- Z_Pick_or_Drop = -47
- Block_Height = 15
- Block_Offset = 15
- Home = [240, 0, 150]

TRAJECTORY STRUCTURE:
1. MOTION: { "type": "motion_command", "object": { "x": 100, "y": 200, "z": 0, "mode": "joint" } }
2. SUCTION: { "type": "suction_command", "object": { "state": true } } (true=ON, false=OFF)
3. HOME: { "type": "home", "object": {} }

LOGIC (Pick A, Place on B):
1. Motion: Above A (Z_Movement, Joint)
2. Motion: Down to A (Z_Pick_or_Drop, Linear)
3. Suction: Close (true)
4. Motion: Up (Z_Movement, Linear)
5. Motion: Above B (Z_Movement, Joint)
6. Motion: Down to B (Z_Pick_or_Drop + Block_Height + 5, Linear)
7. Suction: Open (false)
8. Motion: Up (Z_Movement, Linear)
9. Home
"""

# --- Tool Definitions (UPDATED) ---
FIND_OBJECT_FN = {
    "name": "find_object",
    "description": "Locates ALL referenced objects in the image. Input a SINGLE query string describing all items (e.g., 'red block and blue block').",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}

EXECUTE_TRAJECTORY_FN = {
    "name": "execute_trajectory",
    "description": "Executes the full list of robot commands. Call this immediately after finding objects.",
    "parameters": {
        "type": "object",
        "properties": {
            "trajectory": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["home", "suction_command", "motion_command"]
                        },
                        "object": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                                "z": {"type": "number"},
                                "mode": {"type": "string", "enum": ["joint", "linear"]},
                                "state": {"type": "boolean"}
                            }
                        }
                    },
                    "required": ["type", "object"]
                }
            }
        },
        "required": ["trajectory"]
    }
}

TOOLS = [gt.Tool(function_declarations=[FIND_OBJECT_FN, EXECUTE_TRAJECTORY_FN])]

class Orchestrator(Node):
    def __init__(self):
        super().__init__("orchestrator")
        
        self.cb_group = ReentrantCallbackGroup()

        self._find_object_client = ActionClient(self, FindObject, "find_object", callback_group=self.cb_group)
        self._move_robot_client = ActionClient(self, MoveRobot, "move_robot", callback_group=self.cb_group)

        self._action_done_event = threading.Event()
        self._current_result = None

        # Setup Publisher for questions back to user
        self.feedback_pub = self.create_publisher(StringMsg, "robot_feedback", 10)

        try:
            self.client = genai.Client(vertexai=True, location="us-central1")
            self.chat = self.client.chats.create(
                model=MODEL_ID,
                config=gt.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=TOOLS,
                    temperature=0.0,
                )
            )
            self.get_logger().info("Orchestrator Ready.")
        except Exception as e:
            self.get_logger().error(f"AI Init Failed: {e}")

        self.sub = self.create_subscription(StringMsg, "central_query", self.query_cb, 10, callback_group=self.cb_group)

    def query_cb(self, msg: StringMsg):
        """Main Agent Loop with Parallel Function Call Support"""
        self.get_logger().info(f"Starting Task: {msg.data}")
        try:
            response = self.chat.send_message(msg.data)

            # Loop as long as the model returns parts
            while response.candidates and response.candidates[0].content.parts:
                
                # 1. Extract ALL function calls
                function_calls = [
                    part.function_call for part in response.candidates[0].content.parts 
                    if part.function_call
                ]

                # If no function calls found, check if it's just text (Clarification Question)
                if not function_calls:
                    text_resp = response.text or ""
                    if text_resp:
                        self.get_logger().info(f"ROBOT QUESTION: {text_resp}")
                        self.feedback_pub.publish(StringMsg(data=text_resp))
                    break 
                
                self.get_logger().info(f"Processing {len(function_calls)} tool calls...")
                
                # 2. Execute ALL calls
                function_responses = []
                for fn in function_calls:
                    self.get_logger().info(f"--> Calling: {fn.name}")
                    
                    tool_result = {}
                    if fn.name == "find_object":
                        tool_result = self.handle_find_object(fn.args["query"])
                    elif fn.name == "execute_trajectory":
                        tool_result = self.handle_execute_trajectory(fn.args["trajectory"])
                    
                    # 3. Create response
                    function_responses.append(
                        gt.Part(function_response=gt.FunctionResponse(name=fn.name, response=tool_result))
                    )
                
                # 4. Send ALL responses back to model
                response = self.chat.send_message(function_responses)

            self.get_logger().info("Task Loop Finished.")

        except Exception as e:
            self.get_logger().error(f"Orchestrator Loop Error: {e}")
            traceback.print_exc()

    # --- Tool Handlers ---

    def handle_find_object(self, query):
        self._action_done_event.clear()
        self._current_result = None
        
        if not self._find_object_client.wait_for_server(timeout_sec=2.0):
            return {"error": "Object Finder Server not available"}

        future = self._find_object_client.send_goal_async(FindObject.Goal(query=query))
        future.add_done_callback(self._goal_response_callback)
        
        if not self._action_done_event.wait(timeout=15.0):
            return {"error": "FindObject timed out"}
        
        if not self._current_result:
            return {"error": "No result received"}

        if not self._current_result.labels:
            return {"error": f"Could not find '{query}'"}

        found = []
        count = min(len(self._current_result.labels), len(self._current_result.x_coords))
        for i in range(count):
            rx = self._current_result.x_coords[i] 
            ry = self._current_result.y_coords[i]
            label = self._current_result.labels[i]
            # The label here will now contain the unique ID (e.g. blue_block_1)
            found.append(f"Found {label} at x={rx:.1f}, y={ry:.1f}")
        
        # We join all found items in one string so the orchestrator knows about all of them at once
        return {"info": "; ".join(found)}

    def handle_execute_trajectory(self, trajectory):
        try:
            self.get_logger().info(f"Executing Trajectory with {len(trajectory)} steps.")
            
            for i, step in enumerate(trajectory):
                step_type = step["type"]
                payload = step.get("object", {})
                
                if step_type == "home":
                    self._run_home_robot()
                    
                elif step_type == "suction_command":
                    state = payload.get("state", False)
                    self._run_gripper(state)
                    time.sleep(2)

                elif step_type == "motion_command":
                    success = self._run_move_robot(
                        payload["x"], payload["y"], payload["z"], 
                        0.0, payload["mode"]
                    )
                    if not success:
                        return {"error": f"Motion failed at step {i}"}
                    
            return {"status": "trajectory_complete"}
        except Exception:
            traceback.print_exc()
            return {"error": "Exception during trajectory execution"}

    # --- Helper Methods ---

    def _run_gripper(self, state: bool):
        self._action_done_event.clear()
        goal = MoveRobot.Goal(command=MoveRobot.Goal.CMD_GRIP, gripper_state=state)
        future = self._move_robot_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)
        self._action_done_event.wait(timeout=2.0)
    
    def _run_home_robot(self):
        self._action_done_event.clear()
        goal = MoveRobot.Goal(command=MoveRobot.Goal.CMD_HOME)
        future = self._move_robot_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)
        if not self._action_done_event.wait(timeout=10.0):
            self.get_logger().warn("Home command timed out.")

    def _run_move_robot(self, x, y, z, r, mode):
        self._action_done_event.clear()
        self._current_result = None
        
        if not self._move_robot_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("Move Robot Server not responding.")
            return False

        target_msg = Target(x=float(x), y=float(y), z=float(z), r=float(r), motion_type=mode)
        goal = MoveRobot.Goal(command=MoveRobot.Goal.CMD_MOVE, target=target_msg)
        
        future = self._move_robot_client.send_goal_async(goal)
        future.add_done_callback(self._goal_response_callback)
        
        return self._action_done_event.wait(timeout=10.0)

    def _goal_response_callback(self, future):
        gh = future.result()
        if not gh.accepted:
            self._action_done_event.set()
            return
        gh.get_result_async().add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future):
        self._current_result = future.result().result
        self._action_done_event.set()

def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
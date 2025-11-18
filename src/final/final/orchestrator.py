# orchestrator.py
import os, threading, numpy as np, cv2
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String as StringMsg
from final_interfaces.action import FindObject, MoveRobot
from sensor_msgs.msg import RegionOfInterest

from enum import Enum
import numpy as np

# # Load camera calibration from numpy files
CAMERA_MATRIX = np.load('/home/ibraheem/ras545/midterm2/camera_matrix.npy')
DIST_COEFFS = np.load('/home/ibraheem/ras545/midterm2/dist_coeffs.npy')

# Load calibration points from saved numpy files
CAMERA_PX_POINTS = np.load('/home/ibraheem/ras545final/camera_points.npy').astype(np.float32)
ROBOT_MM_POINTS = np.load('/home/ibraheem/ras545final/robot_points.npy').astype(np.float32)

class HomeCoordinates(Enum):
    x = 240
    y = 0
    z = 150

class IntermediatePoint(Enum):
    x = 240
    y=0
    z = 80

Z_Draw = -40
Z_PICKUP = 0


# --- Google GenAI (function calling + thinking + chat) ---
from google import genai
from google.genai import types as gt

# Tool declarations per docs (JSON-Schema style)  :contentReference[oaicite:1]{index=1}
FIND_OBJECT_FN = {
    "name": "find_object",
    "description": "Find an object by natural language query. Returns labeled ROIs.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "e.g., 'red block'"},
        },
        "required": ["query"],
    },
}

MOVE_ROBOT_FN = {
    "name": "move_robot",
    "description": "Move the robot to (x, y, z, r). Units: mm, mm, mm, deg. motion_type in {'joint','linear'}.",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"},
            "r": {"type": "number"},
            "motion_type": {"type": "string", "enum": ["joint", "linear"]},
        },
        "required": ["x", "y", "z", "r", "motion_type"],
    },
}

# Convenience tool that chains find_object → pixel→robot → move_robot
MOVE_ABOVE_OBJECT_FN = {
    "name": "move_above_object",
    "description": "Find an object by label, convert ROI center (pixels) → robot (mm), then move above it at z=0 (fixed).",
    "parameters": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "description": "Which object label (e.g., 'red block')"},
            "r": {"type": "number", "description": "End-effector rotation in degrees", "default": 0.0},
            "motion_type": {"type": "string", "enum": ["joint", "linear"], "default": "linear"},
        },
        "required": ["label"],
    },
}

TOOLS = [gt.Tool(function_declarations=[FIND_OBJECT_FN, MOVE_ROBOT_FN, MOVE_ABOVE_OBJECT_FN])]


class Orchestrator(Node):
    def __init__(self):
        super().__init__("orchestrator")

        # --- Actions ---
        self._find_object_client = ActionClient(self, FindObject, "find_object")
        self._move_robot_client = ActionClient(self, MoveRobot, "move_robot")
        if not self._find_object_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("FindObject action server not available")
        if not self._move_robot_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("MoveRobot action server not available")

        self._find_object_result_event = threading.Event()
        self._find_object_result = None
        self._move_robot_result_event = threading.Event()
        self._move_robot_result = None

        # --- GenAI chat (history + thinking) ---
        try:
            # Vertex AI route (ADC) — recommended for server-side.  :contentReference[oaicite:2]{index=2}
            self.ai = genai.Client(
                vertexai=True
            )
            # If you want Developer API instead, use: genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

            self.model_id = "gemini-2.5-flash"  # fast, supports thinking + tools  :contentReference[oaicite:3]{index=3}
            self.config = gt.GenerateContentConfig(
                system_instruction=(
                    "You orchestrate robot actions. Use tools when needed. "
                    "Think first, then act. Keep replies brief."
                ),
                temperature=0.0,
                tools=TOOLS,
                # Enable thinking and keep it budgeted.  :contentReference[oaicite:4]{index=4}
                thinking_config=gt.ThinkingConfig(thinking_budget=512),
            )
            # Create a stateful chat; chat stores history automatically.  :contentReference[oaicite:5]{index=5}
            self.chat = self.ai.chats.create(model=self.model_id, config=self.config)
            self.get_logger().info("GenAI chat ready.")
        except Exception as e:
            self.get_logger().error(f"GenAI init failed: {e}")
            self.chat = None

        # --- Query subscriber ---
        self.sub = self.create_subscription(StringMsg, "central_query", self.query_cb, 10)
        self.get_logger().info("Send queries to /central_query")

    # ===== Pixel→Robot helpers =====
    def pixel_to_robot(self, px: float, py: float, correct_distortion: bool = True) -> tuple[float, float]:
        try:
            pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
            if correct_distortion:
                pt = cv2.undistortPoints(pt, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX).reshape(1, 1, 2)
            M = cv2.getPerspectiveTransform(
                np.array(CAMERA_PX_POINTS, dtype=np.float32),
                np.array(ROBOT_MM_POINTS, dtype=np.float32),
            )
            out = cv2.perspectiveTransform(pt, M)
            return float(out[0, 0, 0]), float(out[0, 0, 1])
        except Exception as e:
            self.get_logger().error(f"pixel_to_robot failed: {e}")
            return 0.0, 0.0

    def roi_center_to_robot(self, roi: RegionOfInterest, correct_distortion: bool = True) -> tuple[float, float]:
        try:
            cx = float(roi.x_offset) + float(roi.width) / 2.0
            cy = float(roi.y_offset) + float(roi.height) / 2.0
            return self.pixel_to_robot(cx, cy, correct_distortion)
        except Exception as e:
            self.get_logger().error(f"roi_center_to_robot failed: {e}")
            return 0.0, 0.0

    # ===== Query → LLM =====
    def query_cb(self, msg: StringMsg):
        if not self.chat:
            self.get_logger().error("Chat not initialized")
            return
        try:
            resp = self.chat.send_message(msg.data)  # history retained automatically  :contentReference[oaicite:6]{index=6}
            part = resp.candidates[0].content.parts[0]
            if hasattr(part, "function_call") and part.function_call:
                self._handle_fn_call(part.function_call)
            else:
                self.get_logger().info(resp.text or "")
        except Exception as e:
            self.get_logger().error(f"chat error: {e}")

    # ===== Tool dispatcher =====
    def _handle_fn_call(self, fn_call):
        name, args = fn_call.name, dict(fn_call.args or {})
        self.get_logger().info(f"tool: {name} args: {args}")

        try:
            if name == "find_object":
                q = str(args["query"])
                result = self._act_find_object(q)
                self._return_fn_result(name, {"labels": list(result.labels), "count": len(result.labels)})
                return

            if name == "move_robot":
                result = self._act_move_robot(
                    float(args["x"]), float(args["y"]), float(args["z"]), float(args["r"]),
                    str(args.get("motion_type", "linear")),
                )
                self._return_fn_result(name, {"success": bool(result.success)})
                return

            if name == "move_above_object":
                label = str(args["label"])
                r = float(args.get("r", 0.0))
                motion = str(args.get("motion_type", "linear"))
                # 1) find object(s)
                fo = self._act_find_object(label)
                # pick the first matching label (exact or substring)
                idx = next((i for i, s in enumerate(fo.labels) if label.lower() in s.lower()), 0)
                roi = fo.boxes[idx]
                # 2) pixels → robot (mm)
                rx, ry = self.roi_center_to_robot(roi)
                # 3) move at fixed z=0 (per requirement)
                mv = self._act_move_robot(rx, ry, 0.0, r, motion)
                self._return_fn_result(name, {"x": rx, "y": ry, "z": 0.0, "success": bool(mv.success)})
                return

            self.get_logger().warning(f"unknown tool: {name}")

        except Exception as e:
            self.get_logger().error(f"{name} failed: {e}")
            self._return_fn_result(name, {"error": str(e)})

    def _return_fn_result(self, name: str, payload: dict):
        """Send FunctionResponse back to the chat (completes the tool loop).  :contentReference[oaicite:7]{index=7}"""
        part = gt.Part(function_response=gt.FunctionResponse(name=name, response=payload))
        final = self.chat.send_message(part)  # includes thought signatures & history  :contentReference[oaicite:8]{index=8}
        if final.text:
            self.get_logger().info(final.text)

    # ===== ROS Action helpers =====
    def _act_find_object(self, query: str):
        self._find_object_result_event.clear()
        goal = FindObject.Goal(query=query)
        fut = self._find_object_client.send_goal_async(goal, feedback_callback=self._find_object_fb)
        fut.add_done_callback(self._find_object_goal_resp)
        self._find_object_result_event.wait(timeout=30.0)
        return self._find_object_result

    def _find_object_fb(self, fb):
        self.get_logger().info(f"find_object: {fb.feedback.status}")

    def _find_object_goal_resp(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error("find_object rejected")
            self._find_object_result_event.set()
            return
        gh.get_result_async().add_done_callback(self._find_object_get_result)

    def _find_object_get_result(self, fut):
        self._find_object_result = fut.result().result
        self._find_object_result_event.set()

    def _act_move_robot(self, x: float, y: float, z: float, r: float, motion_type: str):
        self._move_robot_result_event.clear()
        goal = MoveRobot.Goal(x=x, y=y, z=z, r=r, motion_type=motion_type)
        fut = self._move_robot_client.send_goal_async(goal)
        fut.add_done_callback(self._move_robot_goal_resp)
        self._move_robot_result_event.wait(timeout=20.0)
        return self._move_robot_result

    def _move_robot_goal_resp(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error("move_robot rejected")
            self._move_robot_result_event.set()
            return
        gh.get_result_async().add_done_callback(self._move_robot_get_result)

    def _move_robot_get_result(self, fut):
        self._move_robot_result = fut.result().result
        self._move_robot_result_event.set()


def main(args=None):
    rclpy.init(args=args)
    node = Orchestrator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
import pydobot
from pydobot.dobot import MODE_PTP
import glob
from serial.tools import list_ports
from final_interfaces.action import MoveRobot 
import time

class MockDobot:
    def __init__(self, port):
        print(f"Mock Dobot initialized (no real device connected)")
        
    def home(self):
        print("Mock: Homing device")
        
    def move_to(self, x, y, z, r=0, mode=None):
        print(f"Mock: Moving to x={x:.2f}, y={y:.2f}, z={z:.2f}, r={r:.2f}")

    def grip(self, state: bool):
        action = "Closing" if state else "Opening"
        print(f"Mock: {action} gripper")


class RobotControl(Node):
    def __init__(self):
        super().__init__('robot_control_server')

        # --- Hardware Initialization ---
        self.device = self._init_dobot()
        
        # --- SINGLE ACTION SERVER ---
        self._action_server = ActionServer(
            self,
            MoveRobot,
            'move_robot',
            execute_callback=self.execute_callback
        )

        # Home on startup
        self.home_robot()

    def _init_dobot(self):
        """Helper to find and connect to Dobot"""
        def _discover_acm_port() -> str | None:
            for p in list_ports.comports():
                device_path = getattr(p, "device", "") or ""
                desc = getattr(p, "description", "") or ""
                if "ACM" in device_path.upper() or "ACM" in desc.upper() or "DOBOT" in desc.upper():
                    return device_path

        port = _discover_acm_port()

        if port:
            try:
                device = pydobot.Dobot(port)
                self.get_logger().info(f"Dobot initialized on {port}")
                return device
            except Exception as e:
                self.get_logger().error(f"Failed to initialize Dobot on {port}: {e}")
        else:
            self.get_logger().error("Could not discover Dobot ACM port.")
        
        self.get_logger().info("Using MockDobot.")
        return MockDobot(port="/dev/ttyACM_MOCK")

    # --- HARDWARE WRAPPERS ---
    
    def home_robot(self):
        if not self.device: return
        try:
            self.device.home()
        except Exception as e:
            self.get_logger().error(f"Error homing: {e}")

    def move_robot(self, target):
        if not self.device: return
        mapping = {
            'joint': MODE_PTP.MOVJ_XYZ,
            'linear': MODE_PTP.MOVL_XYZ
        }
        if target.motion_type not in mapping:
            raise ValueError(f"Unknown motion type: {target.motion_type}")
        try:
            self.device.move_to(target.x, target.y, target.z, target.r, mode=mapping[target.motion_type])
            
        except Exception as e:
            self.get_logger().error(f"Error moving: {e}")
            raise e

    def grip_robot(self, state):
        if not self.device: return
        try:
            self.device.suck(state)
            time.sleep(2)  # Allow time for gripper action
        except Exception as e:
            self.get_logger().error(f"Error gripping: {e}")
            raise e

    # --- THE DISPATCHER (Main Logic) ---

    def execute_callback(self, goal_handle):
        self.get_logger().info('Received Goal Request')
        goal = goal_handle.request
        result = MoveRobot.Result()
        feedback = MoveRobot.Feedback()

        # 1. Check Device Health
        if self.device is None and not isinstance(self.device, MockDobot):
            goal_handle.abort()
            result.success = False
            result.message = "Device not initialized"
            return result

        try:
            # 2. SWITCH LOGIC based on 'command'
            
            # CASE A: HOME
            if goal.command == MoveRobot.Goal.CMD_HOME:
                feedback.status = "Homing..."
                goal_handle.publish_feedback(feedback)
                
                self.home_robot()
                
                result.message = "Homed successfully"

            # CASE B: MOVE
            elif goal.command == MoveRobot.Goal.CMD_MOVE:
                feedback.status = f"Moving to {goal.target.x}, {goal.target.y}..."
                goal_handle.publish_feedback(feedback)
                
                # We only look at 'goal.target' here
                self.move_robot(goal.target)
                
                result.message = "Movement complete"

            # CASE C: GRIP
            elif goal.command == MoveRobot.Goal.CMD_GRIP:
                state_str = "Closing" if goal.gripper_state else "Opening"
                feedback.status = f"{state_str} Gripper..."
                goal_handle.publish_feedback(feedback)
                
                # We only look at 'goal.gripper_state' here
                self.grip_robot(goal.gripper_state)
                
                result.message = f"Gripper {state_str}"

            # CASE D: UNKNOWN
            else:
                goal_handle.abort()
                result.success = False
                result.message = f"Unknown command ID: {goal.command}"
                return result

            # If we got here, it worked
            goal_handle.succeed()
            result.success = True
            return result

        except Exception as e:
            self.get_logger().error(f"Action failed: {e}")
            goal_handle.abort()
            result.success = False
            result.message = str(e)
            return result

def main(args=None):
    rclpy.init(args=args)
    node = RobotControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
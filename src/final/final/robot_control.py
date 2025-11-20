import rclpy
from rclpy.node import Node
import pydobot
from pydobot.dobot import MODE_PTP
from rclpy.action import ActionServer
from final_interfaces.action import MoveRobot, Gripper, Home
import glob
from serial.tools import list_ports

class MockDobot:
    def __init__(self, port):
        print(f"Mock Dobot initialized (no real device connected)")
        
    def home(self):
        print("Mock: Homing device")
        
    def move_to(self, x, y, z, r=0, mode=None):
        print(f"Mock: Moving to x={x:.2f}, y={y:.2f}, z={z:.2f}")

    def grip(self, state: bool):
        action = "Closing" if state else "Opening"
        print(f"Mock: {action} gripper")


class RobotControl(Node):
    def __init__(self):
        super().__init__('robot_control_action_server')

        # Discover the ACM port the Dobot is attached to, prefer serial list_ports
        def _discover_acm_port() -> str | None:
            try:
                for p in list_ports.comports():
                    # match typical ACM device path or descriptive hints
                    device_path = getattr(p, "device", "") or ""
                    desc = getattr(p, "description", "") or ""
                    if "ACM" in device_path.upper() or "ACM" in desc.upper() or "DOBOT" in desc.upper():
                        return device_path
            except Exception:
                pass
            # fallback to filesystem glob
            try:
                candidates = sorted(glob.glob("/dev/ttyACM*"))
                if candidates:
                    return candidates[0]
            except Exception:
                pass
            return None

        port = _discover_acm_port()
        if port:
            try:
                self.device = pydobot.Dobot(port)
                self.get_logger().info(f"Dobot initialized on {port}")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize Dobot on {port}: {e}")
                self.device = None
        else:
            self.get_logger().error("Could not discover Dobot ACM port (no /dev/ttyACM*).")
            self.get_logger().info("Using MockDobot for testing purposes.")
            self.device = MockDobot(port="/dev/ttyACM_MOCK")

        # Create the action server
        self._action_server = ActionServer(
            self,
            MoveRobot,
            'move_robot',
            execute_callback=self.execute_callback
        )

        self._action_server_gripper = ActionServer(
            self,
            Gripper,
            'gripper_control',
            execute_callback=self.grip
        )

        self._action_server_home = ActionServer(
            self,
            Home,
            'home_robot',
            execute_callback=self.home
        )

        # Home the robot on startup 
        self.home()

    def home(self):
        if self.device is None:
            self.get_logger().error("Device not available to home()")
            return
        try:
            self.device.home()
        except Exception as e:
            self.get_logger().error(f"Error homing device: {e}")

    def home_callback(self, goal_handle):
        self.get_logger().info('Received home goal')

        goal_handle.accepted()

        result = Home.Result()

        # Execute home action
        try:
            self.home()

            goal_handle.succeed()

            result.success = True
            return result

        except Exception as e:
            self.get_logger().error(f"Error executing home action: {e}")
            goal_handle.abort()

            result.success = False
            return result

    def move_to_joint_motion(self, x, y, z, r):
        if self.device is None:
            self.get_logger().error("Device not available for joint motion")
            return
        try:
            self.device.move_to(x, y, z, r, mode=MODE_PTP.MOVJ_XYZ)
        except Exception as e:
            self.get_logger().error(f"Error during joint motion: {e}")

    def move_to_linear_motion(self, x, y, z, r):
        if self.device is None:
            self.get_logger().error("Device not available for linear motion")
            return
        try:
            self.device.move_to(x, y, z, r, mode=MODE_PTP.MOVL_XYZ)
        except Exception as e:
            self.get_logger().error(f"Error during linear motion: {e}")

    def grip(self, on: bool):
        if self.device is None:
            self.get_logger().error("Device not available for gripper control")
            return
        try:
            if on:
                self.device.grip(True)
            else:
                self.device.grip(False)
        except Exception as e:
            self.get_logger().error(f"Error controlling gripper: {e}")

    def gripper_callback(self, goal_handle):
        self.get_logger().info('Received gripper goal')

        goal_handle.accepted()

        req = goal_handle.request
        gripper_state = req.gripper_state
        result = Gripper.Result()

        try:
            self.grip(gripper_state)
            goal_handle.succeed()
            result.success = True
            return result

        except Exception as e:
            self.get_logger().error(f"Error executing gripper action: {e}")
            goal_handle.abort()
            result.success = False
            return result

    def execute_callback(self, goal_handle):
        self.get_logger().info('Received move_robot goal')

        goal_handle.accepted()

        # Extract request
        req = goal_handle.request
        x = req.x
        y = req.y
        z = req.z
        r = req.r
        motion_type = req.motion_type

        # Publish initial feedback
        feedback = MoveRobot.Feedback()
        feedback.status = "Processing"

        goal_handle.publish_feedback(feedback)

        result = MoveRobot.Result()

        # Check device
        if self.device is None:
            self.get_logger().error("No Dobot device available; aborting goal")
            
            goal_handle.abort()
            result.success = False
            result.message = "Device not initialized"
            return result

        # Execute requested motion
        try:
            if motion_type == 'joint':
                self.move_to_joint_motion(x, y, z, r)
            elif motion_type == 'linear':
                self.move_to_linear_motion(x, y, z, r)
            else:
                self.get_logger().error('Invalid motion type specified.')
                goal_handle.abort()
                result.success = False
                result.message = f"Invalid motion_type: {motion_type}"
                return result

            feedback.status = "Motion commanded"
            goal_handle.publish_feedback(feedback)
            goal_handle.succeed()
            result.success = True
            result.message = "Motion completed / commanded"
            return result

        except Exception as e:
            self.get_logger().error(f"Error executing motion: {e}")
            goal_handle.abort()
            result.success = False
            result.message = f"Exception: {e}"
            return result

def main(args=None):
    rclpy.init(args=args)

    robot_control_action_server = RobotControl()

    try:
        rclpy.spin(robot_control_action_server)
    except KeyboardInterrupt:
        pass

    robot_control_action_server.destroy_node()
    rclpy.shutdown()


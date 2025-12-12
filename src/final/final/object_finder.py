import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, RegionOfInterest
from std_msgs.msg import String as StringMsg
import cv2
from cv_bridge import CvBridge
import threading
import numpy as np
from final_interfaces.action import FindObject
from rclpy.action import ActionServer
import sys, os
import time

# --- Google AI Imports ---
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    Blob
)
from pydantic import BaseModel

# --- CONSTANTS ---
api_key = os.getenv("GOOGLE_API_KEY")
# MODEL_ID = "gemini-robotics-er-1.5-preview"
MODEL_ID = "gemini-2.5-pro" 

# Load camera calibration
CAMERA_MATRIX = np.load('/home/ibraheem/ras545/midterm2/camera_matrix.npy')
DIST_COEFFS = np.load('/home/ibraheem/ras545/midterm2/dist_coeffs.npy')
CAMERA_PX_POINTS = np.load('/home/ibraheem/ras545/camera_points.npy').astype(np.float32)
ROBOT_MM_POINTS = np.load('/home/ibraheem/ras545/robot_points.npy').astype(np.float32)

# --- HELPER FUNCTIONS ---

def pixel_to_robot(pixel_x: float, pixel_y: float, correct_distortion: bool = True) -> tuple[float, float]:
    """Convert camera pixel coordinates to robot coordinates in mm."""
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    
    if correct_distortion:
        point = cv2.undistortPoints(point, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX)
        point = point.reshape(1, 1, 2)
    
    M = cv2.getPerspectiveTransform(CAMERA_PX_POINTS, ROBOT_MM_POINTS)
    transformed = cv2.perspectiveTransform(point, M)
    return tuple(transformed[0][0])

class ObjectPoint(BaseModel):
    point: list[int]
    label: str

def plot_points(image: np.ndarray, points: list[ObjectPoint]) -> None:
    """
    Draws crosshairs on the ORIGINAL resolution image, then resizes ONLY for display.
    """
    debug_image = image.copy()
    height, width = debug_image.shape[:2]
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 128, 0)
    ]

    for i, item in enumerate(points):
        # Normalize coordinates
        norm_y = item.point[0]
        norm_x = item.point[1]

        abs_y = int(norm_y / 1000 * height)
        abs_x = int(norm_x / 1000 * width)

        color = colors[i % len(colors)]

        # Draw Crosshair instead of circle (Better for checking centering)
        # MARKER_CROSS = + shape
        # markerSize = 20px
        # thickness = 2
        cv2.drawMarker(debug_image, (abs_x, abs_y), color, markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        
        if item.label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, width * 0.001)
            thickness = max(2, int(width * 0.003))
            
            # Label background/outline
            cv2.putText(debug_image, item.label, (abs_x + 15, abs_y - 10), font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(debug_image, item.label, (abs_x + 15, abs_y - 10), font, font_scale, color, thickness)

    # --- DISPLAY LOGIC ---
    display_h, display_w = debug_image.shape[:2]
    target_width = 1000
    
    if display_w > target_width:
        scale = target_width / display_w
        new_dim = (target_width, int(display_h * scale))
        display_img = cv2.resize(debug_image, new_dim, interpolation=cv2.INTER_AREA)
    else:
        display_img = debug_image

    window_name = "Detected Objects (Visual Debug)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(window_name, display_img)
    cv2.waitKey(7000) 
    # cv2.destroyAllWindows()

class ObjectFinder(Node):
    def __init__(self):
        super().__init__("object_finder_action_server")
        
        import numpy
        if numpy.__version__.startswith("2"):
             self.get_logger().fatal("NumPy 2.0 detected! Please run: pip install 'numpy<2'")

        self._action_server = ActionServer(self, FindObject, 'find_object', self.action_callback)
        self.subscription = self.create_subscription(Image, 'video_frames', self.frame_callback, 10)
        self.command_subscription = self.create_subscription(StringMsg, "finder_command", self.command_callback, 10)

        self.br = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        try:
            self.client = genai.Client(api_key=api_key, vertexai=False)
            
            # --- STRICT CENTER PROMPT ---
            self.config = GenerateContentConfig(
                system_instruction="""
                You are a precision object detection system for a robotic arm.
                
                YOUR TASK: Locate the GEOMETRIC CENTER (Centroid) of the requested objects.
                
                INSTRUCTIONS:
                1. Identify the object or objects requested.
                2. Return the coordinates of the EXACT CENTER of the object's top surface.
                3. Do NOT point to the edges, corners, or shadows. Point to the absolute middle.
                4. Label distinct items uniquely with numbers (e.g., "blue_block_1", "blue_block_2").
                
                The answer should follow the JSON format:
                [{"point": <point>, "label": <label1>}, ...]

                The points are in [y, x] format normalized to 0-1000.
                """,
                temperature=0.5, # Reduced slightly for more deterministic centering
                safety_settings=[
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    ),
                ],
                response_mime_type="application/json",
                response_schema=list[ObjectPoint],
            )
            self.get_logger().info("Google GenAI Client initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Google GenAI Client: {e}")
            self.client = None

    def command_callback(self, msg):
        if self.client is None: return
        self.get_logger().info(f"ObjectFinder received command: '{msg.data}'")
        current_frame = self.get_frame()
        self.find_object(msg.data, current_frame)

    def action_callback(self, goal_handle):
        self.get_logger().info("Received action request to find object.")
        
        current_frame = self.get_frame()
        query = goal_handle.request.query
        self.get_logger().info(f"Action query: '{query}'")

        feedback_msg = FindObject.Feedback()
        feedback_msg.status = "Processing"
        goal_handle.publish_feedback(feedback_msg)

        result_points = self.find_object(query, current_frame)
        result_msg = FindObject.Result()

        if not result_points or current_frame is None:
            goal_handle.abort()
            result_msg.boxes = []
            result_msg.x_coords = []
            result_msg.y_coords = []
            return result_msg

        boxes = []
        labels = []
        x_coords = []
        y_coords = []
        
        img_h, img_w = current_frame.shape[:2]

        for item in result_points:
            try:
                norm_y = item.point[0]
                norm_x = item.point[1]
                
                pixel_x = int(norm_x / 1000 * img_w)
                pixel_y = int(norm_y / 1000 * img_h)

            except Exception:
                continue

            rx, ry = pixel_to_robot(pixel_x, pixel_y, correct_distortion=True)
            self.get_logger().info(f"Object '{item.label}' Pixel({pixel_x},{pixel_y}) -> Robot({rx:.1f},{ry:.1f})")

            box_size = 20
            roi = RegionOfInterest()
            roi.x_offset = max(0, int(pixel_x - box_size/2))
            roi.y_offset = max(0, int(pixel_y - box_size/2))
            roi.width = box_size
            roi.height = box_size

            x_coords.append(float(rx))
            y_coords.append(float(ry))
            boxes.append(roi)
            labels.append(item.label if item.label else "object")

        result_msg.boxes = boxes
        result_msg.labels = labels
        result_msg.x_coords = x_coords
        result_msg.y_coords = y_coords

        self.get_logger().info(f"Found {len(boxes)} objects.")
        goal_handle.succeed()
        return result_msg
    
    def frame_callback(self, data):
        try:
            cv_frame = self.br.imgmsg_to_cv2(data, 'bgr8')
            with self.frame_lock:
                self.latest_frame = cv_frame
        except Exception as e:
            self.get_logger().error(f'Failed to convert frame: {e}')

    def get_frame(self):
        with self.frame_lock:
           if self.latest_frame is None:
               img = cv2.imread("/home/ibraheem/ras545final/test_images/20251120_152041.jpg")
           else:
               img = self.latest_frame.copy()

        if img is None:
            self.get_logger().error("Could not load image.")
            return None
        
        return img

    def find_object(self, query, current_frame) -> list[ObjectPoint] | None:
        if current_frame is None: return None
        try:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret: return None
            
            self.get_logger().info("Sending request to Gemini...")
            response = self.client.models.generate_content(
                model=MODEL_ID,
                contents=[Part(inline_data=Blob(data=buffer.tobytes(), mime_type="image/jpeg")), query],
                config=self.config,
            )

            if response.parsed:
                plot_points(current_frame, response.parsed)
                return response.parsed
            else:
                self.get_logger().info("No objects parsed.")
                return []

        except Exception as e:
            self.get_logger().error(f"AI Processing Error: {e}")
            return []

def main(args=None):
    rclpy.init(args=args)
    object_finder = ObjectFinder()
    try:
        rclpy.spin(object_finder)
    except KeyboardInterrupt:
        pass
    finally:
        object_finder.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
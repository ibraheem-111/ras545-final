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

# --- Google AI and Helper Imports ---
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    SafetySetting,
    Blob
)
from PIL import Image as PILImage, ImageColor, ImageDraw
from pydantic import BaseModel

# --- CONSTANTS ---
api_key = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.0-flash" 

# Load camera calibration
CAMERA_MATRIX = np.load('/home/ibraheem/ras545/midterm2/camera_matrix.npy')
DIST_COEFFS = np.load('/home/ibraheem/ras545/midterm2/dist_coeffs.npy')
CAMERA_PX_POINTS = np.load('/home/ibraheem/ras545/camera_points.npy').astype(np.float32)
ROBOT_MM_POINTS = np.load('/home/ibraheem/ras545/robot_points.npy').astype(np.float32)

def pixel_to_robot(pixel_x: float, pixel_y: float, correct_distortion: bool = True) -> tuple[float, float]:
    """Convert camera pixel coordinates to robot coordinates in mm."""
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    
    if correct_distortion:
        point = cv2.undistortPoints(point, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX)
        point = point.reshape(1, 1, 2)
    
    M = cv2.getPerspectiveTransform(CAMERA_PX_POINTS, ROBOT_MM_POINTS)
    transformed = cv2.perspectiveTransform(point, M)
    return tuple(transformed[0][0])

# --- DATA MODEL ---
class ObjectPoint(BaseModel):
    """
    Represents a specific point on an object with its label.
    Gemini returns points as [y, x] normalized to 0-1000.
    """
    point: list[int]
    label: str

# --- UPDATED VISUALIZATION (Fixes Temp File Issue) ---
def plot_points(image: np.ndarray, points: list[ObjectPoint]) -> None:
    """
    Plots points on an image with labels and SAVES to a static file.
    """
    im = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    width, height = im.size
    draw = ImageDraw.Draw(im)
    colors = list(ImageColor.colormap.keys())

    for i, item in enumerate(points):
        # Normalize coordinates:
        norm_y = item.point[0]
        norm_x = item.point[1]

        abs_y = int(norm_y / 1000 * height)
        abs_x = int(norm_x / 1000 * width)

        color = colors[i % len(colors)]

        # Draw the point as a small circle
        r = 4 
        draw.ellipse((abs_x - r, abs_y - r, abs_x + r, abs_y + r), fill=color)

        if item.label:
            # Position the text near the point
            draw.text((abs_x + 8, abs_y + 6), item.label, fill=color)

    # --- FIX START ---
    # Instead of im.show() which uses flaky temp files, we save to a specific path.
    # Open this file in VS Code/Image Viewer to see the result.
    output_path = "/home/ibraheem/ras545final/debug_detected_objects.jpg"
    try:
        im.save(output_path)
        print(f"Visual debug saved to: {output_path}")
    except Exception as e:
        print(f"Could not save debug image: {e}")
    # --- FIX END ---

class ObjectFinder(Node):
    def __init__(self):
        super().__init__("object_finder_action_server")

        self._action_server = ActionServer(
            self,
            FindObject,
            'find_object',
            self.action_callback
        )
        
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.frame_callback,
            10)

        self.command_subscription = self.create_subscription(
            StringMsg,
            "finder_command",
            self.command_callback,
            10)

        self.br = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # --- Initialize Google AI Client ---
        try:
            self.client = genai.Client(api_key=api_key, vertexai=False)
            
            # UPDATED SYSTEM INSTRUCTION
            self.config = GenerateContentConfig(
                system_instruction="""
                You are an object detection model for a robot arm's camera.
                The Robot's job is to pick and place coloured blocks.
                
                Given an image and a query, identify the object.
                
                CRITICAL RULE:
                If there are MULTIPLE instances of the requested object (e.g., multiple blue blocks),
                you MUST label them uniquely with a suffix number (e.g., "blue_block_1", "blue_block_2").
                
                The answer should follow the JSON format:
                [{"point": <point>, "label": <label1>}, ...]

                The points are in [y, x] format normalized to 0-1000.
                """,
                temperature=0.5,
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
            self.config = None

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

        # Feedback
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
            
            self.get_logger().info(f"Object '{item.label}' at pixel ({pixel_x}, {pixel_y}) -> robot ({rx:.1f} mm, {ry:.1f} mm)")

            # Create Dummy Bounding Box
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
        # --- STATIC IMAGE MODE ---
        return cv2.imread("/home/ibraheem/ras545final/test_images/20251120_152041.jpg")
        
        # --- REAL CAMERA MODE ---
        # with self.frame_lock:
        #     if self.latest_frame is None:
        #         return None
        #     return self.latest_frame.copy()

    def find_object(self, query, current_frame) -> list[ObjectPoint] | None:
        if current_frame is None: return None
        try:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret: return None
            
            self.get_logger().info("Sending request to Gemini...")
            response = self.client.models.generate_content(
                model=MODEL_ID,
                contents=[
                    Part(inline_data=Blob(data=buffer.tobytes(), mime_type="image/jpeg")),
                    query,
                ],
                config=self.config,
            )

            if response.parsed:
                # Plot points using the new visualization function
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
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
import numpy as np

# --- Google AI and Helper Imports ---

import sys, pkgutil, os
print("exe:", sys.executable)
print("first 10 sys.path:")
for p in sys.path[:10]: print("  ", p)

import google
print("\ngoogle.__file__:", getattr(google, "__file__", None))
print("google.__path__:", list(getattr(google, "__path__", [])))

from google import genai
from google.genai.types import (
    GenerateContentConfig,
    HarmBlockThreshold,
    HarmCategory,
    HttpOptions,
    Part,
    SafetySetting,
    Blob
)
from google.oauth2 import service_account, credentials
from PIL import Image as PILImage, ImageColor, ImageDraw
from pydantic import BaseModel

# # Load camera calibration from numpy files
CAMERA_MATRIX = np.load('/home/ibraheem/ras545/midterm2/camera_matrix.npy')
DIST_COEFFS = np.load('/home/ibraheem/ras545/midterm2/dist_coeffs.npy')

# Load calibration points from saved numpy files
CAMERA_PX_POINTS = np.load('/home/ibraheem/ras545/camera_points.npy').astype(np.float32)
ROBOT_MM_POINTS = np.load('/home/ibraheem/ras545/robot_points.npy').astype(np.float32)

def pixel_to_robot(pixel_x: float, pixel_y: float, correct_distortion: bool = True) -> tuple[float, float]:
    """Convert camera pixel coordinates to robot coordinates in mm.
    
    Args:
        pixel_x: x-coordinate in camera pixels
        pixel_y: y-coordinate in camera pixels
        correct_distortion: whether to apply camera distortion correction
    
    Returns:
        tuple[float, float]: (x, y) coordinates in robot space (mm)
    """
    point = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    
    # Step 1: Apply camera distortion correction if needed
    if correct_distortion:
        point = cv2.undistortPoints(point, CAMERA_MATRIX, DIST_COEFFS, P=CAMERA_MATRIX)
        point = point.reshape(1, 1, 2)
    
    # Step 2: Get perspective transform matrix
    M = cv2.getPerspectiveTransform(CAMERA_PX_POINTS, ROBOT_MM_POINTS)
    
    # Step 3: Convert point
    transformed = cv2.perspectiveTransform(point, M)
    
    return tuple(transformed[0][0])

class BoundingBox(BaseModel):
    """
    Represents a bounding box with its 2D coordinates and associated label.
    """
    box_2d: list[int]
    label: str

# Helper function to plot bounding boxes on an image
def plot_bounding_boxes(image: np.ndarray, bounding_boxes: list[BoundingBox]) -> None:
    """
    Plots bounding boxes on an image with labels, using PIL.
    Accepts a CV2 image (NumPy array) in BGR format.
    """
    # Convert CV2 BGR image to PIL RGB image
    im = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    width, height = im.size
    draw = ImageDraw.Draw(im)
    colors = list(ImageColor.colormap.keys())

    for i, bbox in enumerate(bounding_boxes):
        # Scale normalized coordinates (0-1000) to image dimensions
        abs_y_min = int(bbox.box_2d[0] / 1000 * height)
        abs_x_min = int(bbox.box_2d[1] / 1000 * width)
        abs_y_max = int(bbox.box_2d[2] / 1000 * height)
        abs_x_max = int(bbox.box_2d[3] / 1000 * width)

        color = colors[i % len(colors)]

        # Draw the rectangle using the correct (x, y) pairs
        draw.rectangle(
            ((abs_x_min, abs_y_min), (abs_x_max, abs_y_max)),
            outline=color,
            width=4,
        )
        if bbox.label:
            # Position the text at the top-left corner of the box
            draw.text((abs_x_min + 8, abs_y_min + 6), bbox.label, fill=color)

    im.show() # Display the image in a new window


class ObjectFinder(Node):
    def __init__(self):
        super().__init__("object_finder_action_server")

        self._action_server = ActionServer(
            self,
            FindObject,
            'find_object',
            self.action_callback
        )
        
        # Subscribes to the raw video feed
        self.subscription = self.create_subscription(
            Image,
            'video_frames',
            self.frame_callback, # Fast callback to just store the frame
            10) # Use QoS profile 10

        # Subscribes to natural language commands
        self.command_subscription = self.create_subscription(
            StringMsg,
            "finder_command",
            self.command_callback, # Slow callback that runs AI
            10)

        self.br = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # --- Initialize Google AI Client ---
        # NOTE: Set your GOOGLE_API_KEY environment variable
        try:
            self.client = genai.Client(http_options=HttpOptions(api_version="v1"))
            self.config = GenerateContentConfig(
                system_instruction="""
                You're an object detection model for a robot arm's camera.

The Robot's job is to pick and place coloured blocks on a table.

Given an image from the robot's camera and a text query asking for a specific colored block (e.g., 'red block', 'blue block'),

identify and locate the object in the image.

Return Tight bounding boxes as an array with labels.

Never return masks.
                """,
                temperature=0.5,
                safety_settings=[
                    SafetySetting(
                        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    ),
                ],
                response_mime_type="application/json",
                response_schema=list[BoundingBox],
            )
            self.get_logger().info("Google GenAI Client initialized successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize Google GenAI Client: {e}")
            self.client = None
            self.config = None
        # --- End Google AI Init ---

    def command_callback(self, msg):
        """
        This callback runs ONLY when a command is received.
        It performs the heavy object detection logic.
        """
        if self.client is None:
            self.get_logger().error("GenAI client not initialized. Cannot process command.")
            return

        self.get_logger().info(f"ObjectFinder received command: '{msg.data}'")
        
        current_frame = self.get_frame()
        
        self.find_object(msg.data, current_frame)

        

    def action_callback(self, goal_handle):
        self.get_logger().info("Received action request to find object.")
        # Accept the incoming goal
        try:
            goal_handle.accepted()
        except Exception:
            # Some rclpy versions expose `accept` instead of `accepted`.
            try:
                goal_handle.accept()
            except Exception:
                pass

        current_frame = self.get_frame()
        query = goal_handle.request.query

        self.get_logger().info(f"Action query: '{query}'")

        # Publish initial feedback
        feedback_msg = FindObject.Feedback()
        feedback_msg.status = "Processing"
        try:
            goal_handle.publish_feedback(feedback_msg)
        except Exception:
            # publish_feedback may raise on some versions if not supported; ignore
            pass

        result_bboxes = self.find_object(query, current_frame)

        result_msg = FindObject.Result()

        # If the AI didn't return anything, mark goal as aborted and return empty result
        if not result_bboxes:
            try:
                goal_handle.canceled()
            except Exception:
                try:
                    goal_handle.abort()
                except Exception:
                    pass
            result_msg.boxes = []
            result_msg.labels = []
            return result_msg

        # If there was no frame to analyze, abort the goal
        if current_frame is None:
            try:
                goal_handle.canceled()
            except Exception:
                try:
                    goal_handle.abort()
                except Exception:
                    pass
            result_msg.boxes = []
            result_msg.labels = []
            return result_msg

        # Convert parsed BoundingBox objects into sensor_msgs/RegionOfInterest and labels
        boxes = []
        labels = []
        x_coords = []
        y_coords = []
        img_h, img_w = current_frame.shape[:2]

        for bbox in result_bboxes:
            # Expect bbox.box_2d = [ymin, xmin, ymax, xmax] in 0..1000 (per plot code)
            try:
                y_min = int(bbox.box_2d[0] / 1000 * img_h)
                x_min = int(bbox.box_2d[1] / 1000 * img_w)
                y_max = int(bbox.box_2d[2] / 1000 * img_h)
                x_max = int(bbox.box_2d[3] / 1000 * img_w)
            except Exception:
                # If conversion fails, skip this box
                continue

            # y_center = (y_min + y_max) / 2
            # x_center = (x_min + x_max) / 2
            x_center, y_center = self.get_refined_center(
                current_frame, x_min, y_min, x_max, y_max
            )

            # Convert pixel coordinates to robot coordinates
            rx, ry = pixel_to_robot(x_center, y_center, correct_distortion=True)
            self.get_logger().info(f"Object '{bbox.label}' at pixel ({x_center:.1f}, {y_center:.1f}) -> robot ({rx:.1f} mm, {ry:.1f} mm)")

            roi = RegionOfInterest()
            roi.x_offset = max(0, x_min)
            roi.y_offset = max(0, y_min)
            roi.width = max(0, x_max - x_min)
            roi.height = max(0, y_max - y_min)

            x_coords.append(rx.item())
            y_coords.append(ry.item())
            boxes.append(roi)
            labels.append(bbox.label if getattr(bbox, 'label', None) is not None else "")

        self.get_logger().info(f"Result Coordinates: X: {x_coords}, Y: {y_coords}")


        result_msg.boxes = boxes
        result_msg.labels = labels
        result_msg.x_coords = x_coords
        result_msg.y_coords = y_coords

        self.get_logger().info(f"Result Coordinates: X: {x_coords}, Y: {y_coords}")

        # Final feedback and succeed the goal
        feedback_msg.status = f"Found {len(boxes)} objects"
        try:
            goal_handle.publish_feedback(feedback_msg)
        except Exception:
            pass

        try:
            goal_handle.succeed()
        except Exception:
            # some rclpy versions may use succeed(), others may not; ignore failures here
            pass

        return result_msg
    
    def frame_callback(self, data):
        """
        This callback runs for every frame.
        It just converts and saves the latest frame. No processing.
        """
        # self.get_logger().info("Receiving new frame...") # Too noisy, comment out
        try:
            cv_frame = self.br.imgmsg_to_cv2(data, 'bgr8')
            with self.frame_lock:
                self.latest_frame = cv_frame
        except Exception as e:
            self.get_logger().error(f'Failed to convert frame: {e}')

    def find_object(self, query, current_frame) -> list[BoundingBox] | None:
        try:
            # 1. Convert cv2 frame (NumPy) to JPEG bytes
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret:
                self.get_logger().error("Failed to encode frame to JPEG.")
                return
            image_bytes = buffer.tobytes()

            # 2. Create a GenAI Part from the image bytes
            image_part = Part(inline_data=Blob(data=image_bytes, mime_type="image/jpeg"))
            
            # 3. Get the text query
            user_query = query

            # 4. Call the model
            self.get_logger().info("Sending request to Google AI...")
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    image_part,
                    user_query,
                ],
                config=self.config,
            )

            self.get_logger().info(f"Received response: {response.text}")

            # 5. Plot the results
            if response.parsed:
                plot_bounding_boxes(current_frame, response.parsed)
            else:
                self.get_logger().info("No objects found or parsed by the model.")

            return response.parsed

        except Exception as e:
            self.get_logger().error(f"Error during AI processing: {e}")

    def get_frame(self):
        current_frame = None
        # with self.frame_lock:
        #     if self.latest_frame is None:
        #         self.get_logger().warning("ObjectFinder triggered but no frame received yet.")
        #         return
        #     # Make a copy to avoid race conditions while processing
        #     current_frame = self.latest_frame.copy()
        current_frame = cv2.imread("/home/ibraheem/ras545final/test_images/20251120_152041.jpg")
        return current_frame
    
    def get_refined_center(self, frame, x_min, y_min, x_max, y_max):
        """
        Refines the center by looking for the most 'saturated' (colorful) object
        inside the bounding box. This ignores wood grain.
        """
        # 1. Extract ROI
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return (x_min + x_max) / 2, (y_min + y_max) / 2

        # 2. Convert to HSV (Hue, Saturation, Value)
        # We only care about 'S' (Saturation) - index 1
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s_channel = hsv[:, :, 1]

        # 3. Thresholding
        # Otsu's method automatically finds the split between "dull" (wood) and "vibrant" (block)
        # This creates a binary mask where the block is White and wood is Black
        _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4. Morphology (Clean up noise)
        # This removes small speckles of noise (like wood knots)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # DEBUG: Uncomment this to save the mask image to check if it works
        # cv2.imwrite(f'debug_mask_{x_min}.jpg', mask)

        # 5. Find Contours on the MASK
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.get_logger().warn("No contours found in refinement. Using box center.")
            return (x_min + x_max) / 2, (y_min + y_max) / 2

        # 6. Find largest contour (the block)
        largest_contour = max(contours, key=cv2.contourArea)

        # 7. Calculate Center of Mass
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX_local = int(M["m10"] / M["m00"])
            cY_local = int(M["m01"] / M["m00"])
        else:
            return (x_min + x_max) / 2, (y_min + y_max) / 2

        # 8. Convert local ROI coords to global image coords
        final_x = x_min + cX_local
        final_y = y_min + cY_local

        return final_x, final_y

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
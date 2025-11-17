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
                Return bounding boxes as an array with labels.
                Never return masks. Limit to 25 objects.
                If an object is present multiple times, give each object a unique label
                according to its distinct characteristics (colors, size, position, etc..).
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
        with self.frame_lock:
            if self.latest_frame is None:
                self.get_logger().warning("ObjectFinder triggered but no frame received yet.")
                return
            # Make a copy to avoid race conditions while processing
            current_frame = self.latest_frame.copy()
        return current_frame

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
        query = goal_handle.query

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

            roi = RegionOfInterest()
            roi.x_offset = max(0, x_min)
            roi.y_offset = max(0, y_min)
            roi.width = max(0, x_max - x_min)
            roi.height = max(0, y_max - y_min)

            boxes.append(roi)
            labels.append(bbox.label if getattr(bbox, 'label', None) is not None else "")

        result_msg.boxes = boxes
        result_msg.labels = labels

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
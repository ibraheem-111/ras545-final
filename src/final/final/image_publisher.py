import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # ROS Image message
import cv2  # OpenCV library
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images

class ImagePublisher(Node):
    """
    Create an ImagePublisher class, which is a subclass of the Node class.
    This node will publish video frames from your webcam.
    """
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_publisher')

        # Create the publisher. This publisher will publish an Image
        # to the 'video_frames' topic. The queue size is 10 messages.
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)

        # We will publish a message every 0.1 seconds
        timer_period = 0.05  # seconds (for 20 FPS)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Create a VideoCapture object
        # The argument '0' gets the default webcam.
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def timer_callback(self):
        """
        Callback function.
        """
        ret, frame = self.cap.read()

        if ret:
            # Convert OpenCV image to ROS Image message
            # The 'bgr8' encoding is standard for OpenCV images
            ros_image = self.br.cv2_to_imgmsg(frame, 'bgr8')
            
            # Publish the image
            self.publisher_.publish(ros_image)

            # Log a message
            self.get_logger().info('Publishing video frame')
        else:
            self.get_logger().warning('Could not read frame from camera')

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_publisher = ImagePublisher()

    # Spin the node so the callback function is called.
    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        # (optional - otherwise it will be done automatically
        # when the garbage collector destroys the node object)
        image_publisher.cap.release() # Release the webcam
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
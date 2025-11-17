import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image  # ROS Image message
import cv2  # OpenCV library
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images

class ImageSubscriber(Node):
    """
    Create an ImageSubscriber class, which is a subclass of the Node class.
    This node will subscribe to video frames and display them.
    """
    def __init__(self):
        """
        Class constructor to set up the node
        """
        # Initiate the Node class's constructor and give it a name
        super().__init__('image_subscriber')

        # Create the subscriber. This subscriber will receive an Image
        # from the 'video_frames' topic.
        self.subscription = self.create_subscription(
            Image,
            'video_frames',  # Must match the publisher's topic name
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Used to convert between ROS and OpenCV images
        self.br = CvBridge()

    def listener_callback(self, data):
        """
        Callback function.
        """
        # Display the message on the console
        self.get_logger().info('Receiving video frame')

        # Convert ROS Image message to OpenCV image
        try:
            current_frame = self.br.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
            
        # Display image
        cv2.imshow("Camera Feed", current_frame)
        cv2.waitKey(1)  # This is crucial for cv2.imshow to work

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)

    # Create the node
    image_subscriber = ImageSubscriber()

    # Spin the node so the callback function is called.
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Destroy the node explicitly
        cv2.destroyAllWindows() # Close all OpenCV windows
        image_subscriber.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
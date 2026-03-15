import rclpy
from rclpy.node import Node


class G1ControllerNode(Node):
    def __init__(self):
        super().__init__("g1_controller")
        self.get_logger().info("g1_controller placeholder started.")


def main(args=None):
    rclpy.init(args=args)
    node = G1ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
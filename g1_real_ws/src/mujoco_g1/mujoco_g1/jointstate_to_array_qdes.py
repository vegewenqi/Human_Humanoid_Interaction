import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


class JointStateToArrayQdes(Node):
    """
    Convert safe JointState command from CBF layer to the 8-DoF
    Float32MultiArray q_des topic used by the MuJoCo controller or real g1 controller.

    Input:
        /joint_commands   sensor_msgs/JointState
        expected order:
        [waist_roll, waist_pitch, l_sh_pitch, l_sh_roll,
         l_elbow, r_sh_pitch, r_sh_roll, r_elbow]

    Output:
        /g1_upperbody_q_des_safe   std_msgs/Float32MultiArray
        same order as above
    """

    def __init__(self):
        super().__init__("jointstate_to_array_qdes")

        self.declare_parameter("input_topic", "/joint_commands")
        self.declare_parameter("output_topic", "/g1_upperbody_q_des_safe")

        self.declare_parameter(
            "joint_names",
            [
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_elbow_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_elbow_joint",
            ],
        )

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.joint_names = list(self.get_parameter("joint_names").value)

        self.name_to_index = {name: i for i, name in enumerate(self.joint_names)}
        self.expected_dim = len(self.joint_names)

        self.sub = self.create_subscription(
            JointState, self.input_topic, self.on_joint_command, 10
        )
        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)

        self.get_logger().info("JointStateToArrayQdes started.")
        self.get_logger().info(f"input_topic  = {self.input_topic}")
        self.get_logger().info(f"output_topic = {self.output_topic}")
        self.get_logger().info(f"joint_names  = {self.joint_names}")

    def on_joint_command(self, msg: JointState):
        if len(msg.name) == 0:
            self.get_logger().warn("Received /joint_commands with empty name field. Skip.")
            return

        if len(msg.position) == 0:
            self.get_logger().warn("Received /joint_commands with empty position field. Skip.")
            return

        incoming = {name: i for i, name in enumerate(msg.name)}

        q = []
        missing = []
        for name in self.joint_names:
            if name not in incoming:
                missing.append(name)
            else:
                idx = incoming[name]
                if idx >= len(msg.position):
                    missing.append(name)
                else:
                    q.append(float(msg.position[idx]))

        if missing:
            self.get_logger().warn(
                f"Missing joints in /joint_commands: {missing}. Skip."
            )
            return

        if len(q) != self.expected_dim:
            self.get_logger().warn(
                f"Expected {self.expected_dim} joint positions, got {len(q)}. Skip."
            )
            return

        out = Float32MultiArray()
        out.data = q
        self.pub.publish(out)


def main():
    rclpy.init()
    node = None
    try:
        node = JointStateToArrayQdes()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
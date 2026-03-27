import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState


class G1JointMapperNode(Node):
    """
    Input:
        /human_joint_angles_delta   Float32MultiArray
        order:
        [ torso_roll, torso_pitch, l_sh_pitch, l_sh_roll, l_el_pitch, r_sh_pitch, r_sh_roll, r_el_pitch ]

    Output:
        /g1_upperbody_q_des         Float32MultiArray
        order:
        [waist_roll, waist_pitch, left_shoulder_pitch, left_shoulder_roll, left_elbow, right_shoulder_pitch, right_shoulder_roll, right_elbow]
    """

    def __init__(self):
        super().__init__("g1_joint_mapper")

        # topic names
        self.declare_parameter("input_topic", "/human_joint_angles_delta")
        self.declare_parameter("output_topic", "/g1_upperbody_q_des")

        # units
        self.declare_parameter("input_in_degrees", True)
        self.declare_parameter("output_in_degrees", False)

        # logging
        self.declare_parameter("log_output", "both")   # human | qdes | both

        # robot joint names (aligned with g1_controller.py and XML)
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

        # robot home pose q_home
        # home position: stand up straight with arms down
        self.declare_parameter(
            "q_home",
            [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708]
        )

        # direction sign s_j
        # initial recommendation: keep all +1 first, then flip individual entries if needed
        self.declare_parameter(
            "signs",
            [-1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
        )

        # scale gain g_j
        # initial recommendation: 1-to-1 angle mapping
        self.declare_parameter(
            "gains",
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )

        # extra bias b_j
        self.declare_parameter(
            "bias",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )

        # controller-aligned joint limits
        # aligned with the g1_controller.py defaults for g1_mjx.xml
        # (-30-30, -30-30, -177-153, -90-130, -60-120, -177-153, -130,90, -60-120)
        self.declare_parameter(
            "q_min",
            [-0.52, -0.52, -3.0892, -1.5882, -1.0472, -3.0892, -2.2515, -1.0472]
        )
        self.declare_parameter(
            "q_max",
            [0.52, 0.52, 2.6704, 2.2515, 2.0944, 2.6704, 1.5882, 2.0944]
        )

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.input_in_degrees = bool(self.get_parameter("input_in_degrees").value)
        self.output_in_degrees = bool(self.get_parameter("output_in_degrees").value)
        self.log_output = str(self.get_parameter("log_output").value).strip().lower()

        self.joint_names = list(self.get_parameter("joint_names").value)
        self.q_home = np.array(self.get_parameter("q_home").value, dtype=np.float64)
        self.signs = np.array(self.get_parameter("signs").value, dtype=np.float64)
        self.gains = np.array(self.get_parameter("gains").value, dtype=np.float64)
        self.bias = np.array(self.get_parameter("bias").value, dtype=np.float64)
        self.q_min = np.array(self.get_parameter("q_min").value, dtype=np.float64)
        self.q_max = np.array(self.get_parameter("q_max").value, dtype=np.float64)

        self.expected_dim = 8
        for arr_name, arr in [
            ("q_home", self.q_home),
            ("signs", self.signs),
            ("gains", self.gains),
            ("bias", self.bias),
            ("q_min", self.q_min),
            ("q_max", self.q_max),
        ]:
            if arr.shape[0] != self.expected_dim:
                raise ValueError(f"{arr_name} must have length {self.expected_dim}")

        self.sub = self.create_subscription(
            Float32MultiArray, self.input_topic, self.on_delta_angles, 10
        )
        self.pub = self.create_publisher(Float32MultiArray, self.output_topic, 10)

        self.declare_parameter("unsafe_joint_command_topic", "/joint_commands_unsafe")
        self.unsafe_joint_command_topic = str(self.get_parameter("unsafe_joint_command_topic").value)
        self.pub_unsafe_jointstate = self.create_publisher(JointState, self.unsafe_joint_command_topic, 10)

        self.last_log_time = self.get_clock().now()

        self.get_logger().info("G1JointMapperNode started.")
        self.get_logger().info(f"input_topic  = {self.input_topic}")
        self.get_logger().info(f"output_topic = {self.output_topic}")
        self.get_logger().info(f"joint_names  = {self.joint_names}")
        self.get_logger().info(f"q_home       = {self.q_home.tolist()}")
        self.get_logger().info(f"signs        = {self.signs.tolist()}")
        self.get_logger().info(f"gains        = {self.gains.tolist()}")

    def _maybe_deg_to_rad(self, x: np.ndarray) -> np.ndarray:
        if self.input_in_degrees:
            return np.deg2rad(x)
        return x

    def _maybe_rad_to_deg(self, x: np.ndarray) -> np.ndarray:
        if self.output_in_degrees:
            return np.rad2deg(x)
        return x

    def on_delta_angles(self, msg: Float32MultiArray):
        if len(msg.data) != self.expected_dim:
            self.get_logger().warn(
                f"Expected {self.expected_dim} angles, got {len(msg.data)}. Skip."
            )
            return

        theta_delta = np.array(msg.data, dtype=np.float64)
        theta_delta = self._maybe_deg_to_rad(theta_delta)

        # q_des = q_home + s * g * delta + b
        q_des = self.q_home + self.signs * self.gains * theta_delta + self.bias
        q_des = np.clip(q_des, self.q_min, self.q_max)

        out = Float32MultiArray()
        out_data = self._maybe_rad_to_deg(q_des)
        out.data = [float(x) for x in out_data]
        self.pub.publish(out)

        # Publish JointState for 8-DoF CBF node
        q_des_cbf = np.array(
            [
                q_des[0],  # waist_roll
                q_des[1],  # waist_pitch
                q_des[2],  # left_shoulder_pitch
                q_des[3],  # left_shoulder_roll
                q_des[4],  # left_elbow
                q_des[5],  # right_shoulder_pitch
                q_des[6],  # right_shoulder_roll
                q_des[7],  # right_elbow
            ],
            dtype=np.float64,
        )
        msg_js = JointState()
        msg_js.header.stamp = self.get_clock().now().to_msg()
        msg_js.name = [
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_elbow_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_elbow_joint",
        ]
        msg_js.position = [float(x) for x in q_des_cbf]
        msg_js.velocity = []
        msg_js.effort = []
        self.pub_unsafe_jointstate.publish(msg_js)

        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds * 1e-9 > 1.0:
            if self.log_output in ["human", "both"]:
                human_disp = np.rad2deg(theta_delta)
                self.get_logger().info(
                    "[human_delta_deg] "
                    f"torso_roll={human_disp[0]:.2f}, torso_pitch={human_disp[1]:.2f}, "
                    f"l_sh_pitch={human_disp[2]:.2f}, l_sh_roll={human_disp[3]:.2f}, l_el_pitch={human_disp[4]:.2f}, "
                    f"r_sh_pitch={human_disp[5]:.2f}, r_sh_roll={human_disp[6]:.2f}, r_el_pitch={human_disp[7]:.2f}"
                )

            if self.log_output in ["qdes", "both"]:
                q_disp = np.rad2deg(q_des)
                self.get_logger().info(
                    "[g1_q_des_deg] "
                    f"waist_roll={q_disp[0]:.2f}, waist_pitch={q_disp[1]:.2f}, "
                    f"l_sh_pitch={q_disp[2]:.2f}, l_sh_roll={q_disp[3]:.2f}, l_elbow={q_disp[4]:.2f}, "
                    f"r_sh_pitch={q_disp[5]:.2f}, r_sh_roll={q_disp[6]:.2f}, r_elbow={q_disp[7]:.2f}"
                )

            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = G1JointMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
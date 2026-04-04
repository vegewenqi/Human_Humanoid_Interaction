import math
from typing import Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray


def rotx(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s,  c],
    ], dtype=np.float64)


def roty(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float64)


def rotz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c = math.cos(a)
    s = math.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    q = q / n
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_from_z_to_vec(v: np.ndarray) -> Tuple[float, float, float, float]:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)

    v_unit = v / v_norm
    c = float(np.dot(z_axis, v_unit))

    if c > 1.0 - 1e-9:
        return (0.0, 0.0, 0.0, 1.0)

    if c < -1.0 + 1e-9:
        return (1.0, 0.0, 0.0, 0.0)

    axis = np.cross(z_axis, v_unit)
    s = math.sqrt((1.0 + c) * 2.0)
    invs = 1.0 / s

    qx = axis[0] * invs
    qy = axis[1] * invs
    qz = axis[2] * invs
    qw = 0.5 * s
    return (float(qx), float(qy), float(qz), float(qw))


class HumanCapsuleFrameTransform(Node):
    """
    Unified transform node with two modes.

    mode = "sim":
      input must be /human_capsules_local
      p_target = R_place * (R_align * p_local) + t_place

    mode = "real":
      input must be /human_capsules_zed
      p_target = R_extrinsic * p_zed + t_extrinsic

    Output topic is unified for downstream CBF:
      /human_capsules_robot
    """

    def __init__(self):
        super().__init__("human_capsule_frame_transform")

        self.declare_parameter("mode", "sim")  # sim | real

        self.declare_parameter("input_topic", "/human_capsules_local")
        self.declare_parameter("output_topic", "/human_capsules_robot")
        self.declare_parameter("marker_topic", "/human_capsules_markers_robot")
        self.declare_parameter("target_frame", "pelvis")

        # SIM mode: fixed axis alignment
        self.declare_parameter("align_roll_deg", 0.0)
        self.declare_parameter("align_pitch_deg", 0.0)
        self.declare_parameter("align_yaw_deg", 0.0)

        # SIM mode: placement of human pelvis in robot frame
        self.declare_parameter("tx", 0.0)
        self.declare_parameter("ty", 0.8)
        self.declare_parameter("tz", 0.0)
        self.declare_parameter("yaw_deg", 0.0)

        # REAL mode: extrinsic T_robot_from_zed
        self.declare_parameter("extrinsic_tx", 0.0)
        self.declare_parameter("extrinsic_ty", 0.0)
        self.declare_parameter("extrinsic_tz", 0.0)
        self.declare_parameter("extrinsic_qx", 0.0)
        self.declare_parameter("extrinsic_qy", 0.0)
        self.declare_parameter("extrinsic_qz", 0.0)
        self.declare_parameter("extrinsic_qw", 1.0)

        self.mode = str(self.get_parameter("mode").value).strip().lower()
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.marker_topic = str(self.get_parameter("marker_topic").value)
        self.target_frame = str(self.get_parameter("target_frame").value)

        # sim params
        self.align_roll_deg = float(self.get_parameter("align_roll_deg").value)
        self.align_pitch_deg = float(self.get_parameter("align_pitch_deg").value)
        self.align_yaw_deg = float(self.get_parameter("align_yaw_deg").value)

        self.tx = float(self.get_parameter("tx").value)
        self.ty = float(self.get_parameter("ty").value)
        self.tz = float(self.get_parameter("tz").value)
        self.yaw_deg = float(self.get_parameter("yaw_deg").value)

        self.R_align = (
            rotz(self.align_yaw_deg) @
            roty(self.align_pitch_deg) @
            rotx(self.align_roll_deg)
        )
        self.R_place = rotz(self.yaw_deg)
        self.t_place = np.array([self.tx, self.ty, self.tz], dtype=np.float64)

        # real params
        self.R_extrinsic = quat_to_rot(
            float(self.get_parameter("extrinsic_qx").value),
            float(self.get_parameter("extrinsic_qy").value),
            float(self.get_parameter("extrinsic_qz").value),
            float(self.get_parameter("extrinsic_qw").value),
        )
        self.t_extrinsic = np.array([
            float(self.get_parameter("extrinsic_tx").value),
            float(self.get_parameter("extrinsic_ty").value),
            float(self.get_parameter("extrinsic_tz").value),
        ], dtype=np.float64)

        self.sub_caps = self.create_subscription(
            Float32MultiArray,
            self.input_topic,
            self.on_capsules,
            10,
        )

        self.pub_caps = self.create_publisher(
            Float32MultiArray,
            self.output_topic,
            10,
        )

        self.pub_markers = self.create_publisher(
            MarkerArray,
            self.marker_topic,
            10,
        )

        self.get_logger().info("HumanCapsuleFrameTransform started.")
        self.get_logger().info(f"mode         = {self.mode}")
        self.get_logger().info(f"input_topic  = {self.input_topic}")
        self.get_logger().info(f"output_topic = {self.output_topic}")
        self.get_logger().info(f"marker_topic = {self.marker_topic}")
        self.get_logger().info(f"target_frame = {self.target_frame}")

        if self.mode == "sim":
            self.get_logger().info(
                f"sim alignment(deg): roll={self.align_roll_deg:.1f}, pitch={self.align_pitch_deg:.1f}, yaw={self.align_yaw_deg:.1f}"
            )
            self.get_logger().info(
                f"sim placement: tx={self.tx:.3f}, ty={self.ty:.3f}, tz={self.tz:.3f}, yaw_deg={self.yaw_deg:.1f}"
            )
        elif self.mode == "real":
            self.get_logger().info(
                f"real extrinsic t=({self.t_extrinsic[0]:.3f}, {self.t_extrinsic[1]:.3f}, {self.t_extrinsic[2]:.3f})"
            )
        else:
            self.get_logger().warn(f"Unknown mode '{self.mode}', expected 'sim' or 'real'.")

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        if self.mode == "sim":
            p_aligned = self.R_align @ p
            return self.R_place @ p_aligned + self.t_place
        elif self.mode == "real":
            return self.R_extrinsic @ p + self.t_extrinsic
        else:
            return p.copy()

    def on_capsules(self, msg: Float32MultiArray):
        data = np.array(msg.data, dtype=np.float64)
        if data.size == 0:
            return

        if data.size % 7 != 0:
            self.get_logger().warn(
                f"Expected capsule flat array length multiple of 7, got {data.size}"
            )
            return

        n_caps = data.size // 7
        out = data.copy()

        for i in range(n_caps):
            s = 7 * i
            block = data[s:s + 7]

            if not np.all(np.isfinite(block[:6])):
                out[s:s + 7] = block
                continue

            a_in = block[0:3]
            b_in = block[3:6]
            r = block[6]

            a_t = self.transform_point(a_in)
            b_t = self.transform_point(b_in)

            out[s:s + 7] = np.array([
                a_t[0], a_t[1], a_t[2],
                b_t[0], b_t[1], b_t[2],
                r,
            ], dtype=np.float64)

        out_msg = Float32MultiArray()
        out_msg.data = out.astype(np.float32).tolist()
        self.pub_caps.publish(out_msg)

        self.publish_markers(out, n_caps)

    def publish_markers(self, flat_caps: np.ndarray, n_caps: int):
        ma = MarkerArray()

        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        ma.markers.append(delete_all)

        for i in range(n_caps):
            s = 7 * i
            block = flat_caps[s:s + 7]

            if not np.all(np.isfinite(block[:6])):
                continue

            a = block[0:3]
            b = block[3:6]
            r = float(block[6])

            c = 0.5 * (a + b)
            v = b - a
            length = float(np.linalg.norm(v))
            if length < 1e-6:
                continue

            qx, qy, qz, qw = quat_from_z_to_vec(v)

            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.target_frame
            m.ns = "human_capsules_robot"
            m.id = i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD

            m.pose.position.x = float(c[0])
            m.pose.position.y = float(c[1])
            m.pose.position.z = float(c[2])
            m.pose.orientation.x = qx
            m.pose.orientation.y = qy
            m.pose.orientation.z = qz
            m.pose.orientation.w = qw

            m.scale.x = 2.0 * r
            m.scale.y = 2.0 * r
            m.scale.z = length
            # ToDo: add the human skeleton line markers for better visualization
            if i == 0:
                m.color.r = 0.2
                m.color.g = 0.9
                m.color.b = 0.2
                m.color.a = 0.3
            elif i in [1, 2]:
                m.color.r = 0.85
                m.color.g = 0.45
                m.color.b = 0.35
                m.color.a = 0.3
            elif i in [3, 4]:
                m.color.r = 0.25
                m.color.g = 0.45
                m.color.b = 0.95
                m.color.a = 0.3
            elif i == 5:
                m.color.r = 0.9
                m.color.g = 0.9
                m.color.b = 0.2
                m.color.a = 0.3
            else:
                m.color.r = 0.45
                m.color.g = 0.85
                m.color.b = 0.85
                m.color.a = 0.3

            ma.markers.append(m)

        self.pub_markers.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = HumanCapsuleFrameTransform()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
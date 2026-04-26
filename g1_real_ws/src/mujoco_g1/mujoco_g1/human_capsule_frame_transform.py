import math
import time
from typing import Optional, Tuple

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
    Unified transform node with three modes.

    mode = "sim":
      input must be /human_capsules_local
      p_target = R_place * (R_align * p_local) + t_place

    mode = "real_cali":
      input must be /human_capsules_zed
      p_target = R_extrinsic * p_zed + t_extrinsic

    mode = "real_quick_cali":
      input must be /human_capsules_zed
      subscribe pelvis from /human_pelvis_point_zed
      after startup_delay_sec, average first N pelvis frames:
          p_pelvis_ref = mean(pelvis_zed[0:N])
      then:
          p_target = R_extrinsic * (p_zed - p_pelvis_ref) + t_extrinsic

      Here extrinsic_* keeps the SAME parameter names, but means:
      place the averaged initial human pelvis into robot pelvis frame.
    """

    def __init__(self):
        super().__init__("human_capsule_frame_transform")

        self.declare_parameter("mode", "sim")  # sim | real_cali | real_quick_cali

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

        # REAL_CALI / REAL_QUICK_CALI mode:
        # keep exact same interface names
        self.declare_parameter("extrinsic_tx", 0.0)
        self.declare_parameter("extrinsic_ty", 0.0)
        self.declare_parameter("extrinsic_tz", 0.0)
        self.declare_parameter("extrinsic_qx", 0.0)
        self.declare_parameter("extrinsic_qy", 0.0)
        self.declare_parameter("extrinsic_qz", 0.0)
        self.declare_parameter("extrinsic_qw", 1.0)

        # foot marker params
        self.declare_parameter("enable_footprint_marker", True)
        self.declare_parameter("ground_z", 0.0)
        self.declare_parameter("footprint_size", 0.18)
        self.declare_parameter("footprint_line_width", 0.012)

        # quick calibration params
        self.declare_parameter("pelvis_topic", "/human_pelvis_point_zed")
        self.declare_parameter("bootstrap_num_frames", 50)
        self.declare_parameter("startup_delay_sec", 5.0)

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

        self.pelvis_topic = str(self.get_parameter("pelvis_topic").value)
        self.bootstrap_num_frames = int(self.get_parameter("bootstrap_num_frames").value)
        self.startup_delay_sec = float(self.get_parameter("startup_delay_sec").value)
        
        self.enable_footprint_marker = bool(
            self.get_parameter("enable_footprint_marker").value
        )
        self.ground_z = float(self.get_parameter("ground_z").value)
        self.footprint_size = float(self.get_parameter("footprint_size").value)
        self.footprint_line_width = float(
            self.get_parameter("footprint_line_width").value
        )
        
        if self.bootstrap_num_frames <= 0:
            self.get_logger().warn(
                f"bootstrap_num_frames={self.bootstrap_num_frames} invalid, reset to 50"
            )
            self.bootstrap_num_frames = 50
        if self.startup_delay_sec < 0.0:
            self.get_logger().warn(
                f"startup_delay_sec={self.startup_delay_sec} invalid, reset to 5.0"
            )
            self.startup_delay_sec = 5.0

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

        # quick calibration state
        self.sub_pelvis = None
        self.bootstrap_done = False
        self.bootstrap_samples = []
        self.pelvis_ref_zed: Optional[np.ndarray] = None
        self.quick_cali_start_time: Optional[float] = None
        self.quick_cali_started = False

        if self.mode == "real_quick_cali":
            self.sub_pelvis = self.create_subscription(
                Float32MultiArray,
                self.pelvis_topic,
                self.on_pelvis,
                10,
            )
            self.quick_cali_start_time = time.time() + self.startup_delay_sec

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

        elif self.mode == "real_cali":
            self.get_logger().info(
                f"real_cali extrinsic t=({self.t_extrinsic[0]:.3f}, {self.t_extrinsic[1]:.3f}, {self.t_extrinsic[2]:.3f})"
            )

        elif self.mode == "real_quick_cali":
            self.get_logger().info(
                f"real_quick_cali pelvis_topic = {self.pelvis_topic}"
            )
            self.get_logger().info(
                f"real_quick_cali bootstrap_num_frames = {self.bootstrap_num_frames}"
            )
            self.get_logger().info(
                f"real_quick_cali startup_delay_sec = {self.startup_delay_sec:.2f}"
            )
            self.get_logger().info(
                "real_quick_cali meaning of extrinsic_*: place averaged initial human pelvis into robot pelvis frame."
            )
            self.get_logger().info(
                f"real_quick_cali placement t=({self.t_extrinsic[0]:.3f}, {self.t_extrinsic[1]:.3f}, {self.t_extrinsic[2]:.3f})"
            )

        else:
            self.get_logger().warn(
                f"Unknown mode '{self.mode}', expected 'sim', 'real_cali', or 'real_quick_cali'."
            )

    def on_pelvis(self, msg: Float32MultiArray):
        if self.mode != "real_quick_cali":
            return

        now = time.time()
        if self.quick_cali_start_time is None or now < self.quick_cali_start_time:
            return

        if not self.quick_cali_started:
            self.quick_cali_started = True
            self.get_logger().info(
                f"Startup delay finished. Collecting {self.bootstrap_num_frames} valid pelvis frames..."
            )

        if self.bootstrap_done:
            return

        data = np.array(msg.data, dtype=np.float64)
        if data.size < 3:
            return

        p = data[:3]
        if not np.all(np.isfinite(p)):
            return

        self.bootstrap_samples.append(p.copy())
        n = len(self.bootstrap_samples)

        if n == 1 or n == self.bootstrap_num_frames or n % 5 == 0:
            self.get_logger().info(
                f"Pelvis capture progress: {n}/{self.bootstrap_num_frames}"
            )

        if n >= self.bootstrap_num_frames:
            stack = np.stack(self.bootstrap_samples, axis=0)
            self.pelvis_ref_zed = np.mean(stack, axis=0)
            self.bootstrap_done = True

            self.get_logger().info(
                "Pelvis reference captured from averaged frames: "
                f"[{self.pelvis_ref_zed[0]:.4f}, "
                f"{self.pelvis_ref_zed[1]:.4f}, "
                f"{self.pelvis_ref_zed[2]:.4f}]"
            )

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        if self.mode == "sim":
            p_aligned = self.R_align @ p
            return self.R_place @ p_aligned + self.t_place

        elif self.mode == "real_cali":
            return self.R_extrinsic @ p + self.t_extrinsic

        elif self.mode == "real_quick_cali":
            if self.pelvis_ref_zed is None:
                return p.copy()
            return self.R_extrinsic @ (p - self.pelvis_ref_zed) + self.t_extrinsic

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

        if self.mode == "real_quick_cali":
            if not self.bootstrap_done or self.pelvis_ref_zed is None:
                now = time.time()
                if self.quick_cali_start_time is not None and now < self.quick_cali_start_time:
                    remain = self.quick_cali_start_time - now
                    self.get_logger().warn(
                        f"real_quick_cali waiting startup delay... {remain:.1f}s remaining.",
                        throttle_duration_sec=1.0,
                    )
                else:
                    self.get_logger().warn(
                        "real_quick_cali collecting pelvis frames...",
                        throttle_duration_sec=1.0,
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
            is_sphere = length < 1e-6

            m = Marker()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.target_frame
            m.ns = "human_capsules_robot"
            m.id = i
            m.action = Marker.ADD

            if is_sphere:
                m.type = Marker.SPHERE
                m.pose.position.x = float(a[0])
                m.pose.position.y = float(a[1])
                m.pose.position.z = float(a[2])
                m.pose.orientation.x = 0.0
                m.pose.orientation.y = 0.0
                m.pose.orientation.z = 0.0
                m.pose.orientation.w = 1.0

                m.scale.x = 2.0 * r
                m.scale.y = 2.0 * r
                m.scale.z = 2.0 * r
            else:
                qx, qy, qz, qw = quat_from_z_to_vec(v)

                m.type = Marker.CYLINDER
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

            if i == 0:
                m.color.r = 0.2
                m.color.g = 0.9
                m.color.b = 0.2
                m.color.a = 0.5
            elif i in [1, 2]:
                m.color.r = 0.25
                m.color.g = 0.45
                m.color.b = 0.95
                m.color.a = 0.5
            elif i in [3, 4]:
                m.color.r = 0.85
                m.color.g = 0.45
                m.color.b = 0.35
                m.color.a = 0.5
            elif i in [5, 7]:
                m.color.r = 0.45
                m.color.g = 0.85
                m.color.b = 0.85
                m.color.a = 0.5
            elif i in [6, 8]:
                m.color.r = 0.9
                m.color.g = 0.9
                m.color.b = 0.2
                m.color.a = 0.5
            else:
                # head sphere
                m.color.r = 0.95
                m.color.g = 0.75
                m.color.b = 0.80
                m.color.a = 0.5

            ma.markers.append(m)

        if self.enable_footprint_marker:
            self._append_human_footprint_markers(ma, flat_caps)

        self.pub_markers.publish(ma)

    def _append_human_footprint_markers(self, ma: MarkerArray, flat_caps: np.ndarray):
        """
        Add a ground crosshair and vertical line for the human location.

        Uses capsule 0 = torso, whose endpoints are pelvis and neck in
        HumanSkeletonCapsuleNode.CAPSULE_ORDER.
        """
        if flat_caps.size < 7:
            return

        torso = flat_caps[0:7]
        if not np.all(np.isfinite(torso[:6])):
            return

        # In human_skeleton_capsule.py, torso = (pelvis, neck, radius).
        # After transform, torso[0:3] is the human pelvis-like lower torso point.
        p = torso[0:3].copy()
        ground = np.array([p[0], p[1], self.ground_z], dtype=np.float64)

        stamp = self.get_clock().now().to_msg()
        base_id = 1000
        half = 0.5 * self.footprint_size

        # Vertical line: human pelvis/torso lower point down to ground
        m_line = Marker()
        m_line.header.stamp = stamp
        m_line.header.frame_id = self.target_frame
        m_line.ns = "human_footprint"
        m_line.id = base_id
        m_line.type = Marker.LINE_LIST
        m_line.action = Marker.ADD
        m_line.scale.x = float(self.footprint_line_width)
        m_line.color.r = 1.0
        m_line.color.g = 1.0
        m_line.color.b = 1.0
        m_line.color.a = 0.9
        m_line.points.append(self._point_msg(p))
        m_line.points.append(self._point_msg(ground))
        ma.markers.append(m_line)

        # Crosshair on ground
        m_cross = Marker()
        m_cross.header.stamp = stamp
        m_cross.header.frame_id = self.target_frame
        m_cross.ns = "human_footprint"
        m_cross.id = base_id + 1
        m_cross.type = Marker.LINE_LIST
        m_cross.action = Marker.ADD
        m_cross.scale.x = float(self.footprint_line_width)
        m_cross.color.r = 1.0
        m_cross.color.g = 1.0
        m_cross.color.b = 0.0
        m_cross.color.a = 1.0

        p1 = ground + np.array([ half, 0.0, 0.0])
        p2 = ground + np.array([-half, 0.0, 0.0])
        p3 = ground + np.array([0.0,  half, 0.0])
        p4 = ground + np.array([0.0, -half, 0.0])
        m_cross.points.append(self._point_msg(p1))
        m_cross.points.append(self._point_msg(p2))
        m_cross.points.append(self._point_msg(p3))
        m_cross.points.append(self._point_msg(p4))
        ma.markers.append(m_cross)

        # Small flat disk at ground point
        m_disk = Marker()
        m_disk.header.stamp = stamp
        m_disk.header.frame_id = self.target_frame
        m_disk.ns = "human_footprint"
        m_disk.id = base_id + 2
        m_disk.type = Marker.CYLINDER
        m_disk.action = Marker.ADD
        m_disk.pose.position.x = float(ground[0])
        m_disk.pose.position.y = float(ground[1])
        m_disk.pose.position.z = float(ground[2] + 0.003)
        m_disk.pose.orientation.x = 0.0
        m_disk.pose.orientation.y = 0.0
        m_disk.pose.orientation.z = 0.0
        m_disk.pose.orientation.w = 1.0
        m_disk.scale.x = float(self.footprint_size)
        m_disk.scale.y = float(self.footprint_size)
        m_disk.scale.z = 0.006
        m_disk.color.r = 1.0
        m_disk.color.g = 1.0
        m_disk.color.b = 0.0
        m_disk.color.a = 0.35
        ma.markers.append(m_disk)

    @staticmethod
    def _point_msg(p: np.ndarray):
        from geometry_msgs.msg import Point
        return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))


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
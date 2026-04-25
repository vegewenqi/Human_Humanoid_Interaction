import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation


class CalibrationKinematics:
    """Lightweight FK for calibration only.

    Differences from the main G1Kinematics:
    - no Jacobian computation
    - no collision pairs
    - only tracks one calibration tag point rigidly attached to torso_link
    """

    def __init__(self, urdf_path: str,
                 tag_frame: str = 'torso_link',
                 tag_offset_xyz: np.ndarray = None,
                 tag_offset_rot: Rotation = None):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        if tag_offset_xyz is None:
            tag_offset_xyz = np.array([0.08, 0.0, 0.125], dtype=np.float64)
        if tag_offset_rot is None:
            tag_offset_rot = Rotation.identity()

        self.tag_frame = tag_frame
        self.tag_frame_id = self.model.getFrameId(tag_frame)
        if self.tag_frame_id >= self.model.nframes:
            raise ValueError(f"Frame '{tag_frame}' not found in URDF")

        self.tag_offset_se3 = pin.SE3(
            tag_offset_rot.as_matrix(),
            np.asarray(tag_offset_xyz, dtype=np.float64)
        )

        self.q_full = pin.neutral(self.model)

    def update(self, q_full: np.ndarray):
        self.q_full = q_full.copy()
        pin.forwardKinematics(self.model, self.data, self.q_full)
        pin.updateFramePlacements(self.model, self.data)

    def get_tag_pose(self):
        """Return calibration tag center in pelvis/world frame used by URDF FK."""
        oMf = self.data.oMf[self.tag_frame_id]
        T_world = oMf * self.tag_offset_se3
        return T_world.translation.copy(), T_world.rotation.copy()

    def joint_names_to_q_full(self, names: list, positions: list) -> np.ndarray:
        q = pin.neutral(self.model)
        for name, pos in zip(names, positions):
            try:
                jid = self.model.getJointId(name)
                if jid < self.model.njoints:
                    idx_q = self.model.joints[jid].idx_q
                    q[idx_q] = pos
            except Exception:
                pass
        return q
# /mujoco_g1_ik/components/utils.py
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2



# PointCloud2 to numpy array conversion
def pc2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Robust PointCloud2 -> (N,3) float32 conversion."""
    try:
        arr = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"))
        if arr is None:
            raise ValueError("read_points_numpy returned None")

        # structured dtype
        if hasattr(arr, "dtype") and arr.dtype.fields is not None:
            pts = np.empty((arr.shape[0], 3), dtype=np.float32)
            pts[:, 0] = arr["x"]
            pts[:, 1] = arr["y"]
            pts[:, 2] = arr["z"]
            return pts

        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return arr.astype(np.float32, copy=False)

        raise ValueError(f"Unexpected numpy shape/dtype: shape={arr.shape}, dtype={arr.dtype}")

    except Exception:
        gen = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=False)
        return np.array([(float(p[0]), float(p[1]), float(p[2])) for p in gen], dtype=np.float32)
    

# SE3 translation helper
def se3_translation(x) -> np.ndarray:
    """Return translation as (3,) np.float64, compatible with mink SE3 variants."""
    t = x.translation
    if callable(t):
        t = t()
    return np.asarray(t, dtype=np.float64)
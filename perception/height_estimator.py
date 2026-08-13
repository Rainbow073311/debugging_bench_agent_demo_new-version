"""
改进几何高度估计 v2

原理:
  摄像头看桌面上的物体:
  - 物体底部 (桌上) → 像素 u_b, 单应性 → 世界坐标 B=(bx, by, table_z) ✓ 正确
  - 物体顶部 (高度 h) → 像素 u_t, 单应性 → 世界坐标 P=(px, py, table_z) ✗ 偏后

  摄像头位置 C=(cx, cy, cz), 桌面 Z=table_z, 摄像头高度 H=|cz - table_z|

  从摄像头看物体底部: 方向 d_b = normalize(B - C)
  从摄像头看物体顶部: 方向 d_t = normalize(P' - C)  其中 P' 是顶部在桌面上的投影

  相似三角形:
    物体高度 h / 投影偏移 |P - B| = 摄像头高度 H / (摄像头水平距离 + 投影偏移)
    → h = |P - B| * H / (|B_xy - C_xy| + |P - B|)

  实际场景中 |P - B| << |B_xy - C_xy|, 近似:
    → h ≈ |P - B| * H / |B_xy - C_xy|
"""
import numpy as np
import math, yaml, os


class HeightEstimator:
    def __init__(self, camera_config_path):
        with open(camera_config_path, "r") as f:
            data = yaml.safe_load(f)
        self.H = np.array(data["table_homography"]["H"])
        self.table_z = data["table_homography"]["table_z_mm"]
        extrinsics = data.get("extrinsics", {})
        end_camera = extrinsics.get("T_end_to_camera", {})
        self.camera_offset = np.array(
            end_camera.get("t_mm", [0, 0, 0]), dtype=np.float64
        )
        self.cam_t = None
        self.cam_height = 670.0  # 实测镜头距桌面 mm

        K = np.array(data["intrinsics"]["camera_matrix"])
        self.fx, self.fy = K[0,0], K[1,1]
        self.cx_px, self.cy_px = K[0,2], K[1,2]

    def set_robot_pose(self, pose):
        """Bind camera XY/Z via Trans(TCP)@Rz(J1)@t_ec; flange R ignored."""
        xyz = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            raise ValueError("robot XYZ pose must be finite")
        if "j1_deg" in pose and pose["j1_deg"] is not None:
            j1 = float(pose["j1_deg"])
        elif "j1" in pose and pose["j1"] is not None:
            j1 = float(pose["j1"])
        else:
            j1 = math.degrees(math.atan2(float(xyz[1]), float(xyz[0])))
        ang = math.radians(j1)
        c, s = math.cos(ang), math.sin(ang)
        ox, oy, oz = self.camera_offset
        offset_base = np.array(
            [c * ox - s * oy, s * ox + c * oy, oz], dtype=np.float64
        )
        self.cam_t = xyz + offset_base

    def estimate(self, obstacle, blob_pixels, robot_pose=None):
        """估算障碍物高度 (mm) — 经验法: 像素面积 × 距离补偿"""
        if not blob_pixels or len(blob_pixels) < 10:
            return 60.0
        if robot_pose is not None:
            self.set_robot_pose(robot_pose)
        if self.cam_t is None:
            raise ValueError(
                "robot XYZ pose is required for eye-in-hand height estimation"
            )

        bx, by = obstacle["x"], obstacle["y"]
        area = obstacle.get("pixel_area", len(blob_pixels))

        # 物体越远, 像素面积越小 → 需要距离补偿
        cx_cam, cy_cam = self.cam_t[0], self.cam_t[1]
        dist_h = math.hypot(bx - cx_cam, by - cy_cam)

        # 标定: 距摄像头 ~300mm 处, 80mm 高物体 ≈ 多少像素面积
        # 用户可调: 放一个已知高度物体, 运行 scan, 根据输出反算
        # 公式: height = area * dist_h / K_factor
        # K_factor ≈ 15000 (经验值, 需实测校准)

        K_FACTOR = 110000  # 实测校准: 160mm物体 ≈ 28300px² at ~625mm距离
        h = area * dist_h / K_FACTOR
        return max(10, min(300, round(h, 1)))

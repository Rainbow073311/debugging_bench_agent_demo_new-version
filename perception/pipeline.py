"""
Main perception-planning pipeline orchestrating all modules.

Coordinates mask generation, depth estimation, 3D reconstruction,
obstacle grid mapping, and path planning into a unified workflow.

Architecture:
    CameraFrame → MaskGenerator → DepthEstimator → ReconstructionPipeline
    → ObstacleGridMap → Planner (A*/RRT/RRT-Connect) → Executor
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from perception.domain_types import (
    CameraFrame,
    ObstacleMask as ObstacleMaskType,
    DepthMap as DepthMapType,
    PlanPath,
    ExecutionStep,
)
from perception.mask_generator import (
    MaskConfig as MaskGenConfig,
    BackgroundSubtractionMasker,
    ArmExclusionMasker,
    CombinedMaskGenerator,
    ObstacleMask as MaskGenObstacleMask,
)
from perception.depth_estimator import (
    BaseDepthEstimator,
    PlaneFittingAligner,
    DepthMap as DepthEstDepthMap,
)
from perception.reconstruction import (
    ReconstructionPipeline as ReconstructionPipe,
    ObstacleInfo as ReconObstacleInfo,
    PointCloud as ReconPointCloud,
)
from perception.obstacle_map import (
    ObstacleGridMap,
    GridCell,
    ObstacleInfo as MapObstacleInfo,
    compute_workspace_from_camera_view,
)
from perception.planner import (
    AStar2DPlanner,
    RRTPlanner,
    RRTConnectPlanner,
    CollisionChecker,
    Executor,
    LinkCapsule,
)
from perception.config import get_default_config, load_config
from perception.height_estimator import HeightEstimator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Container for intermediate and final results from one pipeline cycle."""

    frame: Optional[CameraFrame] = None
    mask: Optional[MaskGenObstacleMask] = None
    depth: Optional[DepthEstDepthMap] = None
    obstacles: List[ReconObstacleInfo] = field(default_factory=list)
    map_summary: Optional[Dict[str, Any]] = None
    path: Optional[PlanPath] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mask_shape": self.mask.mask.shape if self.mask is not None else None,
            "depth_shape": self.depth.depth.shape if self.depth is not None else None,
            "obstacle_count": len(self.obstacles),
            "map_summary": self.map_summary,
            "path_length": len(self.path.waypoints) if self.path else 0,
            "diagnostics": self.diagnostics,
            "processing_time_ms": self.processing_time_ms,
            "success": self.success,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# PerceptionPlanningPipeline
# ---------------------------------------------------------------------------

class PerceptionPlanningPipeline:
    """
    Orchestrates perception modules (mask, depth, reconstruction, grid map)
    and the motion planner into a single coherent workflow.

    Usage::

        pipeline = PerceptionPlanningPipeline(config_path="config.yaml")
        result = pipeline.capture_and_process(frame)
        path = pipeline.plan(start_pose, goal_pose)
        steps = pipeline.execute(path)
    """

    def __init__(
        self,
        config_path_or_dict: Optional[str | Path | dict] = None,
        camera_config_path: Optional[str | Path] = None,
    ) -> None:
        # Load configuration
        if config_path_or_dict is None:
            self.config = get_default_config()
        elif isinstance(config_path_or_dict, dict):
            self.config = get_default_config()
            self._merge_config(self.config, config_path_or_dict)
        else:
            self.config = load_config(str(config_path_or_dict))

        # Optionally load camera calibration from camera_config.yaml
        if camera_config_path is not None:
            self._load_camera_calibration(str(camera_config_path))

        # Camera intrinsics as 3x3 numpy
        K_flat = self.config.camera.intrinsics_k
        self.K: np.ndarray = np.array(K_flat, dtype=float).reshape(3, 3)
        self._camera_config_path = camera_config_path

        # Pipeline state
        self._frame_count: int = 0
        self._planning_success_count: int = 0
        self._planning_total_count: int = 0
        self._last_result: Optional[PipelineResult] = None
        self._start_time = time.monotonic()

        # Initialise sub-modules (lazy for heavy ones)
        self._mask_generator: Optional[CombinedMaskGenerator] = None
        self._depth_estimator: Optional[BaseDepthEstimator] = None
        self._reconstructor: Optional[ReconstructionPipe] = None
        self._obstacle_map: Optional[ObstacleGridMap] = None
        self._astar_planner: Optional[AStar2DPlanner] = None
        self._rrt_planner: Optional[RRTPlanner] = None
        self._collision_checker: Optional[CollisionChecker] = None
        self._executor: Optional[Executor] = None
        self._height_estimator: Optional[HeightEstimator] = None

        logger.info("PerceptionPlanningPipeline initialised (lazy init).")

    # -- config helper --------------------------------------------------------

    @staticmethod
    def _merge_config(base: Any, overrides: dict) -> None:
        """Recursively merge *overrides* into dataclass *base*."""
        for key, val in overrides.items():
            if hasattr(base, key):
                attr = getattr(base, key)
                if isinstance(val, dict) and hasattr(attr, "__dataclass_fields__"):
                    PerceptionPlanningPipeline._merge_config(attr, val)
                else:
                    setattr(base, key, val)

    def _load_camera_calibration(self, yaml_path: str) -> None:
        """Load camera intrinsics, resolution and table params from *camera_config.yaml*.

        Overrides the corresponding fields in the pipeline config.
        """
        import yaml as _yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            cal = _yaml.safe_load(f)

        # Intrinsics
        if "intrinsics" in cal and "camera_matrix" in cal["intrinsics"]:
            K = cal["intrinsics"]["camera_matrix"]
            self.config.camera.intrinsics_k = [
                K[0][0], K[0][1], K[0][2],
                K[1][0], K[1][1], K[1][2],
                K[2][0], K[2][1], K[2][2],
            ]
            logger.info(f"Loaded intrinsics from {yaml_path}")

        # Distortion
        if "intrinsics" in cal and "distortion_coeffs" in cal["intrinsics"]:
            self.config.camera.dist_coeffs = list(cal["intrinsics"]["distortion_coeffs"])

        # Resolution
        if "calibration" in cal and "resolution" in cal["calibration"]:
            w, h = cal["calibration"]["resolution"]
            self.config.camera.width = w
            self.config.camera.height = h

        # Table Z (for reconstruction)
        if "table_homography" in cal and "table_z_mm" in cal["table_homography"]:
            self.config.reconstruction.table_z_mm = cal["table_homography"]["table_z_mm"]

        # XYZ-only eye-in-hand camera offset (never a fixed base/camera pose)
        if "extrinsics" in cal:
            ext = cal["extrinsics"]
            if "T_end_to_camera" in ext and "t_mm" in ext["T_end_to_camera"]:
                self._camera_offset_in_end_mm = ext["T_end_to_camera"]["t_mm"]

        # Arm exclusion zones
        if "arm_exclusion" in cal and "zones" in cal["arm_exclusion"]:
            zones = cal["arm_exclusion"]["zones"]
            if zones:
                self.config.mask.arm_exclusion_zones = zones

    # -- lazy sub-module factories -------------------------------------------

    def _get_mask_generator(self) -> CombinedMaskGenerator:
        if self._mask_generator is None:
            mc = self.config.mask
            mask_cfg = MaskGenConfig(
                yolo_model_path=mc.yolo_model_path or "",
                yolo_conf_thresh=getattr(mc, "yolo_conf_thresh", 0.10),  # 默认 0.10
                yolo_classes=getattr(mc, "yolo_classes", None),
                blur_kernel_size=mc.blur_k,
                threshold=mc.thresh,
                min_contour_area=float(mc.min_area),
                arm_exclusion_zones=None,  # handled via arm_masker below
            )
            bg_masker = BackgroundSubtractionMasker(mask_cfg)
            arm_masker = None
            if mc.arm_exclusion_zones:
                zones = [np.array(z, dtype=np.float64) for z in mc.arm_exclusion_zones]
                arm_masker = ArmExclusionMasker(arm_zones=zones)
            self._mask_generator = CombinedMaskGenerator(
                config=mask_cfg,
                arm_masker=arm_masker,
                bg_masker=bg_masker,
            )
        return self._mask_generator

    def _get_depth_estimator(self) -> BaseDepthEstimator:
        if self._depth_estimator is None:
            # Try Depth Anything V3 ONNX - NO FALLBACK for debugging
            from perception.depth_estimator import DA3OnnxEstimator
            try:
                self._depth_estimator = DA3OnnxEstimator(self.config.depth)
                logger.info("Using Depth Anything 3 ONNX for depth estimation")
            except Exception as e:
                # Raise instead of fallback - we need to know why DA3 fails
                import traceback
                error_msg = f"DA3 ONNX FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        return self._depth_estimator

    def _get_reconstructor(self) -> ReconstructionPipe:
        if self._reconstructor is None:
            rc = self.config.reconstruction
            recon_cfg = {
                "grid_size_mm": self.config.planning.grid_resolution_mm,
                "height_percentiles": list(rc.height_percentiles),
                "min_points_per_cluster": rc.min_points_count,
            }
            self._reconstructor = ReconstructionPipe(
                config=recon_cfg,
                K=self.K,
                table_z=rc.table_z_mm,
            )
        return self._reconstructor

    def _get_obstacle_map(self) -> ObstacleGridMap:
        if self._obstacle_map is None:
            mc = self.config.map
            # Default workspace: 800x600 mm centred at origin
            bounds = (-400.0, 400.0, -300.0, 300.0)
            gmap = ObstacleGridMap(
                workspace_bounds=bounds,
                cell_size_mm=mc.cell_size_mm,
            )
            # Apply dilation and safety margins
            gmap.dilate_footprint(mc.footprint_dilation_mm)
            gmap.dilate_height(mc.height_safety_margin_mm)
            gmap.apply_uncertainty_safety(mc.uncertainty_dilation_factor)
            self._obstacle_map = gmap
        return self._obstacle_map

    def _get_astar_planner(self, costmap: np.ndarray) -> AStar2DPlanner:
        return AStar2DPlanner(costmap, self.config.map.cell_size_mm)

    def _get_rrt_planner(self) -> RRTPlanner:
        if self._rrt_planner is None:
            bounds = ((-400.0, 400.0), (-300.0, 300.0))
            # Convert grid-map occupied cells to 2D boxes for RRT collision
            obs_boxes: List[Any] = []
            gmap = self._get_obstacle_map()
            for iy in range(gmap.ny):
                for ix in range(gmap.nx):
                    if gmap.grid[iy, ix].occupied:
                        x_min = gmap.x_min + ix * gmap.cell_size_mm
                        y_min = gmap.y_min + iy * gmap.cell_size_mm
                        x_max = x_min + gmap.cell_size_mm
                        y_max = y_min + gmap.cell_size_mm
                        obs_boxes.append(
                            (np.array([x_min, y_min]), np.array([x_max, y_max]))
                        )
            self._rrt_planner = RRTPlanner(workspace_bounds=bounds, obstacles=obs_boxes)
        return self._rrt_planner

    def _get_collision_checker(self) -> CollisionChecker:
        if self._collision_checker is None:
            pc = self.config.planning
            links = [
                LinkCapsule(
                    name=c["name"],
                    p1_mm=np.array(c["p1"], dtype=float),
                    p2_mm=np.array(c["p2"], dtype=float),
                    radius_mm=float(c["radius"]),
                )
                for c in pc.arm_link_capsules
            ]
            grid_stub = _PlannerGridStub(self._get_obstacle_map())
            self._collision_checker = CollisionChecker(arm_links=links, obstacle_map=grid_stub)
        return self._collision_checker

    def _get_executor(self) -> Executor:
        if self._executor is None:
            self._executor = Executor(
                config={
                    "step_size_mm": self.config.planning.step_size_mm,
                    "re_perception_interval_mm": self.config.planning.re_perception_every_mm,
                },
                collision_checker=self._get_collision_checker(),
                perception_pipeline=self,
            )
        return self._executor

    # ================================================================== #
    #  Core pipeline step                                                #
    # ================================================================== #

    def capture_and_process(self, frame: np.ndarray, background_frame: Optional[np.ndarray] = None) -> PipelineResult:
        """
        Run one full perception cycle on a single camera frame.

        Parameters
        ----------
        frame : np.ndarray
            HxWx3 BGR or RGB image.
        background_frame : np.ndarray, optional
            Reference background for subtraction.  If None the masker
            must have been initialised via ``update_background`` already.

        Returns
        -------
        PipelineResult with all intermediate outputs.
        """
        t_start = time.monotonic()
        self._frame_count += 1
        result = PipelineResult()

        try:
            # Step 1: Generate obstacle mask
            masker = self._get_mask_generator()
            if background_frame is not None:
                masker.bg_masker.update_background(background_frame)
            mask = masker.process(frame)
            result.mask = mask
            binary_mask = mask.mask  # HxW uint8

            # Step 2: Estimate depth
            depth_est = self._get_depth_estimator()
            depth_result = depth_est.estimate(frame)

            # Handle both old (DepthMap) and new (tuple) return types
            if isinstance(depth_result, tuple):
                depth_map, raw_depth, table_median = depth_result
                result.depth = depth_map
                depth_array = depth_map.depth
            else:
                depth_map = depth_result
                result.depth = depth_map
                depth_array = depth_map.depth

            # Step 3: Reconstruct 2.5D/3D obstacles
            recon = self._get_reconstructor()
            # obstacle_mask = binary mask > 0
            obstacle_mask_arr = (binary_mask > 0).astype(np.uint8)
            obstacles = recon.process(depth_array, obstacle_mask_arr)
            result.obstacles = obstacles

            # Step 4: Update obstacle grid map
            gmap = self._get_obstacle_map()
            map_obstacles = _recon_to_map_obstacles(obstacles)
            if map_obstacles:
                gmap.update(map_obstacles)
            result.map_summary = _summarise_map(gmap)

            # Step 5: Diagnostics
            result.diagnostics = self.get_diagnostics()
            result.success = True

        except Exception as exc:
            logger.error(f"Pipeline error at frame {self._frame_count}: {exc}", exc_info=True)
            result.success = False
            result.error_message = str(exc)
            result.diagnostics = self.get_diagnostics()

        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        result.processing_time_ms = elapsed_ms
        self._last_result = result

        logger.info(
            f"Frame {self._frame_count} processed in {elapsed_ms:.1f} ms — "
            f"success={result.success}, obstacles={len(result.obstacles)}"
        )
        return result

    # ================================================================== #
    #  Planning                                                          #
    # ================================================================== #

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        use_rrt: bool = False,
    ) -> Optional[PlanPath]:
        """
        Compute a collision-free path from start to goal.

        Parameters
        ----------
        start_xy : (x_mm, y_mm)
        goal_xy : (x_mm, y_mm)
        use_rrt : if True use RRT, otherwise A* on costmap.

        Returns
        -------
        PlanPath or None if no feasible path found.
        """
        self._planning_total_count += 1

        try:
            if use_rrt:
                rrt = self._get_rrt_planner()
                grid_pts = rrt.plan(start_xy, goal_xy)
            else:
                gmap = self._get_obstacle_map()
                costmap = gmap.get_costmap()
                astar = self._get_astar_planner(costmap)
                grid_pts = astar.plan(start_xy, goal_xy)

            if grid_pts is None:
                logger.warning("No valid path found.")
                return None

            # Wrap into PlanPath
            waypoints = [
                {"x": float(p[0]), "y": float(p[1])} for p in grid_pts
            ]
            cost = sum(
                np.linalg.norm(np.array(grid_pts[i]) - np.array(grid_pts[i - 1]))
                for i in range(1, len(grid_pts))
            )
            path = PlanPath(waypoints=waypoints, collision_free=True, cost=float(cost))
            self._planning_success_count += 1
            logger.info(f"  Path found: {len(waypoints)} waypoints, length={cost:.1f} mm")
            return path

        except Exception as exc:
            logger.error(f"Planning failed: {exc}", exc_info=True)
            return None

    # ================================================================== #
    #  Execution                                                         #
    # ================================================================== #

    def execute(
        self,
        path: Optional[PlanPath],
    ) -> List[ExecutionStep]:
        """Execute a planned path with periodic re-perception."""
        if path is None or not path.waypoints:
            logger.warning("No valid path to execute.")
            return []

        executor = self._get_executor()
        # Convert waypoints back to np arrays for executor
        np_path = [np.array([wp["x"], wp["y"]]) for wp in path.waypoints]
        return executor.execute_path(np_path)

    # ================================================================== #
    #  Diagnostics                                                       #
    # ================================================================== #

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return pipeline health metrics."""
        uptime = time.monotonic() - self._start_time
        planning_rate = (
            self._planning_success_count / self._planning_total_count
            if self._planning_total_count > 0
            else 0.0
        )

        diag: Dict[str, Any] = {
            "uptime_seconds": round(uptime, 2),
            "frames_processed": self._frame_count,
            "planning_success_rate": round(planning_rate, 4),
            "planning_total": self._planning_total_count,
            "planning_success": self._planning_success_count,
        }

        if self._last_result:
            diag["last_mask_confidence"] = (
                self._last_result.mask.confidence if self._last_result.mask else None
            )
            diag["last_obstacle_count"] = len(self._last_result.obstacles)
            diag["last_processing_time_ms"] = self._last_result.processing_time_ms

        return diag

    # -- external API for Executor replanning callback ------------------------

    def detect_new_obstacles(self) -> List[Dict[str, Any]]:
        """Stub: called by Executor during re-perception."""
        # In a real system this would capture a new frame and compare.
        return []

    def replan(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """Replan from current position to goal (used by Executor)."""
        result = self.plan((float(start[0]), float(start[1])),
                           (float(goal[0]), float(goal[1])))
        if result is None:
            return None
        return [np.array([wp["x"], wp["y"]]) for wp in result.waypoints]

    def reset_statistics(self) -> None:
        """Reset runtime statistics."""
        self._frame_count = 0
        self._planning_success_count = 0
        self._planning_total_count = 0
        self._last_result = None
        logger.info("Pipeline statistics reset.")


# ---------------------------------------------------------------------------
# Fallback depth estimator (works without deep-learning backends)
# ---------------------------------------------------------------------------

class _FallbackDepthEstimator(BaseDepthEstimator):
    """
    Minimal depth estimator that uses a table-mask heuristic to produce
    a pseudo-depth map.  Suitable for testing and pipeline integration
    when no torch / depth-anything models are installed.

    It generates a relative depth map from image gradient magnitude
    (brighter = closer heuristic) and aligns it via plane fitting.
    """

    def __init__(self, depth_config: Any):
        self.config = depth_config
        self._aligner = PlaneFittingAligner()

    def estimate(self, rgb_image: np.ndarray) -> DepthEstDepthMap:
        gray = rgb_image if rgb_image.ndim == 2 else rgb_image.mean(axis=2)
        H, W = gray.shape

        # Simple heuristic: use vertical position as proxy for relative depth
        # (lower in image = closer to camera = smaller depth value)
        rel = np.linspace(1.0, 0.2, H).reshape(H, 1).repeat(W, axis=1).astype(np.float32)

        # Add slight variation from image brightness
        rel = rel + (1.0 - gray.astype(np.float32) / 255.0) * 0.1

        # Crude table mask: bottom-centre region
        table_mask = np.zeros((H, W), dtype=bool)
        table_mask[int(0.7 * H):, int(0.15 * W):int(0.85 * W)] = True

        known_table_z = 0.0  # table at Z=0 in world frame

        aligned, residual, uncertainty = self._aligner.align_depth_to_plane(
            rel, table_mask, known_table_z, np.eye(3)
        )

        return DepthEstDepthMap(
            depth=aligned,
            uncertainty=uncertainty,
            residual=residual,
            is_metric=True,
        )


# ---------------------------------------------------------------------------
# Adapter helpers
# ---------------------------------------------------------------------------

def _recon_to_map_obstacles(
    obstacles: List[ReconObstacleInfo],
) -> List[MapObstacleInfo]:
    """Convert reconstruction ObstacleInfo to obstacle_map ObstacleInfo."""
    result: List[MapObstacleInfo] = []
    for obs in obstacles:
        # Build 3D footprint from the obstacle's footprint hull + height range
        hull = obs.footprint  # (K, 2)
        if hull.shape[0] == 0:
            continue
        # Create 3D points: base ring + top ring
        base_pts = np.column_stack([hull, np.full(hull.shape[0], obs.base_z)])
        top_pts = np.column_stack([hull, np.full(hull.shape[0], obs.top_z)])
        pts_3d = np.vstack([base_pts, top_pts])
        result.append(MapObstacleInfo(
            footprint_3d=pts_3d,
            confidence=obs.confidence,
            label=f"obstacle_{obs.cluster_id}",
        ))
    return result


def _summarise_map(gmap: ObstacleGridMap) -> Dict[str, Any]:
    """Return a compact summary of the grid map."""
    occ = sum(1 for iy in range(gmap.ny) for ix in range(gmap.nx) if gmap.grid[iy, ix].occupied)
    return {
        "grid_shape": (gmap.ny, gmap.nx),
        "cell_size_mm": gmap.cell_size_mm,
        "occupied_cells": occ,
        "total_cells": gmap.ny * gmap.nx,
    }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_pipeline(
    config: Optional[str | Path | dict] = None,
    camera_config_path: Optional[str | Path] = None,
) -> PerceptionPlanningPipeline:
    """
    Factory function to create a PerceptionPlanningPipeline.

    Parameters
    ----------
    config : path to YAML/JSON, a config dict, or None for defaults.
    camera_config_path : path to camera_config.yaml for calibration data.
                         If provided, intrinsics/resolution/table_z are
                         loaded from this file and override config defaults.

    Returns
    -------
    Fully initialised PerceptionPlanningPipeline.
    """
    return PerceptionPlanningPipeline(
        config_path_or_dict=config,
        camera_config_path=camera_config_path,
    )

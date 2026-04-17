"""
Thin wrapper around NVIDIA FoundationPose for our visual servoing pipeline.

FoundationPose provides 6-DoF object pose from RGB-D + a CAD mesh (or a
small set of reference images via its model-free/NeRF mode). In this repo
we use the model-based path: a programmatic box mesh built from measured
dimensions, or a user-supplied .obj/.ply.

The wrapper is deliberately lazy: the heavy dependencies (torch, nvdiffrast,
kaolin, trimesh plus the FoundationPose repo itself) are only imported when
``FoundationPoseWrapper`` is first instantiated, so the rest of the servo
pipeline keeps running on machines without FP installed.

Intended usage (called from dinov2_servo.py with --mode foundationpose):

    fp = FoundationPoseWrapper(
        mesh_path="meshes/cheez_it.obj",
        # or mesh_extents_m=(0.19, 0.06, 0.22) for a procedural box
        fp_repo_dir="/path/to/FoundationPose",
        weights_dir="/path/to/FoundationPose/weights",
        est_refine_iter=5,
        track_refine_iter=2,
    )
    fp.setup()  # loads networks onto GPU

    T = fp.register(rgb=bgr_to_rgb(frame), depth_m=depth_m,
                    ob_mask=sam2_mask_bool, K=K3x3)
    # then per frame:
    T = fp.track(rgb=..., depth_m=..., K=K3x3)
    tx, ty, tz = T[:3, 3]  # metres, camera frame
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════
#  Procedural mesh helper
# ═════════════════════════════════════════════════════════════════════

def make_box_mesh(extents_m: Tuple[float, float, float]):
    """
    Build an axis-aligned box mesh centred at the origin with the given
    (width, height, depth) in metres. FoundationPose accepts any trimesh
    object; a plain box is a reasonable stand-in for cereal / cardboard
    targets where no CAD is available.
    """
    try:
        import trimesh
    except ImportError as exc:
        raise ImportError(
            "trimesh is required for make_box_mesh. Install with "
            "`pip install trimesh`.") from exc
    w, h, d = (float(v) for v in extents_m)
    if min(w, h, d) <= 0:
        raise ValueError(f"extents must be positive, got {extents_m}")
    return trimesh.creation.box(extents=[w, h, d])


def load_mesh(path: str):
    """Load a CAD mesh from disk (OBJ/PLY/STL via trimesh)."""
    import trimesh
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        raise ValueError(f"Loaded mesh is empty: {path}")
    # FoundationPose expects metres. If the file is in millimetres, the
    # user must scale it before passing it in; we do not autodetect.
    return mesh


# ═════════════════════════════════════════════════════════════════════
#  Main wrapper
# ═════════════════════════════════════════════════════════════════════

class FoundationPoseUnavailable(RuntimeError):
    """Raised when FoundationPose cannot be imported or initialised."""


class FoundationPoseWrapper:
    """
    Holds the FoundationPose estimator plus the loaded mesh and keeps
    track of the most recent pose so callers can alternate between
    ``register`` (first frame / re-init) and ``track`` (subsequent frames).
    """

    def __init__(
        self,
        mesh_path: Optional[str] = None,
        mesh_extents_m: Optional[Tuple[float, float, float]] = None,
        fp_repo_dir: Optional[str] = None,
        weights_dir: Optional[str] = None,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        debug: int = 0,
        debug_dir: Optional[str] = None,
    ):
        if mesh_path is None and mesh_extents_m is None:
            raise ValueError(
                "Provide either mesh_path or mesh_extents_m.")
        self.mesh_path = mesh_path
        self.mesh_extents_m = mesh_extents_m
        self.fp_repo_dir = (
            fp_repo_dir
            or os.environ.get("FOUNDATIONPOSE_ROOT")
            or os.path.join(
                os.path.dirname(__file__), "third-party", "FoundationPose"))
        self.weights_dir = weights_dir
        self.est_refine_iter = int(est_refine_iter)
        self.track_refine_iter = int(track_refine_iter)
        self.debug = int(debug)
        self.debug_dir = debug_dir or "/tmp/foundationpose_debug"

        self._est = None
        self._mesh = None
        self._last_pose: Optional[np.ndarray] = None
        self._registered: bool = False

    # ─────────────────────────────────────────────────────────────
    #  Setup (loads heavy dependencies on demand)
    # ─────────────────────────────────────────────────────────────

    def setup(self):
        """Import FP, construct the estimator, preload networks."""
        if self._est is not None:
            return

        if not os.path.isdir(self.fp_repo_dir):
            raise FoundationPoseUnavailable(
                f"FoundationPose repo not found at {self.fp_repo_dir}. "
                "Set FOUNDATIONPOSE_ROOT or pass fp_repo_dir.")

        # Prepend the repo to sys.path so its imports resolve
        if self.fp_repo_dir not in sys.path:
            sys.path.insert(0, self.fp_repo_dir)

        try:
            from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
            import nvdiffrast.torch as dr
        except ImportError as exc:
            raise FoundationPoseUnavailable(
                "Could not import FoundationPose / nvdiffrast. Make sure "
                "the GPU environment described in SETUP_FOUNDATIONPOSE.md "
                "is active."
            ) from exc

        # Load (or build) the mesh
        if self.mesh_path is not None:
            self._mesh = load_mesh(self.mesh_path)
            logger.info("FP: loaded mesh %s (verts=%d)",
                        self.mesh_path, len(self._mesh.vertices))
        else:
            self._mesh = make_box_mesh(self.mesh_extents_m)
            logger.info("FP: procedural box mesh extents=%s",
                        self.mesh_extents_m)

        os.makedirs(self.debug_dir, exist_ok=True)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()

        self._est = FoundationPose(
            model_pts=self._mesh.vertices,
            model_normals=self._mesh.vertex_normals,
            mesh=self._mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=glctx,
        )
        logger.info("FoundationPose estimator ready.")

    # ─────────────────────────────────────────────────────────────
    #  Registration (first frame / re-init)
    # ─────────────────────────────────────────────────────────────

    def register(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        ob_mask: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """
        First-frame registration. Returns a 4x4 object-in-camera pose.

        rgb       : HxWx3 uint8 RGB (NOT BGR)
        depth_m   : HxW float32 depth in metres (invalid pixels = 0)
        ob_mask   : HxW bool array (True = object)
        K         : 3x3 float camera intrinsics
        """
        if self._est is None:
            self.setup()
        if ob_mask.dtype != bool:
            ob_mask = ob_mask.astype(bool)
        pose = self._est.register(
            K=K.astype(np.float64),
            rgb=rgb,
            depth=depth_m.astype(np.float32),
            ob_mask=ob_mask,
            iteration=self.est_refine_iter,
        )
        pose = np.asarray(pose).reshape(4, 4)
        self._last_pose = pose
        self._registered = True
        return pose

    # ─────────────────────────────────────────────────────────────
    #  Tracking (subsequent frames)
    # ─────────────────────────────────────────────────────────────

    def track(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """Per-frame pose update. Must be preceded by ``register``."""
        if not self._registered:
            raise RuntimeError(
                "FoundationPose.track called before register; "
                "run register() on the first frame.")
        pose = self._est.track_one(
            rgb=rgb,
            depth=depth_m.astype(np.float32),
            K=K.astype(np.float64),
            iteration=self.track_refine_iter,
        )
        pose = np.asarray(pose).reshape(4, 4)
        self._last_pose = pose
        return pose

    # ─────────────────────────────────────────────────────────────
    #  Convenience
    # ─────────────────────────────────────────────────────────────

    @property
    def is_registered(self) -> bool:
        return self._registered

    def reset(self):
        """Drop the current pose track so the next call must be register()."""
        self._last_pose = None
        self._registered = False

    def translation_camera_m(self) -> Optional[np.ndarray]:
        """Return t_obj_in_cam in metres, or None if not registered yet."""
        if self._last_pose is None:
            return None
        return self._last_pose[:3, 3].astype(np.float64)

    def rotation_camera(self) -> Optional[np.ndarray]:
        """Return R_obj_in_cam (3x3), or None if not registered yet."""
        if self._last_pose is None:
            return None
        return self._last_pose[:3, :3].astype(np.float64)

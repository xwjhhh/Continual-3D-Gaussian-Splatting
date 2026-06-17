"""
4D Gaussian Splatting baseline bridge for CL-Splats.

4DGS is not an online continual update method: it learns one 4D Gaussian
distribution from all timestamps at once.  This trainer therefore collects all
requested CL-Splats timesteps, writes a temporary 4DGS-compatible dynamic
transforms dataset, and launches the original 4DGaussians training entrypoint.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import omegaconf
import torch
from loguru import logger
from plyfile import PlyData, PlyElement
from PIL import Image


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    value = cfg.get(key, default)
    return default if value is None else value


def _as_plain_container(value: Any) -> Any:
    if omegaconf.OmegaConf.is_config(value):
        return omegaconf.OmegaConf.to_container(value, resolve=True)
    return value


def _as_list(value: Any) -> List[str]:
    value = _as_plain_container(value)
    if value is None:
        return []
    if isinstance(value, str):
        return [part for part in value.split() if part]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_4dgs_root(model_cfg: Any) -> Path:
    fourdgs_cfg = _cfg_get(model_cfg, "fourdgs", {})
    configured = _cfg_get(fourdgs_cfg, "root", None)
    candidates = []
    if configured:
        candidates.append(Path(str(configured)).expanduser())
    candidates.append(_repo_root() / "4DGaussians-master")
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "train.py").is_file():
            return candidate
    raise FileNotFoundError(
        "4DGaussians-master not found. Set model.fourdgs.root to the directory "
        "that contains 4DGS train.py."
    )


def _camera_to_nerf_matrix(
    cam_info: Any,
    center: Optional[np.ndarray] = None,
    scale: float = 1.0,
) -> List[List[float]]:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(cam_info.R, dtype=np.float64).T
    w2c[:3, 3] = np.asarray(cam_info.T, dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    if center is not None:
        c2w[:3, 3] = (c2w[:3, 3] - np.asarray(center, dtype=np.float64)) * float(scale)
        w2c = np.linalg.inv(c2w)

    # 4DGS reads transforms with the Blender/D-NeRF loader:
    #   matrix = inv(transform_matrix)
    #   R = -matrix[:3, :3].T; R[:, 0] *= -1
    #   T = -matrix[:3, 3]
    # Write the inverse of that mapping so the loader recovers the original
    # COLMAP/3DGS R,T exactly.  The usual NeRF y/z column flip recovers R but
    # changes T to [-tx, ty, tz], which breaks WAT camera projections.
    loader_w2c = np.eye(4, dtype=np.float64)
    loader_w2c[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ w2c[:3, :3]
    loader_w2c[:3, 3] = -w2c[:3, 3]
    return np.linalg.inv(loader_w2c).tolist()


def _fov2focal(fov: float, pixels: float) -> float:
    return float(pixels) / (2.0 * math.tan(float(fov) * 0.5))


def _focal2fov(focal: float, pixels: float) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def _frame_fovs_for_image(
    cam_info: Any,
    image_size: tuple[int, int],
    intrinsics_scale: float = 1.0,
) -> tuple[float, float]:
    image_w, image_h = int(image_size[0]), int(image_size[1])
    ref_w = max(1, int(getattr(cam_info, "width", image_w) or image_w))
    ref_h = max(1, int(getattr(cam_info, "height", image_h) or image_h))

    focal_x = getattr(cam_info, "focal_x", None)
    focal_y = getattr(cam_info, "focal_y", None)
    if focal_x is None:
        focal_x = _fov2focal(float(cam_info.FovX), ref_w)
    if focal_y is None:
        focal_y = _fov2focal(float(cam_info.FovY), ref_h)

    scale_x = float(image_w) / float(ref_w)
    scale_y = float(image_h) / float(ref_h)
    focal_scale = float(intrinsics_scale)
    focal_x = float(focal_x) * scale_x * focal_scale
    focal_y = float(focal_y) * scale_y * focal_scale
    return _focal2fov(focal_x, image_w), _focal2fov(focal_y, image_h)


def _link_or_copy(src: Path, dst: Path, prefer_link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if prefer_link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _write_empty_ply(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 0\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n",
        encoding="ascii",
    )


def _store_4dgs_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb)
    if rgb.max(initial=0.0) <= 1.0:
        rgb = rgb * 255.0
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    normals = np.zeros_like(xyz, dtype=np.float32)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


class FourDGSTrainer:
    """External 4DGS baseline runner with the CL-Splats trainer interface."""

    supports_builtin_eval = True
    eval_after_training_only = True

    def __init__(self, cfg: omegaconf.DictConfig):
        self.cfg = cfg
        self.timestep = 0
        self.history: Dict[str, list] = {
            "losses": [],
            "num_gaussians": [],
            "num_pruned": [],
        }
        self._dataset = None
        self._prepared = False
        self._trained = False
        self._source_dir: Optional[Path] = None
        self._model_path: Optional[Path] = None
        self._last_command: List[str] = []
        self._num_train_frames = 0
        self._num_test_frames = 0
        self._image_extension = ".png"
        self._normalization: Dict[str, Any] = {
            "enabled": False,
            "center": None,
            "scale": 1.0,
            "radius": None,
        }

        model_cfg = cfg.get("model", {})
        self.model_cfg = model_cfg if model_cfg is not None else {}
        self.fourdgs_cfg = _cfg_get(self.model_cfg, "fourdgs", {})
        self.root = _resolve_4dgs_root(self.model_cfg)
        logger.info(f"Using 4DGS root: {self.root}")

    def initialize_from_point_cloud(self, *args, **kwargs) -> None:
        """CL-Splats calls this before the timestep loop; 4DGS initializes itself."""
        return None

    def load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location="cpu")
        self.timestep = int(checkpoint.get("timestep", 0))
        self._trained = bool(checkpoint.get("trained", False))
        source_dir = checkpoint.get("source_dir", None)
        model_path = checkpoint.get("model_path", None)
        self._source_dir = Path(source_dir) if source_dir else None
        self._model_path = Path(model_path) if model_path else None
        self._last_command = list(checkpoint.get("command", []))
        logger.info(f"Loaded 4DGS bridge checkpoint from {path}")

    def prepare_timestep(self, timestep: int, dataset) -> None:
        self.timestep = int(timestep)
        if self._dataset is None:
            self._dataset = dataset

    def train(self) -> Dict[str, Any]:
        if self._dataset is None:
            raise RuntimeError("4DGS trainer has no dataset. prepare_timestep() was not called.")

        if self._trained:
            logger.info(
                f"4DGS baseline already trained on all requested timesteps; "
                f"skip CL timestep {self.timestep}."
            )
            return self._stats()

        self._source_dir = self._build_4dgs_dataset(self._dataset)
        self._model_path = self._resolve_model_path()

        if self._should_skip_existing_output():
            logger.info(f"4DGS output already exists, skip training: {self._model_path}")
        else:
            cmd = self._build_train_command(self._source_dir, self._model_path)
            self._last_command = cmd
            logger.info("Launching 4DGS all-timestep training:")
            logger.info(" ".join(cmd))
            env = os.environ.copy()
            omp_threads = str(env.get("OMP_NUM_THREADS", "")).strip()
            if not omp_threads.isdigit():
                env["OMP_NUM_THREADS"] = "8"
            subprocess.run(cmd, cwd=str(self.root), check=True, env=env)

        self._trained = True
        self.history["losses"].append(0.0)
        self.history["num_gaussians"].append(0)
        self.history["num_pruned"].append(0)
        return self._stats()

    def log_history(self) -> None:
        logger.info(
            f"4DGS bridge history: trained={self._trained}, "
            f"train_frames={self._num_train_frames}, test_frames={self._num_test_frames}"
        )

    def save_checkpoint(
        self,
        path: str,
        ply_path_override: Optional[str] = None,
        save_external_ply: bool = False,
        compact_for_eval: bool = False,
    ) -> None:
        del save_external_ply, compact_for_eval
        payload = {
            "trainer_type": "4dgs_all_timesteps",
            "config": omegaconf.OmegaConf.to_container(self.cfg, resolve=True),
            "timestep": int(self.timestep),
            "trained": bool(self._trained),
            "source_dir": str(self._source_dir) if self._source_dir else None,
            "model_path": str(self._model_path) if self._model_path else None,
            "command": list(self._last_command),
            "num_train_frames": int(self._num_train_frames),
            "num_test_frames": int(self._num_test_frames),
            "image_extension": self._image_extension,
            "normalization": self._normalization,
            "supports_builtin_eval": True,
            "ply_path": ply_path_override,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(payload, path)
        logger.info(f"Saved 4DGS bridge checkpoint to {path}")

    def save_current_ply(self, path: str) -> None:
        dst = Path(path)
        src = self._find_latest_4dgs_ply()
        if src is not None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return
        logger.warning(
            "Could not find a 4DGS point_cloud.ply yet; writing an empty placeholder "
            f"so CL-Splats finalization can complete: {dst}"
        )
        _write_empty_ply(dst)

    def _stats(self) -> Dict[str, Any]:
        return {
            "avg_loss": 0.0,
            "num_gaussians": 0,
            "trained": bool(self._trained),
            "source_dir": str(self._source_dir) if self._source_dir else None,
            "model_path": str(self._model_path) if self._model_path else None,
        }

    def _resolve_num_times(self, dataset) -> int:
        train_cfg = self.cfg.get("train", {})
        requested = int(_cfg_get(train_cfg, "num_times", dataset.get_num_timesteps()))
        return max(1, min(requested, int(dataset.get_num_timesteps())))

    def _resolve_generated_dir(self, dataset) -> Path:
        configured = _cfg_get(self.fourdgs_cfg, "generated_data_dir", None)
        if configured:
            return Path(str(configured)).expanduser().resolve()
        data_name = Path(str(self.cfg.get("dataset", {}).get("path", "dataset"))).name
        output_dir = Path(str(self.cfg.get("output_dir", "outputs"))).expanduser()
        return (output_dir / "_4dgs_input" / data_name).resolve()

    def _build_4dgs_dataset(self, dataset) -> Path:
        source_dir = self._resolve_generated_dir(dataset)
        image_dir = source_dir / "images"
        source_dir.mkdir(parents=True, exist_ok=True)

        num_times = self._resolve_num_times(dataset)
        prefer_hardlinks = bool(_cfg_get(self.fourdgs_cfg, "prefer_hardlinks", True))
        extension = str(_cfg_get(self.fourdgs_cfg, "extension", "") or "")
        normalize_scene = bool(_cfg_get(self.fourdgs_cfg, "normalize_scene", True))
        target_radius = float(_cfg_get(self.fourdgs_cfg, "normalization_radius", 1.0))
        radius_percentile = float(_cfg_get(self.fourdgs_cfg, "normalization_percentile", 95.0))
        max_normalized_radius = float(_cfg_get(self.fourdgs_cfg, "normalization_max_radius", 3.0))
        intrinsics_scale = float(_cfg_get(self.fourdgs_cfg, "intrinsics_scale", 1.0))
        norm_center, norm_scale, norm_radius = self._compute_scene_normalization(
            dataset,
            num_times=num_times,
            enabled=normalize_scene,
            target_radius=target_radius,
            radius_percentile=radius_percentile,
        )

        train_frames: List[dict] = []
        test_frames: List[dict] = []
        first_fovx: Optional[float] = None
        copied_suffix: Optional[str] = None

        for timestep in range(num_times):
            raw_time = float(timestep if num_times > 1 else 1.0)
            scene_info = dataset.get_scene_info(timestep)
            timestep_frames = [
                (False, cam_info) for cam_info in scene_info.train_cameras
            ] + [
                (True, cam_info) for cam_info in scene_info.test_cameras
            ]
            for local_idx, (is_test, cam_info) in enumerate(timestep_frames):
                src = Path(str(cam_info.image_path))
                if not src.is_file():
                    logger.warning(f"Skip missing 4DGS input image: {src}")
                    continue
                try:
                    with Image.open(src) as image_probe:
                        image_size = image_probe.size
                except Exception:
                    logger.warning(f"Could not inspect 4DGS input image size, using camera metadata: {src}")
                    image_size = (
                        int(getattr(cam_info, "width", 1) or 1),
                        int(getattr(cam_info, "height", 1) or 1),
                    )
                frame_fovx, frame_fovy = _frame_fovs_for_image(
                    cam_info,
                    image_size=image_size,
                    intrinsics_scale=intrinsics_scale,
                )
                if (len(train_frames) + len(test_frames)) < 10:
                    logger.info(
                        "4DGS frame intrinsics: "
                        f"path={src.name}, "
                        f"image={image_size[0]}x{image_size[1]}, "
                        f"camera_meta={getattr(cam_info, 'width', '?')}x{getattr(cam_info, 'height', '?')}, "
                        f"focal=({getattr(cam_info, 'focal_x', None)}, {getattr(cam_info, 'focal_y', None)}), "
                        f"intrinsics_scale={intrinsics_scale:.6f}, "
                        f"fovx={frame_fovx:.8f}, fovy={frame_fovy:.8f}"
                    )

                suffix = extension or src.suffix
                if suffix not in _IMAGE_SUFFIXES:
                    suffix = ".png"
                if copied_suffix is None:
                    copied_suffix = suffix
                dst_rel_no_ext = f"images/t{timestep:04d}_{local_idx:05d}"
                dst = source_dir / f"{dst_rel_no_ext}{copied_suffix}"
                _link_or_copy(src, dst, prefer_hardlinks)

                frame = {
                    "file_path": dst_rel_no_ext,
                    "time": raw_time,
                    "camera_angle_x": float(frame_fovx),
                    "camera_angle_y": float(frame_fovy),
                    "transform_matrix": _camera_to_nerf_matrix(
                        cam_info,
                        center=norm_center,
                        scale=norm_scale,
                    ),
                }
                if is_test:
                    test_frames.append(frame)
                else:
                    train_frames.append(frame)
                if first_fovx is None:
                    first_fovx = float(cam_info.FovX)

        if not train_frames:
            raise RuntimeError("No train frames were collected for 4DGS.")
        if not test_frames:
            logger.warning("No test frames found for 4DGS; writing an empty transforms_test.json.")
        if copied_suffix is not None:
            self._image_extension = copied_suffix

        camera_angle_x = float(first_fovx if first_fovx is not None else math.radians(60.0))
        self._write_transforms(source_dir / "transforms_train.json", camera_angle_x, train_frames)
        self._write_transforms(source_dir / "transforms_test.json", camera_angle_x, test_frames)
        self._copy_initial_point_cloud(
            dataset,
            source_dir,
            center=norm_center,
            scale=norm_scale,
            max_normalized_radius=max_normalized_radius if norm_center is not None else None,
        )

        self._normalization = {
            "enabled": bool(norm_center is not None),
            "center": None if norm_center is None else [float(v) for v in norm_center.tolist()],
            "scale": float(norm_scale),
            "radius": None if norm_radius is None else float(norm_radius),
            "target_radius": float(target_radius),
            "radius_percentile": float(radius_percentile),
            "max_normalized_radius": float(max_normalized_radius),
        }

        self._num_train_frames = len(train_frames)
        self._num_test_frames = len(test_frames)
        logger.info(
            f"Prepared 4DGS all-time input at {source_dir}: "
            f"{len(train_frames)} train frames, {len(test_frames)} test frames, "
            f"timesteps=0..{num_times - 1}"
        )
        if norm_center is not None:
            logger.info(
                "4DGS input normalization enabled: "
                f"center={self._normalization['center']}, "
                f"radius_p{radius_percentile:g}={self._normalization['radius']:.6f}, "
                f"scale={self._normalization['scale']:.8f}"
            )
        return source_dir

    @staticmethod
    def _write_transforms(path: Path, camera_angle_x: float, frames: List[dict]) -> None:
        payload = {
            "camera_angle_x": camera_angle_x,
            "frames": frames,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _compute_scene_normalization(
        self,
        dataset,
        num_times: int,
        enabled: bool,
        target_radius: float,
        radius_percentile: float,
    ) -> tuple[Optional[np.ndarray], float, Optional[float]]:
        if not enabled:
            return None, 1.0, None
        try:
            samples: List[np.ndarray] = []
            point_count = 0
            camera_count = 0

            for timestep in range(max(1, int(num_times))):
                scene_info = dataset.get_scene_info(timestep)
                pcd = getattr(scene_info, "point_cloud", None)
                if pcd is not None:
                    points = np.asarray(pcd.points, dtype=np.float64)
                    if points.ndim == 2 and points.shape[1] == 3 and points.size > 0:
                        finite = np.isfinite(points).all(axis=1)
                        points = points[finite]
                        if points.size > 0:
                            samples.append(points)
                            point_count += int(points.shape[0])

                cameras = list(getattr(scene_info, "train_cameras", []))
                cameras += list(getattr(scene_info, "test_cameras", []))
                centers = []
                for cam_info in cameras:
                    w2c = np.eye(4, dtype=np.float64)
                    w2c[:3, :3] = np.asarray(cam_info.R, dtype=np.float64).T
                    w2c[:3, 3] = np.asarray(cam_info.T, dtype=np.float64)
                    center = np.linalg.inv(w2c)[:3, 3]
                    if np.isfinite(center).all():
                        centers.append(center)
                if centers:
                    samples.append(np.asarray(centers, dtype=np.float64))
                    camera_count += len(centers)

            if not samples:
                return None, 1.0, None
            points = np.concatenate(samples, axis=0)
            center = np.median(points, axis=0)
            distances = np.linalg.norm(points - center[None, :], axis=1)
            percentile = float(np.clip(radius_percentile, 50.0, 100.0))
            radius = float(np.percentile(distances, percentile))
            if radius <= 1e-12:
                return None, 1.0, None
            scale = float(target_radius) / radius
            logger.info(
                "Computed global 4DGS normalization from all requested timesteps: "
                f"points={point_count}, camera_centers={camera_count}, "
                f"timesteps={max(1, int(num_times))}"
            )
            return center, scale, radius
        except Exception:
            logger.exception("Failed to compute 4DGS scene normalization; using original coordinates.")
            return None, 1.0, None

    def _copy_initial_point_cloud(
        self,
        dataset,
        source_dir: Path,
        center: Optional[np.ndarray] = None,
        scale: float = 1.0,
        max_normalized_radius: Optional[float] = None,
    ) -> None:
        try:
            scene_info = dataset.get_scene_info(0)
            pcd = scene_info.point_cloud
            if center is not None:
                xyz = (np.asarray(pcd.points, dtype=np.float64) - center[None, :]) * float(scale)
                rgb = np.asarray(pcd.colors)
                if max_normalized_radius is not None and max_normalized_radius > 0:
                    keep = np.linalg.norm(xyz, axis=1) <= float(max_normalized_radius)
                    if keep.any() and keep.sum() < xyz.shape[0]:
                        logger.info(
                            "Filtered 4DGS normalization outliers from fused.ply: "
                            f"kept={int(keep.sum())}/{int(xyz.shape[0])}, "
                            f"max_normalized_radius={float(max_normalized_radius):.6f}"
                        )
                        xyz = xyz[keep]
                        rgb = rgb[keep]
                _store_4dgs_ply(source_dir / "fused.ply", xyz, rgb)
                return
            ply_path = Path(str(scene_info.ply_path))
            if ply_path.is_file():
                shutil.copy2(ply_path, source_dir / "fused.ply")
        except Exception:
            logger.exception("Failed to copy initial point cloud for 4DGS; it will fall back to random points.")

    def _resolve_model_path(self) -> Path:
        configured = _cfg_get(self.fourdgs_cfg, "model_path", None)
        if configured:
            return Path(str(configured)).expanduser().resolve()
        return Path(str(self.cfg.get("output_dir", "outputs"))).expanduser().resolve()

    def _build_train_command(self, source_dir: Path, model_path: Path) -> List[str]:
        python_bin = str(_cfg_get(self.fourdgs_cfg, "python_bin", sys.executable))
        port = int(_cfg_get(self.fourdgs_cfg, "port", 6017))
        ip = str(_cfg_get(self.fourdgs_cfg, "ip", "127.0.0.1"))
        configs = _cfg_get(self.fourdgs_cfg, "configs", None)
        if configs is None:
            default_configs = self.root / "arguments" / "multipleview" / "default.py"
            if default_configs.is_file():
                configs = str(default_configs)
        extension = str(_cfg_get(self.fourdgs_cfg, "extension", "") or "")
        if not extension:
            extension = self._image_extension
        expname = str(_cfg_get(self.fourdgs_cfg, "expname", model_path.name))

        cmd = [
            python_bin,
            "train.py",
            "-s",
            str(source_dir),
            "--model_path",
            str(model_path),
            "--port",
            str(port),
            "--ip",
            ip,
            "--expname",
            expname,
            "--extension",
            extension,
        ]
        if configs:
            cmd.extend(["--configs", str(configs)])

        train_cfg = self.cfg.get("train", {})
        iterations = _cfg_get(self.fourdgs_cfg, "iterations", _cfg_get(train_cfg, "iterations", None))
        if iterations is not None:
            cmd.extend(["--iterations", str(int(iterations))])
        coarse_iterations = _cfg_get(self.fourdgs_cfg, "coarse_iterations", None)
        if coarse_iterations is not None:
            cmd.extend(["--coarse_iterations", str(int(coarse_iterations))])
        coarse_first_timestep_only = bool(
            _cfg_get(self.fourdgs_cfg, "coarse_first_timestep_only", False)
        )
        if coarse_first_timestep_only:
            cmd.append("--coarse_first_timestep_only")

        opacity_reset_interval = int(
            _cfg_get(self.fourdgs_cfg, "opacity_reset_interval", 1_000_000_000)
        )
        cmd.extend(["--opacity_reset_interval", str(opacity_reset_interval)])

        cmd.extend(_as_list(_cfg_get(self.fourdgs_cfg, "extra_args", [])))
        return cmd

    def _should_skip_existing_output(self) -> bool:
        if not bool(_cfg_get(self.fourdgs_cfg, "skip_if_output_exists", True)):
            return False
        return self._find_latest_4dgs_ply() is not None

    def _find_latest_4dgs_ply(self) -> Optional[Path]:
        if self._model_path is None:
            return None
        point_cloud_dir = self._model_path / "point_cloud"
        if not point_cloud_dir.is_dir():
            return None
        candidates = list(point_cloud_dir.glob("iteration_*/point_cloud.ply"))
        if not candidates:
            candidates = list(point_cloud_dir.glob("coarse_iteration_*/point_cloud.ply"))
        if not candidates:
            return None

        def _iter_num(path: Path) -> int:
            digits = "".join(ch for ch in path.parent.name if ch.isdigit())
            return int(digits) if digits else -1

        return max(candidates, key=_iter_num)

"""
Dataset reader for CL-Splats.

Supports loading COLMAP and NeRF Synthetic datasets with multi-timestep support
for continual learning scenarios.
"""

import os
import sys
import json
import random
import re
from typing import Any, List, Dict, NamedTuple, Optional
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData, PlyElement

from clsplats.dataset.colmap_reader import (
    read_extrinsics_text, read_intrinsics_text,
    read_extrinsics_binary, read_intrinsics_binary,
    read_points3D_binary, read_points3D_text,
    qvec2rotmat
)
from clsplats.utils.graphics_utils import (
    BasicPointCloud, getWorld2View2, focal2fov, fov2focal
)
from clsplats.utils.sh_utils import SH2RGB
from clsplats.utils.camera_utils import Camera
from clsplats.utils.mask_utils import load_mask_as_tensor


class CameraInfo(NamedTuple):
    """Camera information container."""
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image_path: str
    image_name: str
    width: int
    height: int
    object_mask_path: str = ""
    depth_path: str = ""
    depth_params: dict = None
    is_test: bool = False
    focal_x: Optional[float] = None
    focal_y: Optional[float] = None


class SceneInfo(NamedTuple):
    """Scene information container."""
    point_cloud: BasicPointCloud
    train_cameras: List[CameraInfo]
    test_cameras: List[CameraInfo]
    nerf_normalization: dict
    ply_path: str
    is_nerf_synthetic: bool = False


class CLSplatsDataset:
    """
    Dataset class for CL-Splats with multi-timestep support.
    
    Supports two data organization modes:
    1. Single timestep: Standard COLMAP/NeRF format
    2. Multi-timestep: Separate folders for each timestep (t0, t1, t2, ...)
    """

    SPLIT_FILE_NAME = "train_test_split.json"
    
    def __init__(
        self,
        path: str,
        resolution_scale: float = 1.0,
        white_background: bool = False,
        eval_mode: bool = False,
        device: str = "cuda",
        split_seed: int = 0,
        prefer_undist: bool = True,
    ):
        """
        Initialize the dataset.
        
        Args:
            path: Path to dataset root
            resolution_scale: Scale factor for image resolution
            white_background: Whether to use white background
            eval_mode: Whether to split train/test
            device: Device for tensors
            split_seed: Seed for deterministic train/test split fallback
            prefer_undist: Whether to prefer images_undist/sparse_undist when both exist
        """
        self.path = path
        self.resolution_scale = resolution_scale
        self.white_background = white_background
        self.eval_mode = eval_mode
        self.device = device
        self.split_seed = split_seed
        self.prefer_undist = prefer_undist
        self._timestep_layout = "single"  # "root_dirs", "images_subdirs", "single"
        self._warned_camera_models = set()
        self._images_root_name = "images"
        self._sparse_root_name = "sparse"
        self._split_file_path = os.path.join(self.path, self.SPLIT_FILE_NAME)
        self._predefined_split = self._load_predefined_split()
        self._force_random_split = False
        
        # Detect dataset type and timesteps
        self.timesteps = self._detect_timesteps()
        self.dataset_type = self._detect_dataset_type()
        
        # Cache for loaded data
        self._scene_info_cache: Dict[int, SceneInfo] = {}
        self._cameras_cache: Dict[tuple, List[Camera]] = {}
        self._images_cache: Dict[tuple, List[torch.Tensor]] = {}
        self._object_masks_cache: Dict[tuple, List[Optional[torch.Tensor]]] = {}
    
    def _detect_timesteps(self) -> List[int]:
        """
        Detect available timesteps in the dataset.

        Supports three layouts:
        1) Root timesteps:
           path/t0, path/t1, ...
        2) Images subdir timesteps:
           path/images_undist/t0... or path/images/t0...
           (shared sparse model at path/sparse_undist or path/sparse)
        3) Single timestep:
           path/images_undist + path/sparse_undist (preferred) or path/images + path/sparse
        """
        # Layout 1: path/t0, path/t1, ...
        root_timesteps = []
        for item in os.listdir(self.path):
            item_path = os.path.join(self.path, item)
            if os.path.isdir(item_path) and item.startswith("t") and item[1:].isdigit():
                root_timesteps.append(int(item[1:]))
        if root_timesteps:
            self._timestep_layout = "root_dirs"
            return sorted(root_timesteps)

        # Layout 2: path/images_undist/t0... or path/images/t0...
        image_root_candidates = ("images_undist", "images") if self.prefer_undist else ("images", "images_undist")
        for images_root_name in image_root_candidates:
            images_root = os.path.join(self.path, images_root_name)
            images_timesteps = []
            if os.path.isdir(images_root):
                for item in os.listdir(images_root):
                    item_path = os.path.join(images_root, item)
                    if os.path.isdir(item_path) and item.startswith("t") and item[1:].isdigit():
                        images_timesteps.append(int(item[1:]))
            if images_timesteps:
                self._timestep_layout = "images_subdirs"
                self._images_root_name = images_root_name
                return sorted(images_timesteps)

        # Layout 3: single timestep dataset
        self._timestep_layout = "single"
        for images_root_name in image_root_candidates:
            if os.path.isdir(os.path.join(self.path, images_root_name)):
                self._images_root_name = images_root_name
                break
        return [0]

    def _resolve_sparse_root_name(self, base_path: str) -> str:
        """Choose sparse root name under base_path, preferring undistorted data."""
        sparse_candidates = ("sparse_undist", "sparse") if self.prefer_undist else ("sparse", "sparse_undist")
        for sparse_root_name in sparse_candidates:
            if os.path.isdir(os.path.join(base_path, sparse_root_name)):
                return sparse_root_name
        return "sparse"
    
    def _detect_dataset_type(self) -> str:
        """Detect dataset type (colmap or blender)."""
        # For root_dirs, sparse/transforms are expected per-timestep.
        # For images_subdirs/single, sparse/transforms are expected at dataset root.
        if self._timestep_layout == "root_dirs":
            check_path = self._get_timestep_path(self.timesteps[0])
        else:
            check_path = self.path
        sparse_root_name = self._resolve_sparse_root_name(check_path)
        self._sparse_root_name = sparse_root_name
        
        if os.path.exists(os.path.join(check_path, sparse_root_name)):
            return "colmap"
        elif os.path.exists(os.path.join(check_path, "transforms_train.json")):
            return "blender"
        else:
            raise ValueError(
                f"Unknown dataset type at {check_path} "
                f"(layout={self._timestep_layout})"
            )
    
    def _get_timestep_path(self, timestep: int) -> str:
        """Get path for a specific timestep."""
        if self._timestep_layout == "root_dirs":
            return os.path.join(self.path, f"t{timestep}")
        return self.path

    def _get_timestep_images_path(self, timestep: int) -> str:
        """Get images folder path for a specific timestep."""
        if self._timestep_layout == "root_dirs":
            t_path = os.path.join(self.path, f"t{timestep}")
            if os.path.isdir(os.path.join(t_path, "images_undist")):
                return os.path.join(t_path, "images_undist")
            return os.path.join(t_path, "images")
        if self._timestep_layout == "images_subdirs":
            return os.path.join(self.path, self._images_root_name, f"t{timestep}")
        return os.path.join(self.path, self._images_root_name)

    def _camera_sort_key(self, cam_info: CameraInfo):
        """
        Sort by the trailing number after the last underscore in image name.
        Example: IMG_123_45.png -> key 45.
        """
        stem = Path(cam_info.image_name).stem
        match = re.search(r"_(\d+)$", stem)
        if match:
            return (0, int(match.group(1)), stem)
        # Fallback: trailing digits without underscore
        match = re.search(r"(\d+)$", stem)
        if match:
            return (1, int(match.group(1)), stem)
        # Final fallback: lexical
        return (2, stem)

    def _load_predefined_split(self) -> Optional[Dict[str, Dict[str, List[str]]]]:
        """
        Load predefined train/test split from dataset root, if available.

        File format:
        {
          "timesteps": {
            "0": {"train": [...], "test": [...]},
            "1": {"train": [...], "test": [...]}
          }
        }
        """
        if not os.path.isfile(self._split_file_path):
            return None

        try:
            with open(self._split_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: Failed to load split file {self._split_file_path}: {exc}")
            return None

        timesteps = data.get("timesteps", None)
        if not isinstance(timesteps, dict):
            print(
                f"Warning: Invalid split file format at {self._split_file_path}. "
                "Expected top-level key 'timesteps'."
            )
            return None

        parsed: Dict[str, Dict[str, List[str]]] = {}
        for timestep_key, split_entry in timesteps.items():
            if not isinstance(split_entry, dict):
                continue
            train_list = split_entry.get("train", [])
            test_list = split_entry.get("test", [])
            if not isinstance(train_list, list) or not isinstance(test_list, list):
                continue
            parsed[str(timestep_key)] = {
                "train": [str(x) for x in train_list],
                "test": [str(x) for x in test_list],
            }

        if not parsed:
            print(
                f"Warning: No valid timestep entries found in split file {self._split_file_path}."
            )
            return None

        print(f"Loaded predefined split file: {self._split_file_path}")
        return parsed

    def _get_predefined_split_for_timestep(self, timestep: int) -> Optional[Dict[str, List[str]]]:
        """Get predefined split entry for one timestep, if present."""
        if bool(getattr(self, "_force_random_split", False)):
            return None
        if self._predefined_split is None:
            return None

        key_candidates = [str(timestep), f"t{timestep}"]
        for key in key_candidates:
            split_entry = self._predefined_split.get(key)
            if split_entry is not None:
                return split_entry
        return None

    def is_force_random_split_enabled(self) -> bool:
        """Whether predefined split file is currently ignored."""
        return bool(getattr(self, "_force_random_split", False))

    def clear_runtime_cache(self) -> None:
        """Clear cached scene/camera/image objects after split-policy changes."""
        if isinstance(self._scene_info_cache, dict):
            self._scene_info_cache.clear()
        if isinstance(self._cameras_cache, dict):
            self._cameras_cache.clear()
        if isinstance(self._images_cache, dict):
            self._images_cache.clear()
        if isinstance(self._object_masks_cache, dict):
            self._object_masks_cache.clear()

    def configure_split_policy(
        self,
        *,
        force_random_split: Optional[bool] = None,
        split_seed: Optional[int] = None,
        eval_mode: Optional[bool] = None,
        clear_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Unified split-policy interface used by both train.py and eval.py.

        Future split behavior changes should be implemented here, so callers
        do not need to duplicate split-file/cache logic.
        """
        changed = False

        if eval_mode is not None:
            eval_mode_value = bool(eval_mode)
            if bool(self.eval_mode) != eval_mode_value:
                self.eval_mode = eval_mode_value
                changed = True

        if split_seed is not None:
            split_seed_value = int(split_seed)
            if int(self.split_seed) != split_seed_value:
                self.split_seed = split_seed_value
                changed = True

        if force_random_split is not None:
            force_random_split_value = bool(force_random_split)
            if bool(self._force_random_split) != force_random_split_value:
                self._force_random_split = force_random_split_value
                if self._force_random_split:
                    self._predefined_split = None
                else:
                    self._predefined_split = self._load_predefined_split()
                changed = True

        if changed and bool(clear_cache):
            self.clear_runtime_cache()

        return {
            "eval_mode": bool(self.eval_mode),
            "split_seed": int(self.split_seed),
            "force_random_split": bool(self._force_random_split),
            "has_predefined_split": bool(self._predefined_split is not None),
        }

    def has_predefined_split(self) -> bool:
        """Whether a valid predefined split file was loaded."""
        return self._predefined_split is not None
    
    def get_num_timesteps(self) -> int:
        """Get number of timesteps."""
        return len(self.timesteps)

    def get_dataset_debug_info(self) -> Dict[str, str]:
        """Get dataset-level debug information for logging."""
        return {
            "configured_root": self.path,
            "resolved_root": str(Path(self.path).expanduser().resolve()),
            "dataset_type": self.dataset_type,
            "layout": self._timestep_layout,
            "images_root": self._images_root_name,
            "sparse_root": self._sparse_root_name,
            "timesteps": ",".join([f"t{t}" for t in self.timesteps]),
            "split_file": self._split_file_path if self.has_predefined_split() else "N/A",
            "split_policy": (
                "random_only"
                if self.is_force_random_split_enabled()
                else ("predefined_or_random" if self.has_predefined_split() else "random_only")
            ),
        }

    def get_timestep_data_source(self, timestep: int = 0) -> Dict[str, str]:
        """Get detailed data source paths for one timestep."""
        timestep_root = self._get_timestep_path(timestep)
        images_path = self._get_timestep_images_path(timestep)
        sparse_root_name = self._resolve_sparse_root_name(timestep_root)
        sparse_model_path = os.path.join(timestep_root, sparse_root_name, "0")

        scene_info = self.get_scene_info(timestep)
        sample_image = "N/A"
        sample_resolution = "N/A"
        if scene_info.train_cameras:
            first_cam = scene_info.train_cameras[0]
            sample_image = first_cam.image_path
            sample_resolution = f"{first_cam.width}x{first_cam.height}"

        return {
            "timestep": str(timestep),
            "dataset_type": self.dataset_type,
            "layout": self._timestep_layout,
            "timestep_root": timestep_root,
            "images_path": images_path,
            "sparse_model_path": sparse_model_path,
            "point_cloud_path": scene_info.ply_path,
            "num_train_cameras": str(len(scene_info.train_cameras)),
            "num_test_cameras": str(len(scene_info.test_cameras)),
            "sample_image": sample_image,
            "sample_resolution": sample_resolution,
        }
    
    def get_scene_info(self, timestep: int = 0) -> SceneInfo:
        """Get scene information for a timestep."""
        if timestep not in self._scene_info_cache:
            path = self._get_timestep_path(timestep)
            
            if self.dataset_type == "colmap":
                scene_info = self._read_colmap_scene(path, timestep=timestep)
            else:
                scene_info = self._read_nerf_synthetic_scene(path)
            
            self._scene_info_cache[timestep] = scene_info
        
        return self._scene_info_cache[timestep]
    
    def get_cameras(self, timestep: int = 0, include_test: bool = False) -> List[Camera]:
        """Get camera objects for a timestep."""
        cache_key = (timestep, include_test)
        if cache_key not in self._cameras_cache:
            scene_info = self.get_scene_info(timestep)
            cameras = []
            
            cam_infos = scene_info.train_cameras
            if include_test:
                cam_infos = cam_infos + scene_info.test_cameras
            
            for cam_info in cam_infos:
                camera = self._load_camera(cam_info)
                cameras.append(camera)
            
            self._cameras_cache[cache_key] = cameras
        
        return self._cameras_cache[cache_key]

    def get_test_cameras(self, timestep: int = 0) -> List[Camera]:
        """Get test camera objects for a timestep."""
        cache_key = (timestep, "test_only")
        if cache_key not in self._cameras_cache:
            scene_info = self.get_scene_info(timestep)
            self._cameras_cache[cache_key] = [self._load_camera(c) for c in scene_info.test_cameras]
        return self._cameras_cache[cache_key]
    
    def get_images(self, timestep: int = 0, include_test: bool = False) -> List[torch.Tensor]:
        """Get ground truth images for a timestep."""
        cache_key = (timestep, include_test)
        if cache_key not in self._images_cache:
            cameras = self.get_cameras(timestep, include_test=include_test)
            images = []
            
            for camera in cameras:
                if camera.original_image is not None:
                    images.append(camera.original_image)
            
            self._images_cache[cache_key] = images
        
        return self._images_cache[cache_key]

    def get_object_masks(self, timestep: int = 0, include_test: bool = False) -> List[Optional[torch.Tensor]]:
        """Get object masks for a timestep, aligned with get_cameras/get_images ordering."""
        cache_key = (timestep, include_test)
        if cache_key not in self._object_masks_cache:
            scene_info = self.get_scene_info(timestep)
            cam_infos = scene_info.train_cameras
            if include_test:
                cam_infos = cam_infos + scene_info.test_cameras

            masks: List[Optional[torch.Tensor]] = []
            for cam_info in cam_infos:
                if cam_info.object_mask_path:
                    masks.append(
                        load_mask_as_tensor(
                            cam_info.object_mask_path,
                            device=self.device,
                            resolution_scale=self.resolution_scale,
                        )
                    )
                else:
                    masks.append(None)
            self._object_masks_cache[cache_key] = masks
        return self._object_masks_cache[cache_key]

    def get_test_object_masks(self, timestep: int = 0) -> List[Optional[torch.Tensor]]:
        """Get object masks for test cameras of one timestep."""
        cache_key = (timestep, "test_only")
        if cache_key not in self._object_masks_cache:
            scene_info = self.get_scene_info(timestep)
            masks: List[Optional[torch.Tensor]] = []
            for cam_info in scene_info.test_cameras:
                if cam_info.object_mask_path:
                    masks.append(
                        load_mask_as_tensor(
                            cam_info.object_mask_path,
                            device=self.device,
                            resolution_scale=self.resolution_scale,
                        )
                    )
                else:
                    masks.append(None)
            self._object_masks_cache[cache_key] = masks
        return self._object_masks_cache[cache_key]

    def get_test_images(self, timestep: int = 0) -> List[torch.Tensor]:
        """Get test ground truth images for a timestep."""
        cache_key = (timestep, "test_only")
        if cache_key not in self._images_cache:
            cameras = self.get_test_cameras(timestep)
            images = []
            for camera in cameras:
                if camera.original_image is not None:
                    images.append(camera.original_image)
            self._images_cache[cache_key] = images
        return self._images_cache[cache_key]
    
    def get_point_cloud(self, timestep: int = 0) -> BasicPointCloud:
        """Get initial point cloud for a timestep."""
        scene_info = self.get_scene_info(timestep)
        return scene_info.point_cloud
    
    def _load_camera(self, cam_info: CameraInfo) -> Camera:
        """Load a Camera object from CameraInfo."""
        # Load image
        image = None
        if os.path.exists(cam_info.image_path):
            pil_image = Image.open(cam_info.image_path)
            img_w, img_h = pil_image.size
            ref_w, ref_h = int(cam_info.width), int(cam_info.height)
            if (img_w != ref_w) or (img_h != ref_h):
                print(
                    "Warning: image size differs from COLMAP camera size for "
                    f"{cam_info.image_name}: image={img_w}x{img_h}, camera={ref_w}x{ref_h}. "
                    "Using actual image size for rendering/eval."
                )
            
            # Resize if needed
            if self.resolution_scale != 1.0:
                # Scale from the actual loaded image size, not camera metadata size.
                new_w = max(1, int(round(float(img_w) / float(self.resolution_scale))))
                new_h = max(1, int(round(float(img_h) / float(self.resolution_scale))))
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
            else:
                new_w, new_h = pil_image.size
            
            # Convert to tensor
            im_data = np.array(pil_image.convert("RGBA"))
            
            # Handle background
            bg = np.array([1, 1, 1]) if self.white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            
            image = torch.from_numpy(arr).float().permute(2, 0, 1).to(self.device)
        else:
            new_w, new_h = cam_info.width, cam_info.height
        
        object_mask = None
        if cam_info.object_mask_path:
            object_mask = load_mask_as_tensor(
                cam_info.object_mask_path,
                device=self.device,
                resolution_scale=self.resolution_scale,
            )

        return Camera(
            uid=cam_info.uid,
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image_width=new_w,
            image_height=new_h,
            image_name=cam_info.image_name,
            image=image,
            device=self.device,
            object_mask=object_mask,
        )

    def _resolve_object_mask_dir(self, timestep: Optional[int] = None) -> Optional[str]:
        """Resolve object_mask directory for one timestep."""
        if self._timestep_layout == "root_dirs":
            if timestep is None:
                return None
            candidate = os.path.join(self.path, f"t{timestep}", "object_mask")
            return candidate if os.path.isdir(candidate) else None

        if self._timestep_layout == "images_subdirs":
            candidate = os.path.join(self.path, "object_mask", f"t{timestep}")
            return candidate if os.path.isdir(candidate) else None

        candidate = os.path.join(self.path, "object_mask")
        return candidate if os.path.isdir(candidate) else None
    
    def _read_colmap_scene(self, path: str, timestep: Optional[int] = None) -> SceneInfo:
        """Read COLMAP format scene."""
        # Try binary first, then text
        sparse_root = os.path.join(path, self._resolve_sparse_root_name(path), "0")
        try:
            cameras_extrinsic_file = os.path.join(sparse_root, "images.bin")
            cameras_intrinsic_file = os.path.join(sparse_root, "cameras.bin")
            cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(sparse_root, "images.txt")
            cameras_intrinsic_file = os.path.join(sparse_root, "cameras.txt")
            cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
            cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
        # Read cameras
        if timestep is not None:
            images_folder = self._get_timestep_images_path(timestep)
        else:
            images_folder = os.path.join(path, "images")
        cam_infos = self._read_colmap_cameras(
            cam_extrinsics, cam_intrinsics, images_folder, timestep=timestep
        )
        cam_infos = sorted(cam_infos, key=self._camera_sort_key)
        
        # Split train/test
        if self.eval_mode:
            split_entry = self._get_predefined_split_for_timestep(
                timestep if timestep is not None else 0
            )
            used_predefined_split = False
            if split_entry is not None:
                train_names = set(split_entry.get("train", []))
                test_names = set(split_entry.get("test", []))

                test_cam_infos = [c for c in cam_infos if c.image_name in test_names]
                train_cam_infos = [c for c in cam_infos if c.image_name in train_names]

                # Any image not explicitly listed is treated as train by default.
                unassigned = [
                    c for c in cam_infos
                    if c.image_name not in test_names and c.image_name not in train_names
                ]
                train_cam_infos.extend(unassigned)

                if len(test_names) > 0 and len(test_cam_infos) == 0:
                    print(
                        f"Warning: Split file has test entries for timestep {timestep}, "
                        "but none matched loaded camera image names. Falling back to random split."
                    )
                else:
                    used_predefined_split = True

            if not used_predefined_split:
                if len(cam_infos) <= 1:
                    train_cam_infos = cam_infos
                    test_cam_infos = []
                else:
                    num_test = max(1, len(cam_infos) // 8)
                    rng = random.Random(self.split_seed + (timestep if timestep is not None else 0))
                    all_indices = list(range(len(cam_infos)))
                    rng.shuffle(all_indices)
                    test_indices = set(all_indices[:num_test])
                    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx not in test_indices]
                    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_indices]
            else:
                # Guardrail: keep at least one train camera if possible.
                if len(train_cam_infos) == 0 and len(test_cam_infos) > 0:
                    train_cam_infos = [test_cam_infos[0]]
                    test_cam_infos = test_cam_infos[1:]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []
        
        # Compute normalization
        nerf_normalization = self._get_nerf_normalization(train_cam_infos)
        
        # Load point cloud
        ply_path = os.path.join(sparse_root, "points3D.ply")
        bin_path = os.path.join(sparse_root, "points3D.bin")
        txt_path = os.path.join(sparse_root, "points3D.txt")
        
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply...")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            self._store_ply(ply_path, xyz, rgb)
        
        pcd = self._fetch_ply(ply_path)
        
        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=nerf_normalization,
            ply_path=ply_path,
            is_nerf_synthetic=False
        )
    
    def _read_colmap_cameras(
        self,
        cam_extrinsics: dict,
        cam_intrinsics: dict,
        images_folder: str,
        timestep: Optional[int] = None
    ) -> List[CameraInfo]:
        """Read camera information from COLMAP data."""
        cam_infos = []
        object_mask_dir = self._resolve_object_mask_dir(timestep)
        
        # Get list of actual images in the folder
        existing_images = set()
        existing_image_basenames = set()
        if os.path.exists(images_folder):
            for name in os.listdir(images_folder):
                file_path = os.path.join(images_folder, name)
                if os.path.isfile(file_path):
                    normalized = name.replace("\\", "/")
                    existing_images.add(normalized)
                    existing_image_basenames.add(os.path.basename(normalized))
        
        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            extr_name = extr.name.replace("\\", "/")
            extr_basename = os.path.basename(extr_name)
            
            # If COLMAP stores names as "tX/xxx.png", keep only cameras for this timestep.
            if timestep is not None and "/" in extr_name:
                prefix = extr_name.split("/", 1)[0]
                if prefix.startswith("t") and prefix[1:].isdigit() and prefix != f"t{timestep}":
                    continue
            
            # Skip if image doesn't exist in this timestep's folder
            if existing_images:
                candidate_names = [extr_name, extr_basename]
                selected_name = None
                for candidate in candidate_names:
                    candidate_path = os.path.join(images_folder, candidate)
                    if os.path.exists(candidate_path):
                        selected_name = candidate
                        break
                if selected_name is None:
                    if extr_name in existing_images:
                        selected_name = extr_name
                    elif extr_basename in existing_image_basenames:
                        selected_name = extr_basename
                    else:
                        continue
            else:
                selected_name = extr_basename
            
            height = intr.height
            width = intr.width
            uid = intr.id
            
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            
            if intr.model == "SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = focal_length_x
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE", "FOV"}:
                # Distortion parameters are ignored here. This loader uses a pinhole projection.
                focal_length_x = intr.params[0]
                focal_length_y = focal_length_x
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
                if intr.model not in self._warned_camera_models:
                    print(
                        f"Warning: Camera model {intr.model} has distortion terms. "
                        "Loader will ignore distortion and use pinhole approximation."
                    )
                    self._warned_camera_models.add(intr.model)
            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
                # Use fx/fy from intrinsic parameters and ignore distortion coefficients.
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
                if intr.model not in self._warned_camera_models:
                    print(
                        f"Warning: Camera model {intr.model} has distortion terms. "
                        "Loader will ignore distortion and use pinhole approximation."
                    )
                    self._warned_camera_models.add(intr.model)
            else:
                raise ValueError(f"Camera model {intr.model} not supported")
            
            image_path = os.path.join(images_folder, selected_name)
            image_name = os.path.basename(selected_name)
            object_mask_path = ""
            if object_mask_dir is not None:
                image_stem = Path(image_name).stem
                candidate = os.path.join(object_mask_dir, image_stem + ".png")
                if os.path.exists(candidate):
                    object_mask_path = candidate
            
            cam_info = CameraInfo(
                uid=uid, R=R, T=T, FovY=FovY, FovX=FovX,
                image_path=image_path, image_name=image_name,
                width=width, height=height,
                object_mask_path=object_mask_path,
                focal_x=float(focal_length_x),
                focal_y=float(focal_length_y),
            )
            cam_infos.append(cam_info)
        
        return cam_infos
    
    def _read_nerf_synthetic_scene(self, path: str) -> SceneInfo:
        """Read NeRF Synthetic format scene."""
        # Read train cameras
        train_cam_infos = self._read_nerf_cameras(
            path, "transforms_train.json", is_test=False
        )
        
        # Read test cameras
        test_cam_infos = []
        if self.eval_mode:
            test_cam_infos = self._read_nerf_cameras(
                path, "transforms_test.json", is_test=True
            )
        
        nerf_normalization = self._get_nerf_normalization(train_cam_infos)
        
        # Generate or load point cloud
        ply_path = os.path.join(path, "points3d.ply")
        if not os.path.exists(ply_path):
            num_pts = 100_000
            print(f"Generating random point cloud ({num_pts})...")
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self._store_ply(ply_path, xyz, SH2RGB(shs) * 255)
        else:
            pcd = self._fetch_ply(ply_path)
        
        return SceneInfo(
            point_cloud=pcd,
            train_cameras=train_cam_infos,
            test_cameras=test_cam_infos,
            nerf_normalization=nerf_normalization,
            ply_path=ply_path,
            is_nerf_synthetic=True
        )
    
    def _read_nerf_cameras(
        self,
        path: str,
        transforms_file: str,
        is_test: bool,
        extension: str = ".png"
    ) -> List[CameraInfo]:
        """Read cameras from NeRF transforms file."""
        cam_infos = []
        transforms_path = os.path.join(path, transforms_file)
        
        if not os.path.exists(transforms_path):
            return cam_infos
        
        with open(transforms_path) as f:
            contents = json.load(f)
        
        fovx = contents["camera_angle_x"]
        
        for idx, frame in enumerate(contents["frames"]):
            file_path = frame["file_path"]
            # Only add extension if file_path doesn't already have one
            if not file_path.endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
                file_path = file_path + extension
            cam_name = os.path.join(path, file_path)
            
            # Camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # Convert from OpenGL to COLMAP convention
            c2w[:3, 1:3] *= -1
            
            # World-to-camera
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]
            
            # Load image to get dimensions
            if os.path.exists(cam_name):
                image = Image.open(cam_name)
                width, height = image.size
            else:
                width, height = 800, 800  # Default
            
            fovy = focal2fov(fov2focal(fovx, width), height)
            
            cam_info = CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                image_path=cam_name, image_name=Path(cam_name).stem,
                width=width, height=height, is_test=is_test
            )
            cam_infos.append(cam_info)
        
        return cam_infos
    
    def _get_nerf_normalization(self, cam_infos: List[CameraInfo]) -> dict:
        """Compute NeRF++ normalization parameters."""
        cam_centers = []
        
        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])
        
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        radius = diagonal * 1.1
        translate = -center.flatten()
        
        return {"translate": translate, "radius": radius}
    
    def _fetch_ply(self, path: str) -> BasicPointCloud:
        """Load point cloud from PLY file."""
        plydata = PlyData.read(path)
        vertices = plydata['vertex']
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        
        try:
            normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        except:
            normals = np.zeros_like(positions)
        
        return BasicPointCloud(points=positions, colors=colors, normals=normals)
    
    def _store_ply(self, path: str, xyz: np.ndarray, rgb: np.ndarray):
        """Store point cloud to PLY file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        
        normals = np.zeros_like(xyz)
        elements = np.empty(xyz.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz, normals, rgb), axis=1)
        elements[:] = list(map(tuple, attributes))
        
        vertex_element = PlyElement.describe(elements, 'vertex')
        PlyData([vertex_element]).write(path)


# Convenience function for backward compatibility
def load_dataset(cfg) -> CLSplatsDataset:
    """Load dataset from configuration."""
    return CLSplatsDataset(
        path=cfg.get("path"),
        resolution_scale=cfg.get("resolution", 1.0),
        white_background=cfg.get("white_background", False),
        eval_mode=cfg.get("eval", False),
        device="cuda" if torch.cuda.is_available() else "cpu",
        split_seed=cfg.get("split_seed", 42),
        prefer_undist=cfg.get("prefer_undist", True),
    )

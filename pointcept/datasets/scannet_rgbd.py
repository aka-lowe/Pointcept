"""
ScanNet Stage1 Dataset with SENS Reader Integration for Pointcept

This version reads directly from .sens files without requiring full dataset extraction,
saving significant storage space while maintaining full functionality.

Author: Modified for Pointcept integration with SENS reader
"""

import logging
import os
import numpy as np
import torch
import cv2
import copy
from PIL import Image
from collections import Counter
from pathlib import Path
from random import random, sample, uniform, choice, randrange
from copy import deepcopy
from typing import List, Optional, Tuple, Union, Dict

import albumentations as A
import volumentations as V
import yaml

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS

# Import SENS reader components
from .sens_reader import (
    LazySensorData, 
    SensDatasetManager, 
    create_sens_data_lists,
    RGBDFrame
)


@DATASETS.register_module()
class ScanNetRGBDDataset(DefaultDataset):
    """
    ScanNet RGB-D Dataset with SENS integration for Pointcept

    This dataset processes ScanNet .sens files directly and converts RGB-D frames
    to point clouds with SAM segmentation for instance segmentation tasks.
    No need to extract the entire dataset - works directly with .sens files!
    """
    
    VALID_ASSETS = [
        "coord",
        "color", 
        "normal",
        "segment",
        "instance",
    ]

    def __init__(
        self,
        sam_folder: str = "sam",
        scenes_to_exclude: str = "",
        use_sens_files: bool = True,
        frame_skip: int = 25,
        sens_split_dir: Optional[str] = None,
        
        # ScanNet-specific parameters
        color_mean_std: Optional[Union[str, Tuple[Tuple[float]]]] = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        add_colors: bool = True,
        add_normals: bool = True,
        add_raw_coordinates: bool = False,
        add_instance: bool = False,
        num_labels: int = -1,
        data_percent: float = 1.0,
        ignore_label: Union[int, Tuple[int]] = 255,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        instance_oversampling: float = 0,
        place_around_existing: bool = False,
        max_cut_region: float = 0,
        point_per_cut: int = 100,
        flip_in_center: bool = False,
        noise_rate: float = 0.0,
        resample_points: float = 0.0,
        add_unlabeled_pc: bool = False,
        task: str = "instance_segmentation",
        cropping: bool = False,
        cropping_args: Optional[dict] = None,
        is_tta: bool = False,
        crop_min_size: int = 20000,
        crop_length: float = 6.0,
        cropping_v1: bool = True,
        reps_per_epoch: int = 1,
        area: float = -1,
        on_crops: bool = False,
        eval_inner_core: float = -1,
        filter_out_classes: List = None,
        label_offset: int = 0,
        add_clip: bool = False,
        is_elastic_distortion: bool = True,
        color_drop: float = 0.0,
        max_frames: Optional[int] = None,
        **kwargs
    ):
        # Initialize parameters before parent class
        self.use_sens_files = use_sens_files
        self.frame_skip = frame_skip
        self.sens_split_dir = sens_split_dir
        self.sam_folder = sam_folder
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_raw_coordinates = add_raw_coordinates
        self.add_instance = add_instance
        self.ignore_label = ignore_label
        self.task = task
        self.max_frames = max_frames
        self.color_drop = color_drop
        self.is_elastic_distortion = is_elastic_distortion
        self.filter_out_classes = filter_out_classes or []
        self.label_offset = label_offset
        
        # Handle scenes to exclude
        self.excluded_scenes = set()
        if scenes_to_exclude:
            self.excluded_scenes.update(scene.strip() for scene in scenes_to_exclude.split(',') if scene.strip())
        
        # Initialize SENS manager if using SENS files
        if self.use_sens_files:
            self.sens_manager = SensDatasetManager(
                data_root=kwargs.get('data_root', 'data/scannet'),
                split_dir=self.sens_split_dir,
                frame_skip=self.frame_skip
            )
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Setup augmentations
        self.setup_augmentations(volume_augmentations_path, image_augmentations_path)
        
        # Color normalization (using ImageNet stats as in original)
        if add_colors:
            color_mean = (0.485, 0.456, 0.406)
            color_std = (0.229, 0.224, 0.225)
            self.normalize_color = A.Normalize(mean=color_mean, std=color_std)
            
        logger = get_root_logger()
        logger.info(f"ScanNet Stage1 Dataset initialized with {len(self.data_list)} samples")
        logger.info(f"Using SENS files: {self.use_sens_files}")
        if self.excluded_scenes:
            logger.info(f"Excluding scenes: {self.excluded_scenes}")

    def get_data_list(self):
        """Override to get data list from SENS files or traditional txt files"""
        if self.use_sens_files:
            # Use SENS manager to get data list
            data_list = self.sens_manager.get_split_data_list(self.split)
        else:
            # Use traditional approach with extracted data
            if self.split == "train":
                data_path = os.path.join(self.data_root, '..', 'scannet_info', 'scannet_train.txt')
            else:
                data_path = os.path.join(self.data_root, '..', 'scannet_info', 'scannet_val.txt')
                
            with open(data_path, "r") as scene_file:
                data_list = scene_file.read().splitlines()
        
        # Filter out excluded scenes
        if self.excluded_scenes:
            original_count = len(data_list)
            data_list = [item for item in data_list if not any(excluded_scene in item for excluded_scene in self.excluded_scenes)]
            logger = get_root_logger()
            logger.info(f'Filtered dataset: {original_count} -> {len(data_list)} items after excluding scenes')
            
        # Apply max_frames limit if specified
        if self.max_frames and len(data_list) > self.max_frames:
            data_list = data_list[:self.max_frames]
            logger = get_root_logger()
            logger.info(f'Limited dataset to {self.max_frames} frames')
            
        return data_list

    def setup_augmentations(self, volume_augmentations_path, image_augmentations_path):
        """Setup volume and image augmentations"""
        self.volume_augmentations = V.NoOp()
        if volume_augmentations_path and volume_augmentations_path != "none":
            self.volume_augmentations = V.load(Path(volume_augmentations_path), data_format="yaml")
            
        self.image_augmentations = A.NoOp()
        if image_augmentations_path and image_augmentations_path != "none":
            self.image_augmentations = A.load(Path(image_augmentations_path), data_format="yaml")

    def get_data_name(self, idx):
        """Get data name for caching and identification"""
        fname = self.data_list[idx % len(self.data_list)]
        scene_id, frame_id = fname.split()
        return f"{scene_id}_{frame_id}"

    def get_data(self, idx):
        """
        Main data loading function - converts RGB-D frame to point cloud
        Works with both SENS files and extracted data
        """
        fname = self.data_list[idx % len(self.data_list)]
        scene_id, frame_id = fname.split()
        
        # Check cache first
        if self.cache:
            cache_name = f"pointcept-{self.get_data_name(idx)}"
            cached_data = shared_dict(cache_name)
            if cached_data is not None:
                return cached_data

        if self.use_sens_files:
            # Load directly from SENS files
            coordinates, color, normals, segments = self.load_from_sens(scene_id, int(frame_id))
        else:
            # Load from extracted data (original method)
            coordinates, color, normals, segments = self.load_from_extracted(scene_id, frame_id)

        # Prepare data dict in Pointcept format
        data_dict = {
            "coord": coordinates.astype(np.float32),
            "color": color.astype(np.float32),
            "normal": normals.astype(np.float32),
            "segment": segments.astype(np.int32),
            "name": self.get_data_name(idx),
        }
        
        # Add instance labels if available
        if self.add_instance:
            data_dict["instance"] = segments.astype(np.int32)
            
        return data_dict

    def load_from_sens(self, scene_id: str, frame_idx: int):
        """Load RGB-D data directly from SENS file and convert to point cloud"""
        # Get frame data from SENS file
        frame_data = self.sens_manager.load_frame_data(scene_id, frame_idx)
        color_image = frame_data['color']
        depth_image = frame_data['depth']
        pose = frame_data['pose']
        
        # Get camera intrinsics from SENS reader
        sens_reader = self.sens_manager.get_sens_reader(scene_id)
        depth_intrinsic = sens_reader.intrinsic_depth
        
        # Load SAM segmentation (fallback to dummy data if not available)
        sam_groups = self.load_sam_segmentation(scene_id, frame_idx, color_image.shape[:2])
        
        # Convert RGB-D to point cloud
        coordinates, color, normals, segments = self.rgbd_to_pointcloud(
            color_image, depth_image, pose, sam_groups, depth_intrinsic
        )
        
        return coordinates, color, normals, segments

    def load_from_extracted(self, scene_id: str, frame_id: str):
        """Load RGB-D data from extracted files (original method)"""
        # Load RGB-D data from extracted files
        color_path = os.path.join(self.data_root, 'scannet', scene_id, 'color', frame_id + '.jpg')
        color_image = cv2.imread(color_path)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        color_image = cv2.resize(color_image, (640, 480))

        depth_path = os.path.join(self.data_root, 'scannet', scene_id, 'depth', frame_id + '.png')
        depth_image = cv2.imread(depth_path, -1)

        pose_path = os.path.join(self.data_root, 'scannet', scene_id, 'pose', str(int(frame_id)) + '.txt')
        pose = np.loadtxt(pose_path)

        # Load depth intrinsics
        intrinsics_path = os.path.join(self.data_root, '..', 'scannet_info', 'intrinsics.txt')
        depth_intrinsic = np.loadtxt(intrinsics_path)

        # Load SAM segmentation
        sam_path = os.path.join(self.data_root, 'scannet', scene_id, self.sam_folder, f'{frame_id}.png')
        with open(sam_path, 'rb') as image_file:
            img = Image.open(image_file)
            sam_groups = np.array(img, dtype=np.int16)

        # Convert RGB-D to point cloud
        coordinates, color, normals, segments = self.rgbd_to_pointcloud(
            color_image, depth_image, pose, sam_groups, depth_intrinsic
        )
        
        return coordinates, color, normals, segments

    def load_sam_segmentation(self, scene_id: str, frame_idx: int, image_shape: Tuple[int, int]):
        """Load SAM segmentation, with fallback to dummy data"""
        try:
            # Try to load from extracted SAM folder
            sam_path = os.path.join(self.data_root, 'scannet', scene_id, self.sam_folder, f'{frame_idx}.png')
            if os.path.exists(sam_path):
                with open(sam_path, 'rb') as image_file:
                    img = Image.open(image_file)
                    sam_groups = np.array(img, dtype=np.int16)
                    # Resize to match color image if needed
                    if sam_groups.shape != image_shape:
                        sam_groups = cv2.resize(sam_groups, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
                    return sam_groups
        except Exception as e:
            logger = get_root_logger()
            logger.warning(f"Could not load SAM segmentation for {scene_id}/{frame_idx}: {e}")
        
        # Fallback: create dummy segmentation (single segment)
        logger = get_root_logger()
        logger.info(f"Using dummy SAM segmentation for {scene_id}/{frame_idx}")
        return np.zeros(image_shape, dtype=np.int16)

    def rgbd_to_pointcloud(self, color_image, depth_image, pose, sam_groups, depth_intrinsic):
        """Convert RGB-D frame to point cloud with SAM segmentation"""
        mask = (depth_image != 0)
        colors = np.reshape(color_image[mask], [-1, 3])
        sam_groups = sam_groups[mask]

        depth_shift = 1000.0
        x, y = np.meshgrid(
            np.linspace(0, depth_image.shape[1]-1, depth_image.shape[1]), 
            np.linspace(0, depth_image.shape[0]-1, depth_image.shape[0])
        )
        uv_depth = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
        uv_depth[:, :, 0] = x
        uv_depth[:, :, 1] = y
        uv_depth[:, :, 2] = depth_image / depth_shift
        uv_depth = np.reshape(uv_depth, [-1, 3])
        uv_depth = uv_depth[np.where(uv_depth[:, 2] != 0), :].squeeze()
        
        # Unproject to 3D
        fx, fy = depth_intrinsic[0, 0], depth_intrinsic[1, 1]
        cx, cy = depth_intrinsic[0, 2], depth_intrinsic[1, 2]
        bx, by = depth_intrinsic[0, 3], depth_intrinsic[1, 3]
        
        n = uv_depth.shape[0]
        points = np.ones((n, 4))
        X = (uv_depth[:, 0] - cx) * uv_depth[:, 2] / fx + bx
        Y = (uv_depth[:, 1] - cy) * uv_depth[:, 2] / fy + by
        points[:, 0] = X
        points[:, 1] = Y
        points[:, 2] = uv_depth[:, 2]
        
        # Transform to world coordinates
        points_world = np.dot(points, np.transpose(pose))
        coordinates = points_world[:, :3]
        
        # Process SAM groups
        sam_groups = self.num_to_natural(sam_groups)
        counts = Counter(sam_groups)
        for num, count in counts.items():
            if count < 100:  # Filter small segments
                sam_groups[sam_groups == num] = -1
        sam_groups = self.num_to_natural(sam_groups)
        
        # Prepare outputs
        color = colors.astype(np.float32)
        normals = np.ones_like(coordinates, dtype=np.float32)  # Placeholder normals
        segments = sam_groups.astype(np.int32)
        
        return coordinates, color, normals, segments

    def num_to_natural(self, group_ids):
        """Change the group number to natural number arrangement"""
        if np.all(group_ids == -1):
            return group_ids
        array = copy.deepcopy(group_ids)
        unique_values = np.unique(array[array != -1])
        if len(unique_values) == 0:
            return array
        mapping = np.full(np.max(unique_values) + 2, -1)
        mapping[unique_values + 1] = np.arange(len(unique_values))
        array = mapping[array + 1]
        return array

    def __len__(self):
        """Return dataset length"""
        return len(self.data_list) * self.loop

    @staticmethod
    def create_sens_data_lists_wrapper(data_root: str, output_dir: str, frame_skip: int = 25):
        """Wrapper function to create data lists from SENS files"""
        return create_sens_data_lists(data_root, output_dir, frame_skip)
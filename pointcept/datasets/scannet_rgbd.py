"""
Updated ScanNet RGB-D Dataset with Original Mesh Label Projection

This version projects original ScanNet mesh labels (semantic + instance) onto RGB-D frames
instead of using SAM segmentation, providing labels that match the original ScanNet dataset.

Author: Modified for original label projection
"""

import logging
import os
import json
import numpy as np
import torch
import cv2
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import trimesh
import open3d as o3d

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset

# Import SENS reader components (assuming these exist)
from .sens_reader import (
    LazySensorData, 
    SensDatasetManager, 
    create_sens_data_lists,
    RGBDFrame
)


class MeshLabelProjector:
    """
    Class to handle projection of original ScanNet mesh labels to RGB-D frames
    """
    
    def __init__(self, scene_root, labels_pd_path):
        self.scene_root = scene_root
        self.labels_pd = pd.read_csv(labels_pd_path, sep="\t", header=0)
        self.mesh_cache = {}
        self.logger = get_root_logger()
        
        # Create label mappings (same as original preprocessing)
        self.create_label_mappings()
    
    def create_label_mappings(self):
        """Create label mappings for ScanNet20 and ScanNet200"""
        # ScanNet20 mapping
        self.scannet20_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        self.label_to_scannet20 = {}
        
        for idx, row in self.labels_pd.iterrows():
            raw_category = row['raw_category']
            nyu40id = row['nyu40id']
            
            if nyu40id in self.scannet20_ids:
                scannet20_idx = self.scannet20_ids.index(nyu40id)
                self.label_to_scannet20[raw_category] = scannet20_idx
            else:
                self.label_to_scannet20[raw_category] = -1  # ignore label
    
    def load_scene_mesh_data(self, scene_id):
        """Load and cache mesh data for a scene"""
        if scene_id in self.mesh_cache:
            return self.mesh_cache[scene_id]
        
        scene_path = os.path.join(self.scene_root, scene_id)
        
        # File paths
        mesh_path = os.path.join(scene_path, f"{scene_id}_vh_clean_2.labels.ply")
        aggregation_path = os.path.join(scene_path, f"{scene_id}_vh_clean_2.aggregation.json")
        segments_path = os.path.join(scene_path, f"{scene_id}_vh_clean_2.0.010000.segs.json")
        
        if not all(os.path.exists(p) for p in [mesh_path, aggregation_path, segments_path]):
            self.logger.warning(f"Missing label files for scene {scene_id}")
            return None
        
        # Load mesh
        mesh = trimesh.load(mesh_path)
        vertices = np.array(mesh.vertices)
        
        # Load segments
        with open(segments_path, 'r') as f:
            segments_data = json.load(f)
            seg_indices = np.array(segments_data['segIndices'])
        
        # Load aggregations
        with open(aggregation_path, 'r') as f:
            aggregation_data = json.load(f)
            seg_groups = aggregation_data['segGroups']
        
        # Generate semantic and instance labels for each vertex
        num_vertices = len(vertices)
        semantic_labels = np.full(num_vertices, -1, dtype=np.int32)  # ignore label
        instance_labels = np.full(num_vertices, -1, dtype=np.int32)  # ignore label
        
        for group in seg_groups:
            # Get all vertices belonging to this instance
            vertex_mask = np.isin(seg_indices, group['segments'])
            
            # Map semantic label
            raw_label = group['label']
            semantic_id = self.label_to_scannet20.get(raw_label, -1)
            
            if semantic_id >= 0:  # Only assign if it's a valid ScanNet20 class
                semantic_labels[vertex_mask] = semantic_id
                instance_labels[vertex_mask] = group['objectId']
        
        # Cache the data
        mesh_data = {
            'vertices': vertices,
            'semantic_labels': semantic_labels,
            'instance_labels': instance_labels,
            'kdtree': None  # Will be created when needed
        }
        
        self.mesh_cache[scene_id] = mesh_data
        self.logger.info(f"Loaded mesh data for {scene_id}: {num_vertices} vertices")
        
        return mesh_data
    
    def project_labels_to_points(self, scene_id, points_3d, max_distance=0.05):
        """
        Project mesh labels to 3D points using nearest neighbor assignment
        
        Args:
            scene_id: Scene identifier
            points_3d: Array of 3D points (N, 3)
            max_distance: Maximum distance for valid label assignment
            
        Returns:
            semantic_labels: Array of semantic labels (N,)
            instance_labels: Array of instance labels (N,)
        """
        mesh_data = self.load_scene_mesh_data(scene_id)
        
        if mesh_data is None:
            # Return ignore labels if mesh data not available
            n_points = len(points_3d)
            return np.full(n_points, -1, dtype=np.int32), np.full(n_points, -1, dtype=np.int32)
        
        # Create KD-tree if not cached
        if mesh_data['kdtree'] is None:
            mesh_data['kdtree'] = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
            mesh_data['kdtree'].fit(mesh_data['vertices'])
        
        # Find nearest mesh vertices for each point
        distances, indices = mesh_data['kdtree'].kneighbors(points_3d)
        distances = distances.flatten()
        indices = indices.flatten()
        
        # Initialize with ignore labels
        n_points = len(points_3d)
        semantic_labels = np.full(n_points, -1, dtype=np.int32)
        instance_labels = np.full(n_points, -1, dtype=np.int32)
        
        # Only assign labels for points within distance threshold
        valid_mask = distances < max_distance
        
        if np.any(valid_mask):
            semantic_labels[valid_mask] = mesh_data['semantic_labels'][indices[valid_mask]]
            instance_labels[valid_mask] = mesh_data['instance_labels'][indices[valid_mask]]
        
        return semantic_labels, instance_labels


@DATASETS.register_module()
class ScanNetRGBDDataset(DefaultDataset):
    """
    ScanNet RGB-D Dataset with Original Mesh Label Projection
    
    This dataset projects original ScanNet mesh labels onto RGB-D frames
    to provide accurate semantic and instance segmentation labels.
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
        # Original mesh projection parameters
        use_original_labels: bool = True,
        mesh_projection_distance: float = 0.05,
        labels_pd_path: str = "meta_data/scannetv2-labels.combined.tsv",
        
        # Original parameters
        sam_folder: str = "sam",
        scenes_to_exclude: str = "",
        use_sens_files: bool = True,
        frame_skip: int = 25,
        sens_split_dir: str = None,
        
        # ScanNet-specific parameters
        color_mean_std = (
            (0.47793125906962, 0.4303257521323044, 0.3749598901421883),
            (0.2834475483823543, 0.27566157565723015, 0.27018971370874995),
        ),
        add_colors: bool = True,
        add_normals: bool = True,
        add_raw_coordinates: bool = False,
        add_instance: bool = True,  # Default to True for original labels
        num_labels: int = 20,  # ScanNet20 by default
        ignore_label: int = -1,
        max_frames: int = None,
        **kwargs
    ):
        # Store parameters
        self.use_original_labels = use_original_labels
        self.mesh_projection_distance = mesh_projection_distance
        self.use_sens_files = use_sens_files
        self.frame_skip = frame_skip
        self.sens_split_dir = sens_split_dir
        self.sam_folder = sam_folder
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.add_raw_coordinates = add_raw_coordinates
        self.add_instance = add_instance
        self.ignore_label = ignore_label
        self.max_frames = max_frames
        
        # Handle scenes to exclude
        self.excluded_scenes = set()
        if scenes_to_exclude:
            self.excluded_scenes.update(scene.strip() for scene in scenes_to_exclude.split(',') if scene.strip())
        
        # Initialize parent class
        super().__init__(**kwargs)
        
        # Initialize label projector if using original labels
        if self.use_original_labels:
            # Construct path to labels file
            if not os.path.isabs(labels_pd_path):
                script_dir = os.path.dirname(__file__)
                labels_pd_path = os.path.join(script_dir, "preprocessing", "scannet", labels_pd_path)
            
            self.label_projector = MeshLabelProjector(
                scene_root=os.path.join(self.data_root, "scans"),
                labels_pd_path=labels_pd_path
            )
        
        # Initialize SENS manager if using SENS files
        if self.use_sens_files:
            self.sens_manager = SensDatasetManager(
                data_root=self.data_root,
                split_dir=self.sens_split_dir,
                frame_skip=self.frame_skip
            )
        
        logger = get_root_logger()
        logger.info(f"ScanNet RGB-D Dataset initialized with {len(self.data_list)} samples")
        logger.info(f"Using original mesh labels: {self.use_original_labels}")
        logger.info(f"Using SENS files: {self.use_sens_files}")

    def get_data_list(self):
        """Get data list from SENS files or traditional txt files"""
        if self.use_sens_files:
            data_list = self.sens_manager.get_split_data_list(self.split)
        else:
            # Use traditional approach with extracted data
            if self.split == "train":
                data_path = os.path.join(self.data_root, 'scannet_train.txt')
            else:
                data_path = os.path.join(self.data_root, 'scannet_val.txt')
                
            with open(data_path, "r") as scene_file:
                scene_lines = scene_file.read().splitlines()
                # Convert to scene_id frame_idx format
                data_list = []
                for line in scene_lines:
                    scene_id = line.strip()
                    # Add frames with skip
                    for frame_idx in range(0, 2000, self.frame_skip):  # Assuming max 2000 frames
                        data_list.append(f"{scene_id} {frame_idx}")
        
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

    def get_data_name(self, idx):
        """Get data name for caching and identification"""
        fname = self.data_list[idx % len(self.data_list)]
        scene_id, frame_id = fname.split()
        return f"{scene_id}_{frame_id}"

    def rgbd_to_pointcloud_with_labels(self, color_image, depth_image, pose, depth_intrinsic, scene_id):
        """
        Convert RGB-D frame to point cloud and assign original mesh labels
        
        Args:
            color_image: RGB image (H, W, 3)
            depth_image: Depth image (H, W)
            pose: Camera pose matrix (4, 4)
            depth_intrinsic: Camera intrinsic matrix (3, 3)
            scene_id: Scene identifier for label projection
            
        Returns:
            coordinates: Point coordinates (N, 3)
            colors: Point colors (N, 3)
            normals: Point normals (N, 3)
            semantic_labels: Semantic labels (N,)
            instance_labels: Instance labels (N,)
        """
        height, width = depth_image.shape
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        depth_values = depth_image.flatten()
        
        # Filter out invalid depth values
        valid_mask = (depth_values > 0) & (depth_values < 10000)  # 10m max depth
        u = u[valid_mask]
        v = v[valid_mask]
        depth_values = depth_values[valid_mask]
        
        if len(depth_values) == 0:
            # Return empty arrays if no valid depth
            empty = np.array([]).reshape(0, 3)
            empty_labels = np.array([], dtype=np.int32)
            return empty, empty, empty, empty_labels, empty_labels
        
        # Convert to 3D camera coordinates
        fx, fy = depth_intrinsic[0, 0], depth_intrinsic[1, 1]
        cx, cy = depth_intrinsic[0, 2], depth_intrinsic[1, 2]
        
        x = (u - cx) * depth_values / fx
        y = (v - cy) * depth_values / fy
        z = depth_values
        
        # Convert depth units (ScanNet depth is in mm, convert to meters)
        points_camera = np.stack([x, y, z], axis=1) / 1000.0
        
        # Transform to world coordinates
        points_homogeneous = np.concatenate([points_camera, np.ones((len(points_camera), 1))], axis=1)
        points_world = (pose @ points_homogeneous.T).T[:, :3]
        
        # Get colors for valid points
        colors = color_image[v, u] / 255.0  # Normalize to [0, 1]
        
        # Compute normals (simple method using local neighborhoods)
        normals = self.compute_normals(points_world)
        
        # Project mesh labels to points
        if self.use_original_labels:
            semantic_labels, instance_labels = self.label_projector.project_labels_to_points(
                scene_id, points_world, max_distance=self.mesh_projection_distance
            )
        else:
            # Fallback to ignore labels
            semantic_labels = np.full(len(points_world), self.ignore_label, dtype=np.int32)
            instance_labels = np.full(len(points_world), self.ignore_label, dtype=np.int32)
        
        return points_world, colors, normals, semantic_labels, instance_labels
    
    def compute_normals(self, points, k=10):
        """Compute point normals using PCA on local neighborhoods"""
        if len(points) < k:
            return np.zeros_like(points)
        
        try:
            # Use Open3D for efficient normal computation
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
            normals = np.asarray(pcd.normals)
        except:
            # Fallback: use simple approach
            normals = np.zeros_like(points)
            # For each point, approximate normal using nearby points
            if len(points) > 3:
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=min(k, len(points)), algorithm='kd_tree').fit(points)
                
                for i in range(len(points)):
                    _, indices = nbrs.kneighbors([points[i]])
                    local_points = points[indices[0]]
                    
                    if len(local_points) >= 3:
                        # Compute normal using PCA
                        centered = local_points - local_points.mean(axis=0)
                        _, _, vh = np.linalg.svd(centered)
                        normals[i] = vh[-1]  # Normal is the last principal component
        
        return normals

    def get_data(self, idx):
        """
        Main data loading function - converts RGB-D frame to point cloud with original labels
        """
        fname = self.data_list[idx % len(self.data_list)]
        scene_id, frame_id = fname.split()
        
        # Check cache first
        if self.cache:
            cache_name = f"pointcept-{self.get_data_name(idx)}"
            cached_data = shared_dict(cache_name)
            if cached_data is not None:
                return cached_data

        try:
            if self.use_sens_files:
                # Load directly from SENS files
                frame_data = self.sens_manager.load_frame_data(scene_id, int(frame_id))
                color_image = frame_data['color']
                depth_image = frame_data['depth']
                pose = frame_data['pose']
                
                # Get camera intrinsics
                sens_reader = self.sens_manager.get_sens_reader(scene_id)
                depth_intrinsic = sens_reader.intrinsic_depth
            else:
                # Load from extracted data
                color_path = os.path.join(self.data_root, 'scannet', scene_id, 'color', frame_id + '.jpg')
                color_image = cv2.imread(color_path)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color_image = cv2.resize(color_image, (640, 480))

                depth_path = os.path.join(self.data_root, 'scannet', scene_id, 'depth', frame_id + '.png')
                depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

                pose_path = os.path.join(self.data_root, 'scannet', scene_id, 'pose', str(int(frame_id)) + '.txt')
                pose = np.loadtxt(pose_path)

                # Load depth intrinsics
                intrinsics_path = os.path.join(self.data_root, 'intrinsics.txt')
                depth_intrinsic = np.loadtxt(intrinsics_path)
            
            # Convert RGB-D to point cloud with original labels
            coordinates, color, normals, semantic_labels, instance_labels = self.rgbd_to_pointcloud_with_labels(
                color_image, depth_image, pose, depth_intrinsic, scene_id
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to load frame {scene_id}_{frame_id}: {e}")
            # Return empty data
            empty = np.array([]).reshape(0, 3)
            empty_labels = np.array([], dtype=np.int32)
            coordinates, color, normals, semantic_labels, instance_labels = empty, empty, empty, empty_labels, empty_labels

        # Prepare data dict in Pointcept format
        data_dict = {
            "coord": coordinates.astype(np.float32),
            "color": color.astype(np.float32),
            "normal": normals.astype(np.float32),
            "segment": semantic_labels.astype(np.int32),
            "name": self.get_data_name(idx),
        }
        
        # Add instance labels if requested
        if self.add_instance:
            data_dict["instance"] = instance_labels.astype(np.int32)
        
        # Cache the result
        if self.cache:
            shared_dict(cache_name, data_dict)
            
        return data_dict
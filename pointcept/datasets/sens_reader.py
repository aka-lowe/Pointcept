"""
SensorData Reader with Lazy Extraction for ScanNet Dataset

This module provides lazy extraction from .sens files with integration into the dataset pipeline.
It processes .sens files on-demand according to train/val/test splits.

Author: Modified for Pointcept integration
"""

import os
import struct
import numpy as np
import zlib
import imageio
import cv2
import png
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


class RGBDFrame:
    """Single RGB-D frame from ScanNet .sens file"""
    
    def __init__(self):
        self.camera_to_world = None
        self.timestamp_color = None
        self.timestamp_depth = None
        self.color_size_bytes = None
        self.depth_size_bytes = None
        self.color_data = None
        self.depth_data = None

    def load(self, file_handle):
        """Load frame data from file handle"""
        self.camera_to_world = np.asarray(
            struct.unpack('f'*16, file_handle.read(16*4)), 
            dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        
        # Read color and depth data as bytes
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        """Decompress depth data based on compression type"""
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        elif compression_type == 'raw_ushort':
            return self.depth_data
        else:
            raise ValueError(f"Unsupported depth compression: {compression_type}")

    def decompress_depth_zlib(self):
        """Decompress zlib compressed depth data"""
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        """Decompress color data based on compression type"""
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        elif compression_type == 'png':
            return self.decompress_color_png()
        else:
            raise ValueError(f"Unsupported color compression: {compression_type}")

    def decompress_color_jpeg(self):
        """Decompress JPEG color data"""
        return imageio.imread(self.color_data)
    
    def decompress_color_png(self):
        """Decompress PNG color data"""
        return imageio.imread(self.color_data)

    def get_color_image(self, compression_type, target_size=(640, 480)):
        """Get color image as numpy array"""
        color = self.decompress_color(compression_type)
        if target_size:
            color = cv2.resize(color, target_size, interpolation=cv2.INTER_LINEAR)
        return color

    def get_depth_image(self, compression_type, depth_width, depth_height, target_size=(640, 480)):
        """Get depth image as numpy array"""
        depth_data = self.decompress_depth(compression_type)
        depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(depth_height, depth_width)
        if target_size:
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        return depth


class LazySensorData:
    """
    Lazy SensorData reader that loads .sens files on-demand
    and provides frame-level access for dataset integration
    """
    
    def __init__(self, sens_file_path: str):
        self.sens_file_path = sens_file_path
        self.scene_id = self._extract_scene_id()
        
        # Metadata loaded immediately
        self.version = None
        self.sensor_name = None
        self.intrinsic_color = None
        self.extrinsic_color = None
        self.intrinsic_depth = None
        self.extrinsic_depth = None
        self.color_compression_type = None
        self.depth_compression_type = None
        self.color_width = None
        self.color_height = None
        self.depth_width = None
        self.depth_height = None
        self.depth_shift = None
        self.num_frames = None
        
        # Frame data positions for lazy loading
        self._frame_positions = []
        self._metadata_loaded = False
        
        # Load metadata immediately
        self._load_metadata()

    def _extract_scene_id(self) -> str:
        """Extract scene ID from .sens file path"""
        return Path(self.sens_file_path).stem

    def _load_metadata(self):
        """Load metadata from .sens file header"""
        if self._metadata_loaded:
            return
            
        with open(self.sens_file_path, 'rb') as f:
            # Read header
            self.version = struct.unpack('I', f.read(4))[0]
            assert self.version == 4, f"Unsupported .sens version: {self.version}"
            
            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = ''.join([
                struct.unpack('c', f.read(1))[0].decode('utf-8') 
                for _ in range(strlen)
            ])
            
            self.intrinsic_color = np.asarray(
                struct.unpack('f'*16, f.read(16*4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack('f'*16, f.read(16*4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack('f'*16, f.read(16*4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack('f'*16, f.read(16*4)), dtype=np.float32
            ).reshape(4, 4)
            
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack('i', f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack('i', f.read(4))[0]
            ]
            
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            self.num_frames = struct.unpack('Q', f.read(8))[0]
            
            # Record frame positions for lazy loading
            for i in range(self.num_frames):
                frame_start = f.tell()
                
                # Skip frame data to get to next frame
                f.seek(16*4, 1)  # camera_to_world
                f.seek(8, 1)     # timestamp_color
                f.seek(8, 1)     # timestamp_depth
                color_size = struct.unpack('Q', f.read(8))[0]
                depth_size = struct.unpack('Q', f.read(8))[0]
                f.seek(color_size + depth_size, 1)  # Skip actual data
                
                self._frame_positions.append({
                    'start': frame_start,
                    'color_size': color_size,
                    'depth_size': depth_size
                })
        
        self._metadata_loaded = True
        logger.info(f"Loaded metadata for {self.scene_id}: {self.num_frames} frames")

    def get_frame(self, frame_idx: int) -> RGBDFrame:
        """Get specific frame by index (lazy loading)"""
        if frame_idx >= self.num_frames:
            raise IndexError(f"Frame {frame_idx} not available (total: {self.num_frames})")
        
        with open(self.sens_file_path, 'rb') as f:
            f.seek(self._frame_positions[frame_idx]['start'])
            frame = RGBDFrame()
            frame.load(f)
            return frame

    def get_frame_data(self, frame_idx: int, target_size: Tuple[int, int] = (640, 480)) -> Dict:
        """
        Get frame data in format compatible with the dataset pipeline
        
        Returns:
            Dict with 'color', 'depth', 'pose' keys
        """
        frame = self.get_frame(frame_idx)
        
        color_image = frame.get_color_image(
            self.color_compression_type, 
            target_size=target_size
        )
        depth_image = frame.get_depth_image(
            self.depth_compression_type,
            self.depth_width,
            self.depth_height,
            target_size=target_size
        )
        
        return {
            'color': color_image,
            'depth': depth_image,
            'pose': frame.camera_to_world,
            'timestamp_color': frame.timestamp_color,
            'timestamp_depth': frame.timestamp_depth,
        }

    def get_valid_frame_indices(self, frame_skip: int = 1) -> List[int]:
        """Get list of valid frame indices with optional frame skipping"""
        return list(range(0, self.num_frames, frame_skip))


class SensDatasetManager:
    """
    Manager for handling multiple .sens files according to train/val/test splits
    """
    
    def __init__(
        self, 
        data_root: str,
        split_dir: Optional[str] = None,
        frame_skip: int = 25
    ):
        self.data_root = Path(data_root)
        self.split_dir = Path(split_dir) if split_dir else self.data_root / "meta_data"
        self.frame_skip = frame_skip
        
        # Load splits
        self.train_scenes = self._load_split_file("scannetv2_train.txt")
        self.val_scenes = self._load_split_file("scannetv2_val.txt")
        
        # Cache for lazy-loaded SensorData objects
        self._sens_cache = {}
        
        logger.info(f"Loaded splits: {len(self.train_scenes)} train, {len(self.val_scenes)} val")

    def _load_split_file(self, filename: str) -> List[str]:
        """Load scene list from split file"""
        split_file = self.split_dir / filename
        if not split_file.exists():
            logger.warning(f"Split file not found: {split_file}")
            return []
        
        with open(split_file, 'r') as f:
            scenes = [line.strip() for line in f.readlines() if line.strip()]
        return scenes

    def get_split_for_scene(self, scene_id: str) -> str:
        """Determine split for a given scene"""
        if scene_id in self.train_scenes:
            return "train"
        elif scene_id in self.val_scenes:
            return "val"
        else:
            return "test"

    def get_sens_reader(self, scene_id: str) -> LazySensorData:
        """Get or create lazy SensorData reader for a scene"""
        if scene_id in self._sens_cache:
            return self._sens_cache[scene_id]
        
        # Find .sens file for this scene
        sens_pattern = f"scene*/{scene_id}.sens"
        sens_files = list(self.data_root.glob(sens_pattern))
        
        if not sens_files:
            # Try alternative pattern
            sens_pattern = f"scans/{scene_id}/{scene_id}.sens"
            sens_files = list(self.data_root.glob(sens_pattern))
        
        if not sens_files:
            raise FileNotFoundError(f"No .sens file found for scene {scene_id}")
        
        sens_file = sens_files[0]
        sensor_data = LazySensorData(str(sens_file))
        self._sens_cache[scene_id] = sensor_data
        
        return sensor_data

    def get_scene_frame_list(self, scene_id: str) -> List[Tuple[str, int]]:
        """Get list of (scene_id, frame_id) tuples for a scene"""
        sens_reader = self.get_sens_reader(scene_id)
        valid_frames = sens_reader.get_valid_frame_indices(self.frame_skip)
        return [(scene_id, frame_idx) for frame_idx in valid_frames]

    def get_split_data_list(self, split: str) -> List[str]:
        """Get data list for specific split in 'scene_id frame_id' format"""
        if split == "train":
            scenes = self.train_scenes
        elif split == "val":
            scenes = self.val_scenes
        else:
            # For test, use all scenes not in train/val
            all_sens_files = list(self.data_root.glob("*/*.sens"))
            all_scenes = [f.stem for f in all_sens_files]
            scenes = [s for s in all_scenes if s not in self.train_scenes and s not in self.val_scenes]
        
        data_list = []
        for scene_id in scenes:
            try:
                scene_frames = self.get_scene_frame_list(scene_id)
                for scene, frame_idx in scene_frames:
                    data_list.append(f"{scene} {frame_idx}")
            except FileNotFoundError as e:
                logger.warning(f"Skipping scene {scene_id}: {e}")
                continue
        
        return data_list

    def load_frame_data(self, scene_id: str, frame_idx: int) -> Dict:
        """Load specific frame data lazily"""
        sens_reader = self.get_sens_reader(scene_id)
        return sens_reader.get_frame_data(frame_idx)


# Utility functions for integration
def create_sens_data_lists(
    data_root: str, 
    output_dir: str,
    frame_skip: int = 25
):
    """
    Create data list files for train/val/test splits from .sens files
    
    This creates text files in the format expected by your dataset:
    - scannet_train.txt
    - scannet_val.txt  
    - scannet_test.txt (optional)
    """
    manager = SensDatasetManager(data_root, frame_skip=frame_skip)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ["train", "val"]:
        data_list = manager.get_split_data_list(split)
        output_file = output_dir / f"scannet_{split}.txt"
        
        with open(output_file, 'w') as f:
            for item in data_list:
                f.write(f"{item}\n")
        
        logger.info(f"Created {split} list with {len(data_list)} items: {output_file}")


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to ScanNet data with .sens files")
    parser.add_argument("--output_dir", required=True, help="Output directory for data lists")
    parser.add_argument("--frame_skip", type=int, default=25, help="Frame skip interval")
    
    args = parser.parse_args()
    
    create_sens_data_lists(args.data_root, args.output_dir, args.frame_skip)
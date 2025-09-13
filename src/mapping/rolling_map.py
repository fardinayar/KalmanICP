"""
Rolling Map for LiDAR SLAM.

This module implements a rolling map that maintains a sliding window of recent LiDAR scans
for use as a target map in ICP-based SLAM. The rolling map helps maintain computational
efficiency by keeping only recent scans and applying spatial cropping.

Key Features:
- Sliding window of recent scans (FIFO)
- Spatial cropping around current position
- Voxel downsampling for computational efficiency
- Memory management with point count limits
"""

import numpy as np
import open3d as o3d
from collections import deque
from typing import Deque, Optional


class RollingMap:
    """Rolling map implementation for LiDAR SLAM.
    
    This class maintains a sliding window of recent LiDAR scans that serves as the target
    map for ICP registration. It provides spatial cropping, downsampling, and memory
    management to keep computational requirements reasonable.
    
    The rolling map works by:
    1. Storing only the last K scans in a circular buffer
    2. Cropping scans spatially around the current position
    3. Downsampling the merged map when it gets too large
    4. Providing the merged map for ICP target matching
    
    Attributes:
        keep_last_k_scans: Number of recent scans to keep in memory
        voxel_size_map: Voxel size for downsampling the merged map (m)
        crop_radius_m: Spatial cropping radius around current position (m)
        max_points: Maximum points in merged map before downsampling
        scans: Circular buffer storing recent point clouds
    """
    
    def __init__(self, keep_last_k_scans: int, voxel_size_map: float, crop_radius_m: float, max_points: int):
        """Initialize the rolling map.
        
        Args:
            keep_last_k_scans: Number of recent scans to keep in sliding window
            voxel_size_map: Voxel size for downsampling merged map (m)
            crop_radius_m: Spatial cropping radius around current position (m)
            max_points: Maximum points in merged map before downsampling
        """
        self.keep_last_k_scans = keep_last_k_scans
        self.voxel_size_map = voxel_size_map
        self.crop_radius_m = crop_radius_m
        self.max_points = max_points
        self.scans: Deque[o3d.geometry.PointCloud] = deque(maxlen=keep_last_k_scans)

    def add_scan(self, pcd_world: o3d.geometry.PointCloud, center_world: Optional[np.ndarray] = None) -> None:
        """Add a new scan to the rolling map.
        
        This method adds a new point cloud scan to the rolling map buffer. The scan is
        optionally cropped spatially around the current position to limit memory usage
        and focus on the local environment.
        
        The scan is added to a circular buffer (deque) that automatically removes the
        oldest scan when the buffer is full, maintaining a sliding window of recent scans.
        
        Args:
            pcd_world: Point cloud in world coordinates to add
            center_world: Current position in world coordinates for spatial cropping.
                         If None, uses origin (0,0,0). If crop_radius_m <= 0, cropping is disabled.
        """
        # Apply spatial cropping around current position if enabled
        if self.crop_radius_m > 0:
            if center_world is None:
                center_world = np.zeros(3)
            c = center_world.reshape(3)
            
            # Create bounding box for spatial cropping
            min_b = c - self.crop_radius_m
            max_b = c + self.crop_radius_m
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
            
            # Crop the point cloud to the bounding box
            pcd_world = pcd_world.crop(bbox)
        
        # Add the (possibly cropped) scan to the circular buffer
        # The deque automatically removes the oldest scan when full
        self.scans.append(pcd_world)

    def build_target_map(self) -> o3d.geometry.PointCloud:
        """Build the target map by merging recent scans.
        
        This method creates the target map for ICP registration by merging all scans
        in the rolling buffer. The merged map is downsampled if it exceeds the maximum
        point count to maintain computational efficiency.
        
        The target map represents the local environment built from recent LiDAR scans
        and serves as the reference for ICP point-to-plane matching.
        
        Returns:
            Merged point cloud representing the target map. Returns empty point cloud
            if no scans are available.
        """
        # Return empty point cloud if no scans available
        if not self.scans:
            return o3d.geometry.PointCloud()
        
        # Merge all scans in the rolling buffer
        merged = o3d.geometry.PointCloud()
        for scan in self.scans:
            merged += scan  # Concatenate point clouds
        
        # Apply downsampling if the merged map is too large
        # This helps maintain computational efficiency for ICP
        if len(merged.points) > self.max_points:
            merged = merged.voxel_down_sample(self.voxel_size_map)
        
        return merged

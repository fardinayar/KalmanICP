import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Protocol
import os


class IcpCovarianceProvider(Protocol):
    """
    Abstract provider for per-frame ICP measurement covariance.

    Implementations can provide different covariances per frame and sequence.
    The returned matrix must match the residual ordering used by the EKF update:
    r = [t_x, t_y, t_z, rot_x, rot_y, rot_z].
    """

    def get_covariance(self, date: str, drive: str, frame_idx: int) -> np.ndarray:
        """
        Return a 6x6 covariance matrix for the given sequence and frame index.
        - Translation variances in m^2.
        - Rotation variances in rad^2.
        """
        ...


class FileBasedCovarianceProvider:
    """
    Concrete implementation of IcpCovarianceProvider that loads covariance data from files.
    
    This provider loads point cloud covariance data from the outputs directory structure:
    - file_names.txt: Contains frame identifiers for each line
    - {model_name}.txt: Contains 6x6 covariance matrices in flattened format (36 values per line)
    
    The covariance matrices are expected to be in roll,pitch,yaw,x,y,z format and are
    automatically converted to the required [t_x, t_y, t_z, rot_x, rot_y, rot_z] format.
    """
    
    def __init__(self, outputs_dir: str, model_name: str = "pointnet"):
        """
        Initialize the covariance provider.
        
        Args:
            outputs_dir: Path to the outputs directory containing covariance files
            model_name: Name of the model (e.g., "pointnet", "cylinder3d") to load covariance data
        """
        self.outputs_dir = outputs_dir
        self.model_name = model_name
        self.file_names_path = os.path.join(outputs_dir, "file_names.txt")
        self.covariance_path = os.path.join(outputs_dir, f"{model_name}.txt")
        
        # Load file names and covariance data
        self._load_data()
    
    def _load_data(self):
        """Load file names and covariance data from files."""
        if not os.path.exists(self.file_names_path):
            raise FileNotFoundError(f"File names not found: {self.file_names_path}")
        if not os.path.exists(self.covariance_path):
            raise FileNotFoundError(f"Covariance file not found: {self.covariance_path}")
        
        # Load file names (first column of each line)
        self.file_names = []
        self.sequence_info = {}  # Store sequence info: {sequence_id: (start_frame, end_frame, count)}
        
        with open(self.file_names_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    file_name = parts[0]  # First column is the file name
                    self.file_names.append(file_name)
                    
                    # Parse sequence and frame from file name (format: XX_YYYYYY)
                    if '_' in file_name:
                        seq_part, frame_part = file_name.split('_', 1)
                        try:
                            sequence_id = int(seq_part)
                            frame_id = int(frame_part)
                            
                            if sequence_id not in self.sequence_info:
                                self.sequence_info[sequence_id] = {'start_frame': frame_id, 'end_frame': frame_id, 'count': 0}
                            
                            self.sequence_info[sequence_id]['start_frame'] = min(self.sequence_info[sequence_id]['start_frame'], frame_id)
                            self.sequence_info[sequence_id]['end_frame'] = max(self.sequence_info[sequence_id]['end_frame'], frame_id)
                            self.sequence_info[sequence_id]['count'] += 1
                        except ValueError:
                            # If parsing fails, treat as single sequence
                            if 'unknown' not in self.sequence_info:
                                self.sequence_info['unknown'] = {'start_frame': 0, 'end_frame': len(self.file_names)-1, 'count': 0}
                            self.sequence_info['unknown']['count'] += 1
        
        # Load covariance matrices (36 values per line, representing 6x6 matrix)
        self.covariances = []
        with open(self.covariance_path, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                if len(values) != 36:
                    raise ValueError(f"Expected 36 values per line, got {len(values)}")
                # Reshape to 6x6 matrix
                cov_matrix = np.array(values).reshape(6, 6)
                self.covariances.append(cov_matrix)
        
        if len(self.file_names) != len(self.covariances):
            raise ValueError(f"Mismatch between file names ({len(self.file_names)}) and covariances ({len(self.covariances)})")
    
    def get_covariance(self, date: str, drive: str, frame_idx: int) -> np.ndarray:
        """
        Return a 6x6 covariance matrix for the given sequence and frame index.
        
        Args:
            date: Date string (not used in this implementation)
            drive: Drive string (not used in this implementation) 
            frame_idx: Frame index to retrieve covariance for
            
        Returns:
            6x6 covariance matrix in [t_x, t_y, t_z, rot_x, rot_y, rot_z] format
        """
        if frame_idx < 0 or frame_idx >= len(self.covariances):
            raise IndexError(f"Frame index {frame_idx} out of range [0, {len(self.covariances)-1}]")
        
        # Get the raw covariance matrix
        raw_cov = self.covariances[frame_idx]
        
        # Convert from roll,pitch,yaw,x,y,z format to t_x,t_y,t_z,rot_x,rot_y,rot_z format
        # The input matrix is ordered as [roll, pitch, yaw, x, y, z]
        # We need to reorder to [x, y, z, roll, pitch, yaw] and convert angles to radians
        
        # Extract rotation and translation parts
        rot_cov = raw_cov[:3, :3]  # roll, pitch, yaw covariance
        trans_cov = raw_cov[3:, 3:]  # x, y, z covariance
        cross_cov = raw_cov[:3, 3:]  # rotation-translation cross terms
        cross_cov_T = raw_cov[3:, :3]  # translation-rotation cross terms
        
        # Convert rotation variances from deg^2 to rad^2
        deg_to_rad = np.pi / 180.0
        rot_cov_rad = rot_cov * (deg_to_rad ** 2)
        cross_cov_rad = cross_cov * deg_to_rad
        cross_cov_T_rad = cross_cov_T * deg_to_rad
        
        # Build the reordered covariance matrix [t_x, t_y, t_z, rot_x, rot_y, rot_z]
        reordered_cov = np.zeros((6, 6))
        reordered_cov[:3, :3] = trans_cov  # translation-translation
        reordered_cov[3:, 3:] = rot_cov_rad  # rotation-rotation (in rad^2)
        reordered_cov[:3, 3:] = cross_cov_T_rad  # translation-rotation cross terms
        reordered_cov[3:, :3] = cross_cov_rad  # rotation-translation cross terms
        
        return reordered_cov
    
    def get_file_name(self, frame_idx: int) -> str:
        """
        Get the file name for a given frame index.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            File name string
        """
        if frame_idx < 0 or frame_idx >= len(self.file_names):
            raise IndexError(f"Frame index {frame_idx} out of range [0, {len(self.file_names)-1}]")
        return self.file_names[frame_idx]
    
    def num_frames(self) -> int:
        """Return the number of available frames."""
        return len(self.file_names)
    
    def get_available_sequences(self) -> dict:
        """
        Get all available sequence names and their frame ranges.
        
        Returns:
            Dictionary mapping sequence IDs to their information:
            {
                sequence_id: {
                    'start_frame': int,  # First frame number in sequence
                    'end_frame': int,    # Last frame number in sequence  
                    'count': int,        # Number of frames in sequence
                    'frame_range': tuple # (start_frame, end_frame) inclusive
                }
            }
        """
        return self.sequence_info.copy()
    
    def get_sequence_frame_indices(self, sequence_id: int) -> list:
        """
        Get the global frame indices for a specific sequence.
        
        Args:
            sequence_id: The sequence ID to get frame indices for
            
        Returns:
            List of global frame indices that belong to this sequence
        """
        if sequence_id not in self.sequence_info:
            raise ValueError(f"Sequence {sequence_id} not found. Available sequences: {list(self.sequence_info.keys())}")
        
        indices = []
        for i, file_name in enumerate(self.file_names):
            if '_' in file_name:
                seq_part, frame_part = file_name.split('_', 1)
                try:
                    if int(seq_part) == sequence_id:
                        indices.append(i)
                except ValueError:
                    continue
        
        return sorted(indices)
    
    def get_sequence_info_summary(self) -> str:
        """
        Get a human-readable summary of all available sequences.
        
        Returns:
            String summary of sequences and their frame ranges
        """
        summary = f"Available sequences for model '{self.model_name}':\n"
        summary += f"Total frames: {self.num_frames()}\n\n"
        
        for seq_id, info in sorted(self.sequence_info.items()):
            summary += f"Sequence {seq_id}:\n"
            summary += f"  Frame range: {info['start_frame']:06d} - {info['end_frame']:06d}\n"
            summary += f"  Frame count: {info['count']}\n"
            summary += f"  Global indices: {len(self.get_sequence_frame_indices(seq_id))} frames\n\n"
        
        return summary


def preprocess_cloud(points_xyzr: np.ndarray, voxel_size: float, normal_radius: float, normal_max_nn: int) -> o3d.geometry.PointCloud:
    """Preprocess point cloud for ICP by downsampling and estimating normals.
    
    Args:
        points_xyzr: Point cloud data (Nx4) with [x,y,z,reflectance]
        voxel_size: Voxel size for downsampling (m)
        normal_radius: Radius for normal estimation (m)
        normal_max_nn: Maximum neighbors for normal estimation
        
    Returns:
        Preprocessed point cloud with normals
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyzr[:, :3])
    
    # Downsample to reduce computational load
    pcd = pcd.voxel_down_sample(voxel_size)
    
    # Estimate surface normals using local neighborhood
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    
    # For LiDAR data, skip normal orientation - let ICP handle normal inconsistencies
    # This avoids introducing systematic errors from arbitrary orientation choices
    
    return pcd


def icp_point_to_plane(source: o3d.geometry.PointCloud,
                       target: o3d.geometry.PointCloud,
                       init_T: np.ndarray,
                       max_correspondence_dist: float,
                       max_iters: int,
                       robust_delta: float) -> Tuple[np.ndarray, float, int]:
    """Point-to-plane ICP registration.
    
    Args:
        source: Source point cloud
        target: Target point cloud  
        init_T: Initial transformation guess (4x4)
        max_correspondence_dist: Maximum correspondence distance (m)
        max_iters: Maximum iterations
        robust_delta: Huber loss parameter
        
    Returns:
        (transformation, rmse, correspondence_count)
    """
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters)
    loss = o3d.pipelines.registration.HuberLoss(robust_delta)
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_dist,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        criteria,
    )
    return result.transformation, result.inlier_rmse, np.asarray(result.correspondence_set).shape[0]


def icp_point_to_point(source: o3d.geometry.PointCloud,
                       target: o3d.geometry.PointCloud,
                       init_T: np.ndarray,
                       max_correspondence_dist: float,
                       max_iters: int) -> Tuple[np.ndarray, float, int]:
    """Point-to-point ICP registration (no normals required).
    
    Args:
        source: Source point cloud
        target: Target point cloud
        init_T: Initial transformation guess (4x4)
        max_correspondence_dist: Maximum correspondence distance (m)
        max_iters: Maximum iterations
        
    Returns:
        (transformation, rmse, correspondence_count)
    """
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iters)
    result = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_dist,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )
    return result.transformation, result.inlier_rmse, np.asarray(result.correspondence_set).shape[0]


def build_measurement_covariance(diag: np.ndarray) -> np.ndarray:
    """
    Build 6x6 measurement covariance matching residual ordering r = [t_x, t_y, t_z, rot_x, rot_y, rot_z].

    Expected input ordering is [sigma_roll^2, sigma_pitch^2, sigma_yaw^2, sigma_x^2, sigma_y^2, sigma_z^2]
    with angles provided in deg^2. We reorder to [trans, rot] and convert angle variances to rad^2.
    """
    vals = np.asarray(diag, dtype=float).reshape(6)
    # Extract as provided (rot in deg^2 first, then trans in m^2)
    rot_deg2 = vals[:3]
    trans_m2 = vals[3:]
    # Convert rotation variances deg^2 -> rad^2
    rot_rad2 = rot_deg2 * ((np.pi / 180.0) ** 2)
    # Reorder to [trans, rot]
    out = np.hstack([trans_m2, rot_rad2])
    return np.diag(out)

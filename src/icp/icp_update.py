import numpy as np
import open3d as o3d
from typing import Optional, Tuple, Protocol


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

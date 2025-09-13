import os
import argparse
import numpy as np
from typing import List, Tuple
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Local imports
from src.icp.icp_update import FileBasedCovarianceProvider


def _load_kitti_odometry_poses(poses_path: str) -> np.ndarray:
    """Load KITTI odometry poses text file into (N,4,4) T_w_cam matrices.

    Each line: 12 floats row-major 3x4 [R|t].
    """
    poses: List[np.ndarray] = []
    with open(poses_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals = [float(x) for x in s.split()]
            if len(vals) != 12:
                raise ValueError("Each pose line must have 12 floats (3x4 matrix)")
            M = np.array(vals, dtype=float).reshape(3, 4)
            T = np.eye(4)
            T[:3, :4] = M
            poses.append(T)
    return np.asarray(poses, dtype=float)


def _read_odometry_calib_T_cam_velo(seq_dir: str) -> np.ndarray:
    """Read T_cam0_velo (3x4 as 4x4) from KITTI odometry calib.txt (Tr entry).

    Odometry calib has 'Tr:' which is velo->cam0 (i.e., T_cam0_velo).
    """
    calib_file = os.path.join(seq_dir, 'calib.txt')
    if not os.path.exists(calib_file):
        raise FileNotFoundError(calib_file)
    Tr = None
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith('Tr'):
                data = line.split(':', 1)[1].strip().split()
                if len(data) != 12:
                    raise ValueError("calib.txt Tr must have 12 floats")
                Tr = np.array([float(x) for x in data], dtype=float).reshape(3, 4)
                break
    if Tr is None:
        raise KeyError("Tr not found in calib.txt")
    T = np.eye(4)
    T[:3, :4] = Tr
    return T


def _load_velodyne_bin(bin_path: str) -> np.ndarray:
    """Load one KITTI velodyne .bin as (N,4) float32 [x,y,z,reflectance]."""
    pts = np.fromfile(bin_path, dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Invalid velodyne file size: {bin_path}")
    return pts.reshape(-1, 4)


def _transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    N = pts_xyz.shape[0]
    homog = np.hstack([pts_xyz, np.ones((N, 1), dtype=pts_xyz.dtype)])
    out = (T @ homog.T).T
    return out[:, :3]


def _cov_xy_to_ellipse_points(cov_xy: np.ndarray, center_xy: np.ndarray, num_pts: int = 64, nsig: float = 2.0) -> np.ndarray:
    """Return (num_pts,3) 3D points (z=0) forming an ellipse for 2x2 covariance around center.

    - cov_xy: 2x2 covariance for [x,y]
    - center_xy: 2, center in world frame
    - nsig: scale in standard deviations
    """
    vals, vecs = np.linalg.eigh(cov_xy)
    vals = np.clip(vals, a_min=0.0, a_max=None)
    radii = nsig * np.sqrt(vals)
    theta = np.linspace(0.0, 2.0 * np.pi, num_pts, endpoint=True)
    circ = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2,K)
    ell = (vecs @ (radii.reshape(2, 1) * circ))  # (2,K)
    xy = ell.T + center_xy.reshape(1, 2)  # (K,2)
    xyz = np.concatenate([xy, np.zeros((xy.shape[0], 1), dtype=xy.dtype)], axis=1)
    return xyz


def visualize(
    dataset_root: str,
    sequence: int,
    cov_dir: str,
    model_name: str = 'pointnet',
    max_frames: int = 300,
    frame_stride: int = 5,
    point_stride: int = 20,  # More aggressive downsampling
    nsig: float = 2.0,
    max_points: int = 100000,  # Limit total points
    map_point_size: float = 0.1,
    traj_point_size: float = 1.5,
    cov_clip_low: float = 5.0,
    cov_clip_high: float = 97.5,
    cov_gamma: float = 0.5,
):
    """Visualize LiDAR map and overlay per-frame x–y covariance ellipses using viser.

    Assumptions:
    - KITTI odometry layout at dataset_root: sequences/{seq:02d}/(velodyne, calib.txt), poses/{seq:02d}.txt
    - Poses in odometry are in camera-0 frame (T_w_cam0). We convert to LiDAR via T_cam0_velo.
    - Covariance provider: uses file_names.txt with entries like '00_000123'.
    """
    try:
        import viser
    except ImportError as e:
        raise ImportError("viser not installed. Install with `pip install viser`." ) from e

    seq_str = f"{sequence:02d}"
    seq_dir = os.path.join(dataset_root, 'sequences', seq_str)
    poses_path = os.path.join(dataset_root, 'poses', f"{seq_str}.txt")
    if not os.path.exists(seq_dir):
        raise FileNotFoundError(seq_dir)
    if not os.path.exists(poses_path):
        raise FileNotFoundError(poses_path)

    # Load GT poses (world<-cam0)
    T_w_cam_list = _load_kitti_odometry_poses(poses_path)

    # Load cam0<-velo
    T_cam_velo = _read_odometry_calib_T_cam_velo(seq_dir)

    # Covariance provider (file_names first column: e.g., 00_000123)
    provider = FileBasedCovarianceProvider(outputs_dir=cov_dir, model_name=model_name)
    seq_indices = provider.get_sequence_frame_indices(sequence)
    if len(seq_indices) == 0:
        raise ValueError(f"No frames for sequence {sequence} found in covariance files at {cov_dir}")

    # Decide which frames to load
    selected_global = seq_indices[::frame_stride][:max_frames]

    # Accumulate a light-weight LiDAR map and collect trajectory data
    map_points: List[np.ndarray] = []
    trajectory_points: List[np.ndarray] = []
    trajectory_covariances: List[float] = []

    for i_local, gidx in enumerate(selected_global):
        fname = provider.get_file_name(gidx)  # e.g., '00_000123'
        seq_part, frame_part = fname.split('_', 1)
        if int(seq_part) != sequence:
            continue
        frame_id = int(frame_part)

        # Paths
        bin_path = os.path.join(seq_dir, 'velodyne', f"{frame_id:06d}.bin")
        scan = None
        if os.path.exists(bin_path):
            scan = _load_velodyne_bin(bin_path)

        # Pose of LiDAR in world: T_w_l = T_w_cam * T_cam_velo
        if frame_id < 0 or frame_id >= len(T_w_cam_list):
            continue
        T_w_cam = T_w_cam_list[frame_id]
        T_w_l = T_w_cam @ T_cam_velo

        # Add points transformed to world (with aggressive downsampling)
        if scan is not None and scan.shape[0] > 0:
            pts = scan[::max(1, point_stride), :3]
            pts_w = _transform_points(T_w_l, pts)
            map_points.append(pts_w)

        # Store trajectory point and covariance magnitude
        trajectory_points.append(T_w_l[:3, 3])  # LiDAR position
        
        # Covariance: 6x6 -> take x,y block and compute magnitude
        R = provider.get_covariance(date='', drive='', frame_idx=gidx)
        cov_xy = R[:2, :2]
        # Use trace as covariance magnitude (sum of x,y variances)
        cov_magnitude = np.trace(cov_xy)
        trajectory_covariances.append(cov_magnitude)

    # Concatenate map points and limit total points
    if len(map_points) > 0:
        all_pts = np.vstack(map_points)
        # Further downsample if too many points
        if all_pts.shape[0] > max_points:
            indices = np.random.choice(all_pts.shape[0], max_points, replace=False)
            all_pts = all_pts[indices]
    else:
        all_pts = np.zeros((0, 3), dtype=float)
    
    # Convert trajectory data to arrays
    if len(trajectory_points) > 0:
        traj_pts = np.array(trajectory_points)
        traj_covs = np.array(trajectory_covariances)
    else:
        traj_pts = np.zeros((0, 3), dtype=float)
        traj_covs = np.array([])

    # Start viser server and add primitives
    server = viser.ViserServer()

    # Point cloud (downsampled)
    if all_pts.shape[0] > 0:
        colors = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (all_pts.shape[0], 1))
        server.scene.add_point_cloud(
            name=f"map/sequence_{seq_str}",
            points=all_pts.astype(np.float32),
            colors=colors,
            point_size=float(map_point_size),  # User-controlled
        )

    # Trajectory with covariance magnitude as color (green -> red), constant point size
    if traj_pts.shape[0] > 1:
        # Percentile clipping for contrast
        if traj_covs.size > 0:
            cov_low_val = float(np.percentile(traj_covs, float(cov_clip_low)))
            cov_high_val = float(np.percentile(traj_covs, float(cov_clip_high)))
        else:
            cov_low_val, cov_high_val = 0.0, 1.0
        if not np.isfinite(cov_low_val) or not np.isfinite(cov_high_val) or cov_high_val <= cov_low_val:
            cov_low_val = float(np.min(traj_covs)) if traj_covs.size > 0 else 0.0
            cov_high_val = float(np.max(traj_covs)) if traj_covs.size > 0 else 1.0
            if abs(cov_high_val - cov_low_val) < 1e-12:
                cov_high_val = cov_low_val + 1e-12

        all_seg_pts = []
        all_seg_cols = []

        # Create line segments between consecutive trajectory points
        for i in range(traj_pts.shape[0] - 1):
            p1, p2 = traj_pts[i], traj_pts[i + 1]
            avg_cov = (traj_covs[i] + traj_covs[i + 1]) * 0.5
            # Normalize to [0,1] using percentile window
            alpha = float((avg_cov - cov_low_val) / (cov_high_val - cov_low_val))
            alpha = max(0.0, min(1.0, alpha))
            # Gamma shaping for contrast (cov_gamma < 1 boosts high values)
            alpha = float(alpha ** float(cov_gamma))
            # Color map: green (low) to red (high)
            color = np.array([int(255 * alpha), int(255 * (1.0 - alpha)), 0], dtype=np.uint8)

            # Build a thin polyline as a small set of points
            num_seg_pts = 12
            t_values = np.linspace(0, 1, num_seg_pts)
            seg_pts = p1[None, :] + t_values[:, None] * (p2 - p1)[None, :]
            seg_cols = np.tile(color[None, :], (num_seg_pts, 1))
            all_seg_pts.append(seg_pts)
            all_seg_cols.append(seg_cols)

        if len(all_seg_pts) > 0:
            traj_points = np.vstack(all_seg_pts).astype(np.float32)
            traj_colors = np.vstack(all_seg_cols)
            server.scene.add_point_cloud(
                name=f"trajectory/sequence_{seq_str}",
                points=traj_points,
                colors=traj_colors,
                point_size=float(traj_point_size),
            )

    print(f"Loaded {all_pts.shape[0]} map points and {traj_pts.shape[0]} trajectory points for seq {seq_str}.")
    print("Trajectory color encodes XY covariance magnitude: Green (low) -> Red (high)")
    print("Viser running. Open the printed URL in your browser.")
    # Keep server alive until user closes it
    try:
        wait_closed = getattr(server, 'wait_closed', None)
        if callable(wait_closed):
            wait_closed()
        else:
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(description="Visualize x–y covariance on KITTI odometry LiDAR map using viser")
    parser.add_argument('--dataset_root', type=str, default='dataset', help="KITTI odometry root with sequences/ and poses/")
    parser.add_argument('--sequence', type=int, default=0, help="Odometry sequence number (0-10)")
    parser.add_argument('--cov_dir', type=str, default='covariances', help="Directory containing file_names.txt and model covariances")
    parser.add_argument('--model', type=str, default='pointnet', choices=['pointnet', 'cylinder3d'], help="Covariance model file to use")
    parser.add_argument('--max_frames', type=int, default=300, help="Max frames to load for map/ellipses")
    parser.add_argument('--frame_stride', type=int, default=1, help="Stride between frames to reduce load")
    parser.add_argument('--point_stride', type=int, default=100, help="Subsample LiDAR points by stride")
    parser.add_argument('--max_points', type=int, default=100000, help="Maximum total points in map")
    parser.add_argument('--nsig', type=float, default=2.0, help="Ellipse scale in standard deviations (unused in trajectory mode)")
    parser.add_argument('--map_point_size', type=float, default=0.1, help="Point size for map points")
    parser.add_argument('--traj_point_size', type=float, default=3, help="Point size for trajectory rendering")
    parser.add_argument('--cov_clip_low', type=float, default=5.0, help="Lower percentile for covariance clipping (improves contrast)")
    parser.add_argument('--cov_clip_high', type=float, default=97.5, help="Upper percentile for covariance clipping (improves contrast)")
    parser.add_argument('--cov_gamma', type=float, default=0.5, help="Gamma for covariance color mapping (<1 increases contrast)")
    args = parser.parse_args()
    
    visualize(
        dataset_root=args.dataset_root,
        sequence=args.sequence,
        cov_dir=args.cov_dir,
        model_name=args.model,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        point_stride=args.point_stride,
        max_points=args.max_points,
        nsig=args.nsig,
        map_point_size=args.map_point_size,
        traj_point_size=args.traj_point_size,
        cov_clip_low=args.cov_clip_low,
        cov_clip_high=args.cov_clip_high,
        cov_gamma=args.cov_gamma,
    )


if __name__ == '__main__':
    main()


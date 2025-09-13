import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import pykitti
from datetime import datetime


# KITTI Odometry sequence mapping
KITTI_ODOMETRY_SEQUENCES = {
    0: ("2011_10_03_drive_0027", 0, 4540),
    1: ("2011_10_03_drive_0042", 0, 1100),
    2: ("2011_10_03_drive_0034", 0, 4660),
    3: ("2011_09_26_drive_0067", 0, 800),
    4: ("2011_09_30_drive_0016", 0, 270),
    5: ("2011_09_30_drive_0018", 0, 2760),
    6: ("2011_09_30_drive_0020", 0, 1100),
    7: ("2011_09_30_drive_0027", 0, 1100),
    8: ("2011_09_30_drive_0028", 1100, 5170),
    9: ("2011_09_30_drive_0033", 0, 1590),
    10: ("2011_09_30_drive_0034", 0, 1200),
}


@dataclass
class SequenceSpec:
    root_dir: str
    date: str
    drive: str
    frames: Optional[List[int]]
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    # New fields for odometry support
    odometry_sequence: Optional[int] = None  # KITTI odometry sequence number (0-10)
    is_odometry: bool = False  # Whether this is an odometry sequence


def _read_kitti_timestamps(path: str) -> np.ndarray:
    times: List[float] = []
    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if '.' in s:
                base, frac = s.split('.')
                frac6 = (frac + '000000')[:6]
                dt = datetime.strptime(base, '%Y-%m-%d %H:%M:%S')
                dt = dt.replace(microsecond=int(frac6))
            else:
                dt = datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
            times.append(dt.timestamp())
    return np.asarray(times, dtype=float)


class KITTIRawLoader:
    def __init__(self, spec: SequenceSpec):
        self.spec = spec
        self.dataset = pykitti.raw(spec.root_dir, spec.date, spec.drive)
        seq_dir = os.path.join(spec.root_dir, spec.date, f"{spec.date}_drive_{spec.drive}_sync")

        lidar_ts_file = os.path.join(seq_dir, 'velodyne_points', 'timestamps.txt')
        imu_ts_file = os.path.join(seq_dir, 'oxts', 'timestamps.txt')

        if os.path.exists(lidar_ts_file):
            self.lidar_timestamps_all = _read_kitti_timestamps(lidar_ts_file)
        else:
            raise FileNotFoundError(f"Missing LiDAR timestamps: {lidar_ts_file}")

        if os.path.exists(imu_ts_file):
            self.imu_timestamps = _read_kitti_timestamps(imu_ts_file)
        else:
            raise FileNotFoundError(f"Missing IMU timestamps: {imu_ts_file}")

        # Handle odometry sequences with frame offset
        if spec.is_odometry and spec.odometry_sequence is not None:
            # For odometry sequences, we need to apply the frame offset
            _, odom_start, _ = KITTI_ODOMETRY_SEQUENCES[spec.odometry_sequence]
            self.odometry_offset = odom_start
        else:
            self.odometry_offset = 0

        # Apply frame subset if provided
        if spec.frames and len(spec.frames) > 0:
            idx = np.array(spec.frames, dtype=int)
            # For odometry sequences, add the offset
            if spec.is_odometry:
                idx = idx + self.odometry_offset
            # Validate indices within range
            if np.any(idx < 0) or np.any(idx >= len(self.lidar_timestamps_all)):
                raise IndexError("One or more requested frame indices are out of range for this sequence")
            self.frame_indices = idx
            self.lidar_timestamps = self.lidar_timestamps_all[idx]
        elif (spec.start_frame is not None) or (spec.end_frame is not None):
            n = len(self.lidar_timestamps_all)
            start = int(spec.start_frame) if spec.start_frame is not None else 0
            end = int(spec.end_frame) if spec.end_frame is not None else (n - 1)
            
            # For odometry sequences, add the offset
            if spec.is_odometry:
                start = start + self.odometry_offset
                end = end + self.odometry_offset
            
            if start < 0 or end < 0 or start >= n or end >= n or start > end:
                raise IndexError("Invalid start/end frame range for this sequence")
            idx = np.arange(start, end + 1, dtype=int)
            self.frame_indices = idx
            self.lidar_timestamps = self.lidar_timestamps_all[idx]
        else:
            self.frame_indices = None
            self.lidar_timestamps = self.lidar_timestamps_all

    def num_lidar(self) -> int:
        return len(self.lidar_timestamps)

    def get_velodyne(self, idx: int) -> np.ndarray:
        # Returns Nx4 (x,y,z,reflectance)
        dataset_idx = int(self.frame_indices[idx]) if self.frame_indices is not None else idx
        return self.dataset.get_velo(dataset_idx)

    def get_imu_packet(self, idx: int):
        return self.dataset.oxts[idx]

    def get_nearest_imu_indices(self, t_start: float, t_end: float) -> Tuple[int, int]:
        # Return index range in imu_timestamps that spans [t_start, t_end]
        i0 = int(np.searchsorted(self.imu_timestamps, t_start, side='left'))
        i1 = int(np.searchsorted(self.imu_timestamps, t_end, side='right'))
        i0 = max(0, i0 - 1)
        i1 = min(len(self.imu_timestamps) - 1, i1)
        return i0, i1

    def get_imu_measurements(self, i0: int, i1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Returns times, gyro (rad/s), accel (m/s^2)
        times = self.imu_timestamps[i0:i1 + 1]
        omegas = []
        accs = []
        for k in range(i0, i1 + 1):
            pkt = self.dataset.oxts[k]
            omegas.append(np.array([pkt.packet.wx, pkt.packet.wy, pkt.packet.wz], dtype=float))
            accs.append(np.array([pkt.packet.ax, pkt.packet.ay, pkt.packet.az], dtype=float))
        return times, np.asarray(omegas), np.asarray(accs)


class KITTIOdometryLoader:
    """
    KITTI Odometry dataset loader.
    
    This loader provides access to KITTI odometry sequences (00-10) with
    LiDAR and IMU data. It automatically maps odometry sequence numbers
    to the corresponding raw dataset sequences.
    """
    
    def __init__(self, root_dir: str, sequence: int, start_frame: Optional[int] = None, 
                 end_frame: Optional[int] = None, frames: Optional[List[int]] = None):
        """
        Initialize KITTI odometry loader.
        
        Args:
            root_dir: Root directory of KITTI dataset
            sequence: Odometry sequence number (0-10)
            start_frame: Start frame index (optional)
            end_frame: End frame index (optional)
            frames: Specific frame indices (optional)
        """
        if sequence not in KITTI_ODOMETRY_SEQUENCES:
            raise ValueError(f"Invalid odometry sequence {sequence}. Available: {list(KITTI_ODOMETRY_SEQUENCES.keys())}")
        
        self.sequence = sequence
        self.root_dir = root_dir
        
        # Get the raw sequence info
        raw_sequence, odom_start, odom_end = KITTI_ODOMETRY_SEQUENCES[sequence]
        date, drive = raw_sequence.split('_drive_')
        
        # Calculate frame range
        if frames is not None:
            # Use specific frame indices
            self.frame_indices = np.array(frames, dtype=int)
            # Validate indices
            if np.any(self.frame_indices < odom_start) or np.any(self.frame_indices > odom_end):
                raise IndexError(f"Frame indices must be in range [{odom_start}, {odom_end}] for sequence {sequence}")
        else:
            # Use start/end frame range
            start = start_frame if start_frame is not None else odom_start
            end = end_frame if end_frame is not None else odom_end
            
            if start < odom_start or end > odom_end or start > end:
                raise IndexError(f"Frame range [{start}, {end}] invalid for sequence {sequence}. Valid range: [{odom_start}, {odom_end}]")
            
            self.frame_indices = np.arange(start, end + 1, dtype=int)
        
        # Create the underlying raw loader
        self.raw_loader = KITTIRawLoader(SequenceSpec(
            root_dir=root_dir,
            date=date,
            drive=drive,
            frames=self.frame_indices.tolist(),
            is_odometry=True,
            odometry_sequence=sequence
        ))
        
        # Store odometry-specific info
        self.odom_start = odom_start
        self.odom_end = odom_end
    
    def num_lidar(self) -> int:
        """Return number of LiDAR frames."""
        return len(self.frame_indices)
    
    def get_velodyne(self, idx: int) -> np.ndarray:
        """
        Get LiDAR point cloud for frame index.
        
        Args:
            idx: Frame index in the odometry sequence
            
        Returns:
            Point cloud as Nx4 array (x, y, z, reflectance)
        """
        return self.raw_loader.get_velodyne(idx)
    
    def get_imu_packet(self, idx: int):
        """Get IMU packet for frame index."""
        return self.raw_loader.get_imu_packet(idx)
    
    def get_nearest_imu_indices(self, t_start: float, t_end: float) -> Tuple[int, int]:
        """Get IMU indices for time range."""
        return self.raw_loader.get_nearest_imu_indices(t_start, t_end)
    
    def get_imu_measurements(self, i0: int, i1: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get IMU measurements for index range."""
        return self.raw_loader.get_imu_measurements(i0, i1)
    
    def get_odometry_frame_range(self) -> Tuple[int, int]:
        """Get the odometry frame range for this sequence."""
        return self.odom_start, self.odom_end
    
    def get_sequence_info(self) -> dict:
        """Get information about this odometry sequence."""
        raw_sequence, _, _ = KITTI_ODOMETRY_SEQUENCES[self.sequence]
        return {
            'sequence': self.sequence,
            'raw_sequence': raw_sequence,
            'odometry_range': (self.odom_start, self.odom_end),
            'total_frames': self.odom_end - self.odom_start + 1,
            'loaded_frames': len(self.frame_indices),
            'frame_indices': self.frame_indices.tolist()
        }


def create_odometry_sequence_spec(root_dir: str, sequence: int, 
                                 start_frame: Optional[int] = None,
                                 end_frame: Optional[int] = None,
                                 frames: Optional[List[int]] = None) -> SequenceSpec:
    """
    Create a SequenceSpec for a KITTI odometry sequence.
    
    Args:
        root_dir: Root directory of KITTI dataset
        sequence: Odometry sequence number (0-10)
        start_frame: Start frame index (optional)
        end_frame: End frame index (optional)
        frames: Specific frame indices (optional)
        
    Returns:
        SequenceSpec configured for the odometry sequence
    """
    if sequence not in KITTI_ODOMETRY_SEQUENCES:
        raise ValueError(f"Invalid odometry sequence {sequence}. Available: {list(KITTI_ODOMETRY_SEQUENCES.keys())}")
    
    raw_sequence, odom_start, odom_end = KITTI_ODOMETRY_SEQUENCES[sequence]
    date, drive = raw_sequence.split('_drive_')
    
    return SequenceSpec(
        root_dir=root_dir,
        date=date,
        drive=drive,
        frames=frames,
        start_frame=start_frame,
        end_frame=end_frame,
        odometry_sequence=sequence,
        is_odometry=True
    )

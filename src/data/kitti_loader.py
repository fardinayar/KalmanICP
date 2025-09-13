import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pykitti
from datetime import datetime


@dataclass
class SequenceSpec:
    root_dir: str
    date: str
    drive: str
    frames: Optional[List[int]]
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None


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

        # Apply frame subset if provided
        if spec.frames and len(spec.frames) > 0:
            idx = np.array(spec.frames, dtype=int)
            # Validate indices within range
            if np.any(idx < 0) or np.any(idx >= len(self.lidar_timestamps_all)):
                raise IndexError("One or more requested frame indices are out of range for this sequence")
            self.frame_indices = idx
            self.lidar_timestamps = self.lidar_timestamps_all[idx]
        elif (spec.start_frame is not None) or (spec.end_frame is not None):
            n = len(self.lidar_timestamps_all)
            start = int(spec.start_frame) if spec.start_frame is not None else 0
            end = int(spec.end_frame) if spec.end_frame is not None else (n - 1)
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

import os
import yaml
import time
import numpy as np
from typing import Any, Dict, Tuple
from .geometry import write_kitti_poses_txt, T_from_R_t


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_run_dir(root: str, date: str, drive: str, suffix: str = None) -> str:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    name = f"{stamp}_{date}_{drive}"
    if suffix:
        name = f"{name}_{suffix}"
    run_dir = os.path.join(root, name)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'debug'), exist_ok=True)
    return run_dir


def save_kitti_trajectory(path: str, Twb_list: np.ndarray) -> None:
    write_kitti_poses_txt(path, Twb_list)


def list_to_T(array_16: list) -> np.ndarray:
    T = np.array(array_16, dtype=float).reshape(4, 4)
    return T


def _read_kitti_matrix_tr(path: str, key: str) -> np.ndarray:
    with open(path, 'r') as f:
        for line in f:
            if line.startswith(key + ':'):
                data = line.split(':', 1)[1].strip().split()
                vals = np.array([float(x) for x in data], dtype=float)
                break
        else:
            raise KeyError(f"Key {key} not found in {path}")
    M = np.eye(4)
    M[:3, :4] = vals.reshape(3, 4)
    return M


def _read_kitti_RT(path: str) -> np.ndarray:
    R = None
    t = None
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('R:'):
                data = line.split(':', 1)[1].strip().split()
                R = np.array([float(x) for x in data], dtype=float).reshape(3, 3)
            elif line.startswith('T:'):
                data = line.split(':', 1)[1].strip().split()
                t = np.array([float(x) for x in data], dtype=float).reshape(3)
    if R is None or t is None:
        raise KeyError(f"R/T not found in {path}")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_T_imu_velo_from_kitti(root_dir: str, date: str) -> np.ndarray:
    calib_path = os.path.join(root_dir, date, 'calib_imu_to_velo.txt')
    if not os.path.exists(calib_path):
        raise FileNotFoundError(calib_path)
    # Some KITTI files provide R/T separately; some provide Tr (3x4)
    try:
        T_imu_to_velo = _read_kitti_RT(calib_path)
    except KeyError:
        T_imu_to_velo = _read_kitti_matrix_tr(calib_path, 'Tr')
    # We need IMU<-VELO, i.e., velo->imu
    T_velo_to_imu = np.linalg.inv(T_imu_to_velo)
    return T_velo_to_imu

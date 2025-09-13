import os
import argparse
import numpy as np
from typing import List

from src.utils.io import load_yaml_config, ensure_run_dir, save_kitti_trajectory
from src.data.kitti_loader import KITTIRawLoader, SequenceSpec


def export_gt(config_path: str, date: str, drive: str):
    cfg = load_yaml_config(config_path)
    run_dir = ensure_run_dir(cfg['logging']['run_root'], date, drive, suffix='gt')

    spec = SequenceSpec(root_dir=cfg['dataset']['root_dir'], date=date, drive=drive, frames=cfg['dataset'].get('frames', []))
    ds = KITTIRawLoader(spec)

    # pykitti provides T_w_oxts per OXTS packet via dataset.oxts[i].T_w_imu
    Twb_list: List[np.ndarray] = []

    for i in range(ds.num_lidar()):
        # Nearest OXTS to lidar timestamp
        t = ds.lidar_timestamps[i]
        idx = int(np.searchsorted(ds.imu_timestamps, t, side='left'))
        idx = np.clip(idx, 0, len(ds.imu_timestamps) - 1)
        T_w_imu = ds.dataset.oxts[idx].T_w_imu
        Twb_list.append(T_w_imu)

    traj_arr = np.stack(Twb_list, axis=0)
    out_gt = os.path.join(run_dir, 'trajectory_gt.kitti')
    save_kitti_trajectory(out_gt, traj_arr)
    print(f"Saved ground-truth trajectory: {out_gt}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--drive', type=str, required=True)
    args = parser.parse_args()
    export_gt(args.config, args.date, args.drive)

"""
Run IMU+ICP SLAM on KITTI Raw.

Pipeline overview:
- Load dataset and calibration (IMU<-LiDAR extrinsic).
- Maintain IMU pose with a pose EKF (prediction) and fuse ICP poses (update).
- Optionally smooth ICP poses with a lightweight EKF before fusing/using them.
- Keep a rolling map for ICP target points.

Modes:
- ekf: IMU predict + ICP update
- imu_only: IMU predict only
- lidar_only: ICP-only odometry
- cakf: Smooth ICP pose (position + global rotation) then set pose directly

Frames:
- T_wi: world<-IMU, T_il: IMU<-LiDAR, T_wl: world<-LiDAR
"""
import os
import json
import argparse
import numpy as np
from typing import List, Optional
import time
from tqdm import tqdm
import subprocess

from src.utils.io import load_yaml_config, ensure_run_dir, list_to_T, save_kitti_trajectory, load_T_imu_velo_from_kitti
from src.utils.geometry import invert_se3
from src.data.kitti_loader import KITTIRawLoader, SequenceSpec
from src.estimation.pose_ekf import PoseEKF, PoseEKFConfig
from src.estimation.icp_cakf import ConstantAccelerationKF, CAEKFConfig
from src.icp.icp_update import preprocess_cloud, icp_point_to_plane, build_measurement_covariance, IcpCovarianceProvider
from src.mapping.rolling_map import RollingMap
from src.eval.evo_eval import save_evo_metrics_json


def _load_T_il(cfg: dict, date: str) -> np.ndarray:
    """Load IMU<-LiDAR extrinsic from KITTI or config."""
    if cfg.get('calibration', {}).get('auto_load_imu_velo', False):
        try:
            T_il = load_T_imu_velo_from_kitti(cfg['dataset']['root_dir'], date)
            print("Loaded T_imu_velo from KITTI calib_imu_to_velo.txt")
            return T_il
        except Exception as e:
            print(f"Failed to auto-load T_imu_velo: {e}. Falling back to config value.")
    return list_to_T(cfg['calibration']['T_imu_velo'])


def _init_pose(cfg: dict, ds: KITTIRawLoader) -> tuple:
    """Initialize pose (R0, p0) from config and optionally from OXTS."""
    from src.utils.geometry import R_from_rpy_deg
    R0 = R_from_rpy_deg(cfg['filter']['init']['R0_deg_rpy'])
    p0 = np.array(cfg['filter']['init']['p0'], dtype=float)

    if cfg['filter'].get('init_from_oxts', False):
        if hasattr(ds, 'lidar_timestamps') and len(ds.lidar_timestamps) > 0:
            t0 = ds.lidar_timestamps[0]
            idx0 = int(np.searchsorted(ds.imu_timestamps, t0, side='left'))
            idx0 = np.clip(idx0, 0, len(ds.imu_timestamps) - 1)
            T_w_imu0 = ds.dataset.oxts[idx0].T_w_imu
            R0 = T_w_imu0[:3, :3]
            p0 = T_w_imu0[:3, 3]
            print("Initialized R0,p0 from OXTS")
    return R0, p0


def _make_pose_ekf(cfg: dict, R0: np.ndarray, p0: np.ndarray) -> PoseEKF:
    """Construct PoseEKF from config and initial pose."""
    ekf_cfg = PoseEKFConfig(
        sigma_g=float(cfg['filter']['noise']['sigma_g']),
        sigma_a=float(cfg['filter']['noise']['sigma_a']),
        gravity=np.array(cfg['filter']['gravity'], dtype=float),
        gyro_bias=np.array(cfg['filter']['static_bias_comp']['gyro_bias'], dtype=float),
        accel_bias=np.array(cfg['filter']['static_bias_comp']['accel_bias'], dtype=float),
        P0_diag=np.array(cfg['filter']['init']['P0_diag'], dtype=float),
        use_joseph_form=bool(cfg['filter']['ekf']['use_joseph_form']),
        min_dt=float(cfg['filter']['ekf']['min_dt']),
        max_dt=float(cfg['filter']['ekf']['max_dt']),
    )
    return PoseEKF(R0=R0, p0=p0, config=ekf_cfg)


def _make_cakf_if_enabled(cfg: dict, mode: str) -> ConstantAccelerationKF:
    """Construct ICP smoother EKF (global rotation + position) if mode=cakf."""
    if mode != 'cakf':
        return None
    cakf_cfg = CAEKFConfig(
            q_acc=float(cfg['cakf']['q_acc']),
            r_pos=float(cfg['icp']['measurement']['R_const_diag'][3]),
            p0_pos=float(cfg['cakf']['p0_pos']),
            p0_vel=float(cfg['cakf']['p0_vel']),
        )
    if 'q_rot_acc' in cfg['cakf']:
        cakf_cfg.q_rot_acc = float(cfg['cakf']['q_rot_acc'])
    # Derive rotational measurement std from ICP diag (deg^2 -> rad)
    if 'measurement' in cfg['icp'] and 'R_const_diag' in cfg['icp']['measurement']:
        # Use yaw variance as a representative; convert deg^2 -> rad^2 and then std
        rot_var_deg2 = float(cfg['icp']['measurement']['R_const_diag'][2])
        rot_std_rad = (np.sqrt(rot_var_deg2) * np.pi / 180.0)
        cakf_cfg.r_rot = rot_std_rad
    if 'p0_rot' in cfg['cakf']:
        cakf_cfg.p0_rot = float(cfg['cakf']['p0_rot'])
    if 'p0_rot_vel' in cfg['cakf']:
        cakf_cfg.p0_rot_vel = float(cfg['cakf']['p0_rot_vel'])
    return ConstantAccelerationKF(cakf_cfg)


def _imu_predict_between(ekf: PoseEKF,
                         cakf: ConstantAccelerationKF,
                         ds: KITTIRawLoader,
                         t_prev: float,
                         t_curr: float,
                         mode: str) -> tuple:
    """Advance filters between two timestamps using all IMU packets in [t_prev, t_curr].

    Returns: (imu_steps, elapsed_seconds)
    """
    i0, i1 = ds.get_nearest_imu_indices(t_prev, t_curr)
    times, omegas, accs = ds.get_imu_measurements(i0, i1)
    t0 = time.time()
    for k in range(len(times) - 1):
        dt = float(times[k + 1] - times[k])
        if mode in ('ekf', 'imu_only'):
            ekf.predict(omegas[k], accs[k], dt)
        if mode == 'cakf' and cakf is not None:
            cakf.predict(dt)
    return max(0, len(times) - 1), (time.time() - t0)


def _prepare_icp_inputs(ds: KITTIRawLoader,
                        i: int,
                        cfg: dict,
                        ekf: PoseEKF,
                        T_il: np.ndarray,
                        rolling_map: RollingMap) -> tuple:
    """Prepare source cloud, target map, and initial guess for ICP."""
    pts = ds.get_velodyne(i)
    pcd_src = preprocess_cloud(pts, cfg['icp']['voxel_size_scan'], cfg['icp']['normal_radius'], cfg['icp']['normal_max_nn'])
    target_map = rolling_map.build_target_map()
    # Ensure the merged target map has normals for point-to-plane ICP
    if len(target_map.points) > 0 and not target_map.has_normals():
        import open3d as o3d
        target_map.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=float(cfg['icp']['normal_radius']),
                max_nn=int(cfg['icp']['normal_max_nn'])
            )
        )
    T_init = ekf.get_T_wi() @ T_il
    return pcd_src, target_map, T_init


def _smooth_icp_with_cakf(ekf: PoseEKF,
                          cakf: ConstantAccelerationKF,
                          T_wl: np.ndarray,
                          T_il: np.ndarray) -> np.ndarray:
    """Smooth ICP pose using the CAKF and return the smoothed T_wl_used."""
    T_wl_used = np.array(T_wl, dtype=float, copy=True)
    # Translation smoothing
    p_icp = np.array(T_wl[:3, 3], dtype=float, copy=True)
    if not cakf.initialized:
        T_pred = ekf.get_T_wi() @ T_il
        cakf.initialize(p_icp, R_world=T_pred[:3, :3])
    else:
        cakf.update_position(p_icp)
    # Global rotation smoothing
    R_meas_wl = T_wl[:3, :3]
    cakf.update_rotation_global(R_meas_wl)
    # Build smoothed transform
    T_wl_used[:3, 3] = cakf.get_position()
    T_wl_used[:3, :3] = cakf.get_rotation_matrix()
    return T_wl_used


def _fuse_pose_or_set(ekf: PoseEKF, mode: str, T_wl_used: np.ndarray, T_il: np.ndarray, R_icp: np.ndarray) -> None:
    """Fuse ICP pose with EKF or set pose directly depending on mode."""
    if mode == 'ekf':
        ekf.update_with_icp_pose(T_wl_used, T_il, R_icp)
    else:
        T_wi = T_wl_used @ invert_se3(T_il)
        ekf.R = T_wi[:3, :3]
        ekf.p = T_wi[:3, 3]


def run_slam(config_path: str, date: str, drive: str,
             start_frame: Optional[int] = None,
             end_frame: Optional[int] = None,
             mode: Optional[str] = None,
             icp_cov_provider: Optional[IcpCovarianceProvider] = None):
    """Run the SLAM pipeline for a KITTI Raw sequence.

    Args:
        config_path: YAML config path
        date: KITTI date (e.g., 2011_09_26)
        drive: KITTI drive ID (e.g., 0001)

    Steps:
    1) Read config and dataset indices
    2) Load calibration T_il (IMU<-LiDAR)
    3) Initialize pose EKF from config or OXTS
    4) For each LiDAR frame: integrate IMU, run ICP, fuse or set pose
    5) Save predicted trajectory and evaluate
    """
    # 1) Load configuration
    cfg = load_yaml_config(config_path)

    # 2) Prepare run directory for outputs (append mode to name)
    # Get mode early to use in directory name
    mode_for_dir = mode if mode is not None else cfg['mode']['type']
    run_dir = ensure_run_dir(cfg['logging']['run_root'], date, drive, suffix=mode_for_dir)

    # 2.5) Copy config file to run directory for reproducibility
    import shutil
    config_copy_path = os.path.join(run_dir, 'config.yaml')
    shutil.copy2(config_path, config_copy_path)
    print(f"Copied config to: {config_copy_path}")

    # 3) Open dataset (timestamps, LiDAR, IMU)
    spec = SequenceSpec(
        root_dir=cfg['dataset']['root_dir'],
        date=date,
        drive=drive,
        frames=cfg['dataset'].get('frames', []),
        start_frame=start_frame if start_frame is not None else cfg['dataset'].get('start_frame'),
        end_frame=end_frame if end_frame is not None else cfg['dataset'].get('end_frame'),
    )
    ds = KITTIRawLoader(spec)

    # Quick IMU/LiDAR timing sanity
    num_imu_samples = int(len(ds.imu_timestamps))
    num_lidar_samples = int(ds.num_lidar())
    mean_imu_dt = float(np.mean(np.diff(ds.imu_timestamps))) if num_imu_samples > 1 else None
    mean_lidar_dt = float(np.mean(np.diff(ds.lidar_timestamps))) if num_lidar_samples > 1 else None
    if mean_imu_dt is not None and mean_imu_dt > 0:
        print(f"IMU samples: {num_imu_samples}, mean dt={mean_imu_dt:.6f}s (~{1.0/mean_imu_dt:.1f} Hz)")
    else:
        print(f"IMU samples: {num_imu_samples}")
    if mean_lidar_dt is not None and mean_lidar_dt > 0:
        print(f"LiDAR frames: {num_lidar_samples}, mean dt={mean_lidar_dt:.6f}s (~{1.0/mean_lidar_dt:.1f} Hz)")
    else:
        print(f"LiDAR frames: {num_lidar_samples}")
    # Require at least one LiDAR frame for processing and evaluation
    if ds.num_lidar() <= 0:
        raise ValueError("No LiDAR frames after applying frames/start_frame/end_frame; both predicted and GT must be non-empty.")

    # 4) Load extrinsic calibration IMU<-LiDAR (T_il)
    T_il = _load_T_il(cfg, date)

    # 5) Select operating mode (CLI overrides config)
    mode = mode if mode is not None else cfg['mode']['type']  # ekf | lidar_only | imu_only | cakf

    # 6) Initialize pose (R0, p0)
    R0, p0 = _init_pose(cfg, ds)

    # 7) Create pose EKF for IMU prediction and ICP fusion
    ekf = _make_pose_ekf(cfg, R0, p0)

    # 8) Create optional ICP smoother (global rotation + position)
    cakf = _make_cakf_if_enabled(cfg, mode)

    # 9) Create rolling map for ICP target points
    rolling_map = RollingMap(
        keep_last_k_scans=int(cfg['map']['keep_last_k_scans']),
        voxel_size_map=float(cfg['map']['voxel_size_map']),
        crop_radius_m=float(cfg['map']['crop_radius_m']),
        max_points=int(cfg['map']['max_points']),
    )

    # 10) Build ICP measurement covariance (constant or per-frame)
    R_icp_const = None
    if 'measurement' in cfg['icp'] and 'R_const_diag' in cfg['icp']['measurement']:
        R_icp_const = build_measurement_covariance(np.array(cfg['icp']['measurement']['R_const_diag'], dtype=float))

    traj_Twb: List[np.ndarray] = []

    prev_lidar_time = ds.lidar_timestamps[0] if ds.num_lidar() > 0 else None

    # Logging
    log_every = max(1, int(cfg['logging'].get('write_every_n', 1)))
    time_imu_total = 0.0
    time_icp_total = 0.0

    # Per-frame debug/metrics accumulation
    per_frame_rmse: List[float] = []
    per_frame_corr: List[int] = []
    per_frame_map_pts: List[int] = []
    per_frame_imu_steps: List[int] = []
    per_frame_imu_runtime: List[float] = []

    # Sanity print once
    did_print_sanity = False

    # 11) Main processing loop over LiDAR frames
    iterator = tqdm(range(ds.num_lidar()), desc=f"{mode} {date}_{drive}")
    for i in iterator:
        last_rmse = None
        last_corr = 0
        last_map_pts = 0
        imu_steps = 0

        # 11.a) Integrate IMU between previous and current LiDAR timestamps
        t_curr = ds.lidar_timestamps[i]
        frame_dt_imu = 0.0
        if prev_lidar_time is not None and (mode in ('ekf', 'imu_only') or mode == 'cakf'):
            steps, dt_imu = _imu_predict_between(ekf, cakf, ds, prev_lidar_time, t_curr, mode)
            time_imu_total += dt_imu
            imu_steps = steps
            frame_dt_imu = dt_imu

        # 11.b) Prepare and run ICP if enabled by mode
        if mode in ('ekf', 'lidar_only', 'cakf'):
            pcd_src, target_map, T_init = _prepare_icp_inputs(ds, i, cfg, ekf, T_il, rolling_map)
            last_map_pts = len(target_map.points)
            # Skip ICP only if the target map is strictly empty (first frame)
            if last_map_pts == 0:
                # Bootstrap: add first scan with initial guess, set pose for non-ekf modes
                pcd_world = pcd_src.transform(T_init)
                rolling_map.add_scan(pcd_world, center_world=ekf.p)
                # For lidar_only/cakf, pose remains ekf.get_T_wi(); no change needed
                # Continue to next frame
                per_frame_rmse.append(None)
                per_frame_corr.append(0)
                per_frame_map_pts.append(last_map_pts)
                per_frame_imu_steps.append(imu_steps)
                per_frame_imu_runtime.append(frame_dt_imu)
                traj_Twb.append(ekf.get_T_wi())
                prev_lidar_time = t_curr
                if i % log_every == 0:
                    icp_rmse_str = "nan"
                    tqdm.write(
                        f"i={i}/{ds.num_lidar()} mode={mode} imu_steps={imu_steps} "
                        f"icp_rmse={icp_rmse_str} corr=0 map_pts={last_map_pts} "
                        f"runtime_imu_total_s={time_imu_total:.2f} runtime_icp_total_s={time_icp_total:.2f}"
                    )
                continue

            # 11.c) Run point-to-plane ICP with initial guess
            t0 = time.time()
            T_wl, rmse, n_corr = icp_point_to_plane(pcd_src, target_map, T_init, cfg['icp']['max_correspondence_dist'], cfg['icp']['max_iters'], cfg['icp']['robust_delta'])
            time_icp_total += time.time() - t0
            last_rmse = rmse
            last_corr = int(n_corr)

            if not did_print_sanity:
                # 11.d) Print first residual sanity once
                from src.utils.geometry import se3_log, invert_se3
                T_pred = ekf.get_T_wi() @ T_il
                T_res = invert_se3(T_pred) @ T_wl
                xi = se3_log(T_res)
                print(f"Sanity residual (w.r.t. first ICP): transl={np.linalg.norm(xi[:3]):.3f} m, rot={np.linalg.norm(xi[3:]):.3f} rad")
                did_print_sanity = True

            # Choose transform to use (copy if we will modify)
            T_wl_used = T_wl

            # 11.e) Optionally smooth ICP pose with CAKF (position + global rotation)
            if mode == 'cakf' and cakf is not None:
                T_wl_used = _smooth_icp_with_cakf(ekf, cakf, T_wl, T_il)

            # 11.f) Fuse ICP into EKF or set pose directly
            R_icp_use = R_icp_const
            if icp_cov_provider is not None:
                # Map iterator index to dataset frame index if subset used
                dataset_idx = int(ds.frame_indices[i]) if getattr(ds, 'frame_indices', None) is not None else i
                R_icp_use = icp_cov_provider.get_covariance(date, drive, dataset_idx)
            if R_icp_use is None:
                raise ValueError("ICP covariance is not defined. Provide R_const_diag or a provider.")
            _fuse_pose_or_set(ekf, mode, T_wl_used, T_il, R_icp_use)

            # 11.g) Update rolling map with current scan
            pcd_world = pcd_src.transform(T_wl_used)
            rolling_map.add_scan(pcd_world, center_world=ekf.p)

            # Record per-frame metrics
            per_frame_rmse.append(float(last_rmse) if last_rmse is not None else None)
            per_frame_corr.append(int(last_corr))
            per_frame_map_pts.append(int(last_map_pts))
            per_frame_imu_steps.append(int(imu_steps))
            per_frame_imu_runtime.append(float(frame_dt_imu))

        # 11.i) Append current pose to trajectory and advance time
        traj_Twb.append(ekf.get_T_wi())
        prev_lidar_time = t_curr

        # 11.j) Periodic progress logging
        if i % log_every == 0:
            icp_rmse_str = f"{last_rmse:.3f}" if last_rmse is not None else "nan"
            tqdm.write(
                f"i={i}/{ds.num_lidar()} mode={mode} imu_steps={imu_steps} "
                f"icp_rmse={icp_rmse_str} corr={last_corr} map_pts={last_map_pts} "
                f"runtime_imu_total_s={time_imu_total:.2f} runtime_icp_total_s={time_icp_total:.2f}"
            )

    # 12) Save predicted trajectory to KITTI format (strict)
    out_pred = None
    if len(traj_Twb) <= 0:
        raise RuntimeError("No predicted trajectory frames generated; both predicted and GT must be non-empty.")
    traj_arr = np.stack(traj_Twb, axis=0)
    out_pred = os.path.join(run_dir, 'trajectory_pred.kitti')
    save_kitti_trajectory(out_pred, traj_arr)
    # Verify file is non-empty and well-formed (at least 1 non-empty line)
    def _count_kitti_pose_lines(path: str) -> int:
        n = 0
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    n += 1
        return n
    num_pred_lines = _count_kitti_pose_lines(out_pred)
    if num_pred_lines <= 0:
        raise RuntimeError(f"Predicted KITTI pose file is empty: {out_pred}")
    print(f"Saved predicted trajectory: {out_pred} (poses={num_pred_lines})")

    # 13) Export ground-truth aligned to LiDAR timestamps (strict)
    out_gt = None
    num_gt_poses = 0
    Twb_gt: List[np.ndarray] = []
    for i in range(ds.num_lidar()):
        t = ds.lidar_timestamps[i]
        idx = int(np.searchsorted(ds.imu_timestamps, t, side='left'))
        idx = np.clip(idx, 0, len(ds.imu_timestamps) - 1)
        T_w_imu = ds.dataset.oxts[idx].T_w_imu
        Twb_gt.append(T_w_imu)
    num_gt_poses = len(Twb_gt)
    if num_gt_poses <= 0:
        raise RuntimeError("No ground-truth poses; both predicted and GT must be non-empty.")
    gt_arr = np.stack(Twb_gt, axis=0)
    out_gt = os.path.join(run_dir, 'trajectory_gt.kitti')
    save_kitti_trajectory(out_gt, gt_arr)
    num_gt_lines = _count_kitti_pose_lines(out_gt)
    if num_gt_lines <= 0:
        raise RuntimeError(f"Ground-truth KITTI pose file is empty: {out_gt}")
    print(f"Saved ground-truth trajectory: {out_gt} (poses={num_gt_lines})")

    # 14) Optional evaluation with evo (APE & RPE) and save aligned/unaligned metrics
    ev = cfg.get('evaluation', {})
    if ev.get('enable', True) and out_gt is not None and out_pred is not None and num_pred_lines >= 2 and num_gt_lines >= 2:
        args_ape = ['evo_ape', 'kitti', out_gt, out_pred]
        if ev.get('ape', {}).get('align', True):
            args_ape.append('-a')
        if ev.get('ape', {}).get('stats', True):
            args_ape.append('-s')
        if ev.get('show_plots', False):
            args_ape.append('--plot')
        if ev.get('save_plots', True):
            args_ape += ['--save_plot', os.path.join(run_dir, 'ape.pdf')]

        args_rpe = ['evo_rpe', 'kitti', out_gt, out_pred]
        if ev.get('rpe', {}).get('align', True):
            args_rpe.append('-a')
        if ev.get('rpe', {}).get('stats', True):
            args_rpe.append('-s')
        args_rpe += ['--delta', str(ev.get('rpe', {}).get('delta', 1))]
        args_rpe += ['--delta_unit', ev.get('rpe', {}).get('delta_unit', 'm')]
        if ev.get('show_plots', False):
            args_rpe.append('--plot')
        if ev.get('save_plots', True):
            args_rpe += ['--save_plot', os.path.join(run_dir, 'rpe.pdf')]

        try:
            subprocess.run(args_ape, check=False)
            subprocess.run(args_rpe, check=False)
        except FileNotFoundError:
            print("evo not found; skip automatic evaluation. Install with: pip install evo")

        # Save evo metrics via API for both aligned and unaligned cases
        save_evo_metrics_json(run_dir, out_gt, out_pred, filename='evo_metrics.json')
        print(f"Saved evo metrics (aligned and unaligned): {os.path.join(run_dir, 'evo_metrics.json')}")

        print(f"Evaluation done. See run_dir: {run_dir}")
    elif ev.get('enable', True):
        raise RuntimeError("Evaluation enabled but not enough poses: need at least 2 in both GT and prediction.")

    # 15) Save debug metrics as JSON
    def _nanmean(values):
        vs = [v for v in values if v is not None]
        return float(np.mean(vs)) if len(vs) > 0 else None

    metrics = {
        "mode": mode,
        "date": date,
        "drive": drive,
        "num_frames": int(len(traj_Twb)),
        "num_imu_samples": num_imu_samples,
        "num_lidar_samples": num_lidar_samples,
        "mean_imu_dt_s": mean_imu_dt,
        "mean_lidar_dt_s": mean_lidar_dt,
        "runtime_imu_total_s": float(time_imu_total),
        "runtime_icp_total_s": float(time_icp_total),
        "mean_icp_rmse": _nanmean(per_frame_rmse),
        "mean_icp_corr": float(np.mean(per_frame_corr)) if per_frame_corr else None,
        "mean_map_pts": float(np.mean(per_frame_map_pts)) if per_frame_map_pts else None,
        "mean_imu_steps": float(np.mean(per_frame_imu_steps)) if per_frame_imu_steps else None,
        "imu_steps_hist": {str(k): int((np.array(per_frame_imu_steps)==k).sum()) for k in set(per_frame_imu_steps)} if per_frame_imu_steps else {},
        "per_frame": {
            "rmse": per_frame_rmse,
            "corr": per_frame_corr,
            "map_pts": per_frame_map_pts,
            "imu_steps": per_frame_imu_steps,
            "imu_runtime_s": per_frame_imu_runtime,
        },
    }
    metrics_path = os.path.join(run_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics: {metrics_path}")

    # 16) Save a small debug sample of frames
    debug_indices = []
    n = len(traj_Twb)
    if n > 0:
        debug_indices = sorted(set([0, max(0, n//2), n-1]))
    debug_frames = []
    for idx in debug_indices:
        T = traj_Twb[idx]
        debug_frames.append({
            "i": int(idx),
            "pose_t": [float(x) for x in T[:3, 3].tolist()],
            "rmse": per_frame_rmse[idx] if idx < len(per_frame_rmse) else None,
            "corr": per_frame_corr[idx] if idx < len(per_frame_corr) else None,
            "map_pts": per_frame_map_pts[idx] if idx < len(per_frame_map_pts) else None,
            "imu_steps": per_frame_imu_steps[idx] if idx < len(per_frame_imu_steps) else None,
        })
    debug_path = os.path.join(run_dir, 'frames_debug.json')
    with open(debug_path, 'w') as f:
        json.dump({"frames": debug_frames}, f, indent=2)
    print(f"Saved debug frames: {debug_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--date', type=str, required=True)
    parser.add_argument('--drive', type=str, required=True)
    parser.add_argument('--start_frame', type=int, default=None)
    parser.add_argument('--end_frame', type=int, default=None)
    parser.add_argument('--mode', type=str, default=None, choices=['ekf', 'lidar_only', 'imu_only', 'cakf'],
                        help='Override mode from config file')
    args = parser.parse_args()
    run_slam(args.config, args.date, args.drive, start_frame=args.start_frame, end_frame=args.end_frame, mode=args.mode)

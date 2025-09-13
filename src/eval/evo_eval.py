import os
import json
from typing import Tuple, Dict, Any
import numpy as np

from evo.core import metrics as evo_metrics
from evo.core import sync as evo_sync
from evo.core import trajectory as evo_traj


def load_kitti_pose_with_index_timestamps(path: str) -> evo_traj.PoseTrajectory3D:
    # Reuse evo.tools.file_interface for robustness
    from evo.tools import file_interface as evo_io
    pose_path = evo_io.read_kitti_poses_file(path)
    poses = pose_path.poses_se3
    n = len(poses)
    ts = np.arange(n, dtype=float)
    return evo_traj.PoseTrajectory3D(poses_se3=poses, timestamps=ts)


def _align_est_to_ref(ref: evo_traj.PoseTrajectory3D,
                      est: evo_traj.PoseTrajectory3D,
                      enable_align: bool) -> evo_traj.PoseTrajectory3D:
    ref_sync, est_sync = ref, est
    try:
        if ref.timestamps is not None and est.timestamps is not None and len(ref.timestamps) == len(est.timestamps):
            ref_sync, est_sync = ref, est
        else:
            ref_sync, est_sync = evo_sync.associate_trajectories(ref, est)
    except Exception:
        ref_sync, est_sync = ref, est

    if not enable_align:
        return evo_traj.PoseTrajectory3D(poses_se3=est_sync.poses_se3, timestamps=est_sync.timestamps)

    aligned_res = est_sync.align(ref_sync, correct_scale=False)

    if isinstance(aligned_res, tuple):
        cand = aligned_res[0]
    else:
        cand = aligned_res

    if isinstance(cand, evo_traj.PoseTrajectory3D):
        return cand
    if isinstance(cand, np.ndarray):
        if cand.ndim == 3 and cand.shape[1:] == (4, 4):
            mats = [cand[i] for i in range(cand.shape[0])]
            return evo_traj.PoseTrajectory3D(poses_se3=mats, timestamps=est_sync.timestamps)
        if cand.shape == (4, 4):
            T = cand
            mats = [T @ P for P in est_sync.poses_se3]
            return evo_traj.PoseTrajectory3D(poses_se3=mats, timestamps=est_sync.timestamps)

    return evo_traj.PoseTrajectory3D(poses_se3=est_sync.poses_se3, timestamps=est_sync.timestamps)


def compute_evo_metrics(ref: evo_traj.PoseTrajectory3D,
                        est: evo_traj.PoseTrajectory3D) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    for mode, do_align in {"aligned": True, "unaligned": False}.items():
        est_use = _align_est_to_ref(ref, est, do_align)

        ape_metric = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
        ape_metric.process_data((ref, est_use))
        ape_stats = ape_metric.get_all_statistics()

        rpe_metric = evo_metrics.RPE(evo_metrics.PoseRelation.translation_part, delta=1, delta_unit=evo_metrics.Unit.frames)
        rpe_metric.process_data((ref, est_use))
        rpe_stats = rpe_metric.get_all_statistics()

        # Convert numpy types to native floats
        results[mode] = {
            "ape_rmse": float(ape_stats["rmse"]) if "rmse" in ape_stats else None,
            "ape_mean": float(ape_stats["mean"]) if "mean" in ape_stats else None,
            "ape_median": float(ape_stats["median"]) if "median" in ape_stats else None,
            "ape_std": float(ape_stats["std"]) if "std" in ape_stats else None,
            "rpe_rmse": float(rpe_stats["rmse"]) if "rmse" in rpe_stats else None,
            "rpe_mean": float(rpe_stats["mean"]) if "mean" in rpe_stats else None,
            "rpe_median": float(rpe_stats["median"]) if "median" in rpe_stats else None,
            "rpe_std": float(rpe_stats["std"]) if "std" in rpe_stats else None,
        }
    return results


def save_evo_metrics_json(out_dir: str,
                          ref_path: str,
                          est_path: str,
                          filename: str = "evo_metrics.json") -> Dict[str, Dict[str, float]]:
    os.makedirs(out_dir, exist_ok=True)
    ref = load_kitti_pose_with_index_timestamps(ref_path)
    est = load_kitti_pose_with_index_timestamps(est_path)
    metrics = compute_evo_metrics(ref, est)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics



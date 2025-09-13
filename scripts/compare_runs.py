"""
Compare two SLAM runs (each with predicted and GT trajectories) using evo (Python API).

Both runs are expected to contain:
- trajectory_pred.kitti (estimate)
- trajectory_gt.kitti   (ground truth)

We plot both estimates together with the GT in the same figure (XY), after optionally aligning
each estimate to the GT via SE(3) (Umeyama). We also compute APE and RPE metrics for
each run relative to the GT used for plotting/alignment.

Usage:
  python -m scripts.compare_runs --run_a /path/to/runA --run_b /path/to/runB --save_dir out
  python -m scripts.compare_runs --run_a /path/to/runA --run_b /path/to/runB --save_dir out --no_align
"""
import os
import argparse
import numpy as np
import json

from evo.core import metrics as evo_metrics
from evo.core import sync as evo_sync
from evo.core import trajectory as evo_traj
from evo.core import result as evo_result
from evo.tools import file_interface as evo_io
from evo.tools import plot as evo_plot
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval.evo_eval import compute_evo_metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_kitti_pose(path: str) -> evo_traj.PoseTrajectory3D:
    """Load KITTI poses and add synthetic frame-index timestamps for evo.

    KITTI pose files don't contain timestamps; evo requires them for PoseTrajectory3D.
    We use 0..N-1 as timestamps to enable association and metrics.
    """
    pose_path = evo_io.read_kitti_poses_file(path)  # PosePath3D
    poses = pose_path.poses_se3
    n = len(poses)
    ts = np.arange(n, dtype=float)
    return evo_traj.PoseTrajectory3D(poses_se3=poses, timestamps=ts)


def align_and_metrics(ref_traj: evo_traj.PoseTrajectory3D,
                      est_traj: evo_traj.PoseTrajectory3D,
                      align: bool = True):
    """Align est to ref and compute APE/RPE metrics.

    If timestamps are absent or lengths match, skip association and use index alignment.
    If align=False, skip alignment and use raw trajectories for metrics.
    """
    ref_sync, est_sync = ref_traj, est_traj
    try:
        if ref_traj.timestamps is not None and est_traj.timestamps is not None and len(ref_traj.timestamps) == len(est_traj.timestamps):
            # Index-aligned; no need to associate
            ref_sync, est_sync = ref_traj, est_traj
        else:
            ref_sync, est_sync = evo_sync.associate_trajectories(ref_traj, est_traj)
    except Exception:
        ref_sync, est_sync = ref_traj, est_traj
    
    if align:
        aligned_res = est_sync.align(ref_sync, correct_scale=False)
    else:
        # Skip alignment, use original trajectories
        aligned_res = est_sync
    # Handle evo API variants: may return
    # - (traj, transform)
    # - a 4x4 ndarray transform
    # - a PoseTrajectory3D
    # - an ndarray of shape (N,4,4)
    est_aligned = None
    if isinstance(aligned_res, tuple):
        cand = aligned_res[0]
        if isinstance(cand, evo_traj.PoseTrajectory3D):
            est_aligned = cand
        elif isinstance(cand, np.ndarray):
            if cand.ndim == 3 and cand.shape[1:] == (4, 4):
                mats = [cand[i] for i in range(cand.shape[0])]
                est_aligned = evo_traj.PoseTrajectory3D(poses_se3=mats, timestamps=est_sync.timestamps)
            elif cand.shape == (4, 4):
                T = cand
                mats = [T @ P for P in est_sync.poses_se3]
                est_aligned = evo_traj.PoseTrajectory3D(poses_se3=mats, timestamps=est_sync.timestamps)
    elif isinstance(aligned_res, np.ndarray):
        if aligned_res.ndim == 3 and aligned_res.shape[1:] == (4, 4):
            mats = [aligned_res[i] for i in range(aligned_res.shape[0])]
            est_aligned = evo_traj.PoseTrajectory3D(poses_se3=mats, timestamps=est_sync.timestamps)
        elif aligned_res.shape == (4, 4):
            T = aligned_res
            mats = [T @ P for P in est_sync.poses_se3]
            est_aligned = evo_traj.PoseTrajectory3D(poses_se3=mats, timestamps=est_sync.timestamps)
    else:
        est_aligned = aligned_res
    # Fallback: if still not a PoseTrajectory3D, wrap from est_sync
    if not isinstance(est_aligned, evo_traj.PoseTrajectory3D):
        est_aligned = evo_traj.PoseTrajectory3D(poses_se3=est_sync.poses_se3, timestamps=est_sync.timestamps)

    # Use translation-only metrics to avoid PoseRelation.se3 enum differences across evo versions
    ape_metric = evo_metrics.APE(evo_metrics.PoseRelation.translation_part)
    ape_metric.process_data((ref_sync, est_aligned))
    ape_stats = ape_metric.get_all_statistics()

    rpe_metric = evo_metrics.RPE(evo_metrics.PoseRelation.translation_part, delta=1, delta_unit=evo_metrics.Unit.frames)
    rpe_metric.process_data((ref_sync, est_aligned))
    rpe_stats = rpe_metric.get_all_statistics()

    return est_aligned, ape_stats, rpe_stats


def save_plots(ref: evo_traj.PoseTrajectory3D,
               estA: evo_traj.PoseTrajectory3D,
               estB: evo_traj.PoseTrajectory3D,
               labelA: str,
               labelB: str,
               save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    # Helper to get positions safely
    def _xy(t: evo_traj.PoseTrajectory3D):
        P = t.positions_xyz
        return P[:, 0], P[:, 1]

    def _xz(t: evo_traj.PoseTrajectory3D):
        # Use Y (forward) vs Z (vertical) to show elevation profile correctly
        P = t.positions_xyz
        return P[:, 1], P[:, 2]

    # XY plot (single draw per trajectory)
    fig, ax = plt.subplots()
    gx, gy = _xy(ref)
    ax.plot(gx, gy, '-', color='k', label='GT', linewidth=2)
    ax.plot(*_xy(estA), '--', color='r', label=labelA, linewidth=1.5)
    ax.plot(*_xy(estB), '--', color='b', label=labelB, linewidth=1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    fig.savefig(os.path.join(save_dir, 'traj_xy.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # XZ plot
    fig, ax = plt.subplots()
    gx, gz = _xz(ref)
    ax.plot(gx, gz, '-', color='k', label='GT', linewidth=2)
    ax.plot(*_xz(estA), '--', color='r', label=labelA, linewidth=1.5)
    ax.plot(*_xz(estB), '--', color='b', label=labelB, linewidth=1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Z [m]')
    ax.legend()
    fig.savefig(os.path.join(save_dir, 'traj_xz.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _safe_label(run_dir: str) -> str:
    # Try mode from metrics.json, else infer from folder name suffix
    metrics_path = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                m = json.load(f)
                if 'mode' in m:
                    return str(m['mode'])
        except Exception:
            pass
    return os.path.basename(run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_a', type=str, required=True, help='Run A dir (has trajectory_pred.kitti and trajectory_gt.kitti)')
    parser.add_argument('--run_b', type=str, required=True, help='Run B dir (has trajectory_pred.kitti and trajectory_gt.kitti)')
    parser.add_argument('--save_dir', type=str, required=False, default=None)
    parser.add_argument('--no_align', action='store_true', help='Skip trajectory alignment (use raw trajectories)')
    args = parser.parse_args()

    # Load A
    gt_a_path = os.path.join(args.run_a, 'trajectory_gt.kitti')
    est_a_path = os.path.join(args.run_a, 'trajectory_pred.kitti')
    if not os.path.exists(gt_a_path) or not os.path.exists(est_a_path):
        raise FileNotFoundError(f"Missing GT or pred in {args.run_a}")
    gt_a = load_kitti_pose(gt_a_path)
    est_a = load_kitti_pose(est_a_path)

    # Load B
    gt_b_path = os.path.join(args.run_b, 'trajectory_gt.kitti')
    est_b_path = os.path.join(args.run_b, 'trajectory_pred.kitti')
    if not os.path.exists(gt_b_path) or not os.path.exists(est_b_path):
        raise FileNotFoundError(f"Missing GT or pred in {args.run_b}")
    gt_b = load_kitti_pose(gt_b_path)
    est_b = load_kitti_pose(est_b_path)

    # Align BOTH estimates to GT A before plotting (unless --no_align)
    align_enabled = not args.no_align
    est_a_aligned, ape_stats_a, rpe_stats_a = align_and_metrics(gt_a, est_a, align=align_enabled)
    est_b_aligned, ape_stats_b, rpe_stats_b = align_and_metrics(gt_a, est_b, align=align_enabled)

    # Compute evo APE/RPE metrics (aligned and unaligned) using shared helper
    metrics_a = compute_evo_metrics(gt_a, est_a)
    metrics_b = compute_evo_metrics(gt_a, est_b)

    # Save metrics JSON
    out_dir = args.save_dir or os.path.join(args.run_b, 'compare')
    os.makedirs(out_dir, exist_ok=True)
    out_metrics = {
        'run_a': {
            'label': _safe_label(args.run_a),
            'ape': ape_stats_a,
            'rpe_trans': rpe_stats_a,
            'evo': metrics_a,  # aligned and unaligned summary numbers
        },
        'run_b': {
            'label': _safe_label(args.run_b),
            'ape': ape_stats_b,
            'rpe_trans': rpe_stats_b,
            'evo': metrics_b,
        }
    }
    with open(os.path.join(out_dir, 'compare_metrics.json'), 'w') as f:
        json.dump(out_metrics, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))

    # Save plots with both estimates vs the GT (GT from run A)
    save_plots(gt_a, est_a_aligned, est_b_aligned, _safe_label(args.run_a), _safe_label(args.run_b), out_dir)
    print(f"Saved comparison metrics and plots to: {out_dir}")


if __name__ == '__main__':
    main()



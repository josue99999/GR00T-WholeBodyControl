#!/usr/bin/env python3
"""
Converts robot_motion_g1.pkl (video2robot format) to the CSV format
expected by GR00T-WholeBodyControl's gear_sonic_deploy.

Input PKL keys:
  fps, root_pos (N,3), root_rot (N,4) [x,y,z,w], dof_pos (N,29),
  local_body_pos (N,38,3) [root-relative], link_body_list (38,)

Output: reference/<motion_name>/<motion_name>/  with CSV files at 50 Hz
"""

import argparse
import os
import sys
import pickle
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

# Body indices that match the reference motions' metadata (_body_indexes)
# These index into the G1's link_body_list (38 links)
BODY_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]

TARGET_FPS = 50.0  # Hz expected by gear_sonic_deploy

# video2robot dof_pos is in MuJoCo URDF order; deploy CSVs expect IsaacLab order.
# isaaclab_to_mujoco from visualize_motion.py (apply to IsaacLab CSV → MuJoCo for display)
_IL_TO_MJ = np.array([0,3,6,9,13,17,1,4,7,10,14,18,2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28])
MJ_TO_IL = np.argsort(_IL_TO_MJ)  # inverse: reorders MuJoCo columns → IsaacLab order


def resample_positions(data, src_fps, dst_fps):
    """Linearly interpolate (N, ...) position array from src_fps to dst_fps."""
    n_src = data.shape[0]
    t_src = np.arange(n_src) / src_fps
    t_dst = np.arange(0, t_src[-1], 1.0 / dst_fps)
    shape_rest = data.shape[1:]
    flat = data.reshape(n_src, -1)
    out = interp1d(t_src, flat, axis=0, kind="linear", fill_value="extrapolate")(t_dst)
    return out.reshape(-1, *shape_rest), t_dst


def resample_quats_xyzw(quats_xyzw, src_fps, dst_fps):
    """Slerp a (N,4) quaternion array (x,y,z,w) from src_fps to dst_fps."""
    n = quats_xyzw.shape[0]
    t_src = np.arange(n) / src_fps
    t_dst = np.arange(0, t_src[-1], 1.0 / dst_fps)
    rotations = Rotation.from_quat(quats_xyzw)
    slerp = Slerp(t_src, rotations)
    return slerp(t_dst)


def finite_diff(data, dt):
    """Central-differences velocity, forward/backward at edges."""
    vel = np.gradient(data, dt, axis=0)
    return vel


def save_csv(path, array, headers):
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for row in array:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def convert(pkl_path, output_base_dir, motion_name=None):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    fps = float(d["fps"])
    root_pos = np.array(d["root_pos"], dtype=np.float64)       # (N, 3)
    root_rot_xyzw = np.array(d["root_rot"], dtype=np.float64)  # (N, 4) x,y,z,w
    dof_pos = np.array(d["dof_pos"], dtype=np.float64)         # (N, 29)
    local_body_pos = np.array(d["local_body_pos"], dtype=np.float64)  # (N, 38, 3)
    link_body_list = list(d["link_body_list"])

    n_orig = root_pos.shape[0]
    print(f"Source: {n_orig} frames @ {fps:.2f} fps  ({n_orig/fps:.1f} s)")

    # ── 1. Resample to 50 Hz ─────────────────────────────────────────────
    root_pos_50, t_dst = resample_positions(root_pos, fps, TARGET_FPS)
    root_rot_50 = resample_quats_xyzw(root_rot_xyzw, fps, TARGET_FPS)
    dof_pos_50, _ = resample_positions(dof_pos, fps, TARGET_FPS)
    local_body_pos_50, _ = resample_positions(local_body_pos.reshape(n_orig, -1), fps, TARGET_FPS)
    local_body_pos_50 = local_body_pos_50.reshape(-1, local_body_pos.shape[1], 3)

    N = root_pos_50.shape[0]
    dt = 1.0 / TARGET_FPS
    print(f"Resampled: {N} frames @ {TARGET_FPS:.0f} Hz  ({N/TARGET_FPS:.1f} s)")

    # ── 2. Reorder joints MuJoCo → IsaacLab, then compute velocities ────
    dof_pos_50 = dof_pos_50[:, MJ_TO_IL]
    joint_vel_50 = finite_diff(dof_pos_50, dt)  # (N, 29)

    # ── 3. Build world-frame body positions (14 key bodies) ──────────────
    # world_pos[i] = root_pos + R(root_rot) @ local_body_pos[i]
    # root body (index 0) local pos is always [0,0,0] → world_pos = root_pos
    R_all = root_rot_50.as_matrix()  # (N, 3, 3)
    body_pos_w = np.zeros((N, len(BODY_INDEXES), 3))
    for j, bidx in enumerate(BODY_INDEXES):
        local = local_body_pos_50[:, bidx, :]  # (N, 3)
        # matrix-vector multiply per frame
        body_pos_w[:, j, :] = root_pos_50 + np.einsum("nij,nj->ni", R_all, local)

    # ── 4. World-frame body quaternions (use root rotation for all bodies) ─
    # We only have root orientation; approximate all bodies with root rot.
    root_wxyz = root_rot_50.as_quat()  # scipy returns (x,y,z,w)
    root_wxyz = np.concatenate([root_wxyz[:, 3:4], root_wxyz[:, :3]], axis=-1)  # → (w,x,y,z)
    body_quat_w = np.broadcast_to(
        root_wxyz[:, None, :], (N, len(BODY_INDEXES), 4)
    ).copy()  # (N, 14, 4) w,x,y,z

    # ── 5. Velocities for body positions / orientations ───────────────────
    body_lin_vel_w = finite_diff(body_pos_w.reshape(N, -1), dt).reshape(N, len(BODY_INDEXES), 3)

    # Angular velocity from successive rotations: ω ≈ (R[t+1] R[t]^-1 → axis*angle) / dt
    root_rots = root_rot_50
    ang_vel_root = np.zeros((N, 3))
    for t in range(N - 1):
        delta_r = root_rots[t + 1] * root_rots[t].inv()
        rv = delta_r.as_rotvec()   # axis * angle (rad)
        ang_vel_root[t] = rv / dt
    ang_vel_root[-1] = ang_vel_root[-2]
    body_ang_vel_w = np.broadcast_to(
        ang_vel_root[:, None, :], (N, len(BODY_INDEXES), 3)
    ).copy()

    # ── 6. Write CSVs ─────────────────────────────────────────────────────
    if motion_name is None:
        motion_name = os.path.splitext(os.path.basename(pkl_path))[0]

    out_dir = os.path.join(output_base_dir, motion_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Writing to: {out_dir}/")

    # joint_pos.csv
    save_csv(
        os.path.join(out_dir, "joint_pos.csv"),
        dof_pos_50,
        [f"joint_{i}" for i in range(29)],
    )

    # joint_vel.csv
    save_csv(
        os.path.join(out_dir, "joint_vel.csv"),
        joint_vel_50,
        [f"joint_vel_{i}" for i in range(29)],
    )

    # body_pos.csv
    bp_flat = body_pos_w.reshape(N, -1)
    bp_headers = [f"body_{i//3}_{'xyz'[i%3]}" for i in range(bp_flat.shape[1])]
    save_csv(os.path.join(out_dir, "body_pos.csv"), bp_flat, bp_headers)

    # body_quat.csv  (w, x, y, z)
    bq_flat = body_quat_w.reshape(N, -1)
    bq_headers = [f"body_{i//4}_{'wxyz'[i%4]}" for i in range(bq_flat.shape[1])]
    save_csv(os.path.join(out_dir, "body_quat.csv"), bq_flat, bq_headers)

    # body_lin_vel.csv
    blv_flat = body_lin_vel_w.reshape(N, -1)
    blv_headers = [f"body_{i//3}_vel_{'xyz'[i%3]}" for i in range(blv_flat.shape[1])]
    save_csv(os.path.join(out_dir, "body_lin_vel.csv"), blv_flat, blv_headers)

    # body_ang_vel.csv
    bav_flat = body_ang_vel_w.reshape(N, -1)
    bav_headers = [f"body_{i//3}_angvel_{'xyz'[i%3]}" for i in range(bav_flat.shape[1])]
    save_csv(os.path.join(out_dir, "body_ang_vel.csv"), bav_flat, bav_headers)

    # metadata.txt
    body_idx_str = "[" + " ".join(str(b) for b in BODY_INDEXES) + "]"
    with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
        f.write(f"Metadata for: {motion_name}\n")
        f.write("=" * 30 + "\n\n")
        f.write("Body part indexes:\n")
        f.write(body_idx_str + "\n\n")
        f.write(f"Total timesteps: {N}\n\n")
        f.write("Data arrays summary:\n")
        f.write(f"  joint_pos: ({N}, 29) (float64)\n")
        f.write(f"  joint_vel: ({N}, 29) (float64)\n")
        f.write(f"  body_pos_w: ({N}, {len(BODY_INDEXES)}, 3) (float64)\n")
        f.write(f"  body_quat_w: ({N}, {len(BODY_INDEXES)}, 4) (float64)\n")
        f.write(f"  body_lin_vel_w: ({N}, {len(BODY_INDEXES)}, 3) (float64)\n")
        f.write(f"  body_ang_vel_w: ({N}, {len(BODY_INDEXES)}, 3) (float64)\n")
        f.write(f"  _body_indexes: ({len(BODY_INDEXES)},) (int64)\n")
        f.write(f"  time_step_total: () (int64)\n")

    # info.txt
    with open(os.path.join(out_dir, "info.txt"), "w") as f:
        f.write(f"Motion: {motion_name}\n")
        f.write(f"Source: {pkl_path}\n")
        f.write(f"Original FPS: {fps}\n")
        f.write(f"Resampled FPS: {TARGET_FPS}\n")
        f.write(f"Frames: {n_orig} -> {N}\n")
        f.write(f"Duration: {N/TARGET_FPS:.2f} s\n")
        f.write(f"Body indexes: {BODY_INDEXES}\n")
        f.write("Body names:\n")
        for idx in BODY_INDEXES:
            f.write(f"  [{idx}] {link_body_list[idx]}\n")

    print(f"✓ Done: {N} frames, 7 files written")
    print(f"\nBody parts selected:")
    for idx in BODY_INDEXES:
        print(f"  [{idx:2d}] {link_body_list[idx]}")


def main():
    parser = argparse.ArgumentParser(description="Convert video2robot G1 PKL to GR00T reference CSV")
    parser.add_argument("pkl", help="Path to robot_motion_g1.pkl")
    parser.add_argument("output_dir", help="Output base directory (e.g. gear_sonic_deploy/reference/danza_caporal)")
    parser.add_argument("--name", help="Motion name (default: pkl filename without extension)")
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"Error: {args.pkl} not found")
        sys.exit(1)

    convert(args.pkl, args.output_dir, motion_name=args.name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


JOINT_LABELS = {
    'waist_roll_joint': 'Waist roll',
    'waist_pitch_joint': 'Waist pitch',
    'left_shoulder_pitch_joint': 'Left shoulder pitch',
    'left_shoulder_roll_joint': 'Left shoulder roll',
    'left_elbow_joint': 'Left elbow',
    'right_shoulder_pitch_joint': 'Right shoulder pitch',
    'right_shoulder_roll_joint': 'Right shoulder roll',
    'right_elbow_joint': 'Right elbow',
}


def load_csv(path):
    df = pd.read_csv(path)
    df = df.sort_values('t_sec').reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--safe-csv', default='/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/result1/human_robot_metrics_with_cbf.csv')
    ap.add_argument('--unsafe-csv', default='/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/result1/human_robot_metrics_no_cbf.csv')
    ap.add_argument('--outdir', default='/home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/ecc_sim/figures1')
    ap.add_argument('--joint-a', default='left_shoulder_roll_joint')
    ap.add_argument('--joint-b', default='left_elbow_joint')
    ap.add_argument('--rep1', default='left_arm__right_forearm_hand')
    ap.add_argument('--rep2', default='left_upper_arm__right_forearm_hand')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df_safe = load_csv(args.safe_csv)
    df_unsafe = load_csv(args.unsafe_csv)

    # figure 1: representative pair 1
    for rep_tag, out_name in [
        (args.rep1, 'fig_human_robot_rep1.png'),
        (args.rep2, 'fig_human_robot_rep2.png'),
    ]:
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, constrained_layout=True)

        ax = axes[0]
        t = df_safe['t_sec']
        ax.plot(t, df_safe[f'unsafe_{args.joint_a}'], label=f'Unsafe command ({JOINT_LABELS.get(args.joint_a, args.joint_a)})')
        ax.plot(t, df_safe[f'safe_{args.joint_a}'], label=f'Safe command ({JOINT_LABELS.get(args.joint_a, args.joint_a)})')
        ax.plot(t, df_safe[f'unsafe_{args.joint_b}'], label=f'Unsafe command ({JOINT_LABELS.get(args.joint_b, args.joint_b)})')
        ax.plot(t, df_safe[f'safe_{args.joint_b}'], label=f'Safe command ({JOINT_LABELS.get(args.joint_b, args.joint_b)})')
        ax.set_ylabel('Joint angle (rad)')
        ax.legend(fontsize=8, ncol=2)
        ax.set_title(f'Commands and signed distance: {rep_tag.replace("__", " vs ")}')
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(df_unsafe['t_sec'], df_unsafe[f'{rep_tag}_signed_distance'], label='Without CBF')
        ax.plot(df_safe['t_sec'], df_safe[f'{rep_tag}_signed_distance'], label='With CBF')
        if 'human_global_min' in df_safe.columns and 'human_global_min' in df_unsafe.columns:
            ax.plot(df_unsafe['t_sec'], df_unsafe['human_global_min'], linestyle='--', label='Without CBF global min')
            ax.plot(df_safe['t_sec'], df_safe['human_global_min'], linestyle='--', label='With CBF global min')
        ax.axhline(0.0, linestyle=':', linewidth=1.0, label='Safety boundary')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signed distance (m)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(args.outdir, out_name), dpi=220)
        plt.close(fig)

    # optional correction figure
    corr = pd.DataFrame({'t_sec': df_safe['t_sec'].copy()})
    joint_cols = [c for c in df_safe.columns if c.startswith('unsafe_') and c != 'unsafe_t_sec']
    safe_cols = [c.replace('unsafe_', 'safe_') for c in joint_cols]
    corr['correction_norm'] = (
        (df_safe[joint_cols].values - df_safe[safe_cols].values) ** 2
    ).sum(axis=1) ** 0.5

    fig = plt.figure(figsize=(8, 2.8), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.plot(corr['t_sec'], corr['correction_norm'])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$||q_{safe}-q_{unsafe}||_2$ (rad)')
    ax.set_title('CBF correction magnitude')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.outdir, 'fig_cbf_correction_norm.png'), dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    main()

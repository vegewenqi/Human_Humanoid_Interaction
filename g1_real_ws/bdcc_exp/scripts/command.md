## 启动pipeline
ros2 launch mujoco_g1 bdcc_unified_pipeline.launch.py run_sim:=false run_real:=true use_cbf:=true

## 录制
chmod +x /ws/bdcc_exp/scripts/record/record_skeleton_segment.py

## 测试录制 S1_self_collision
python3 /ws/bdcc_exp/scripts/record/record_skeleton_segment.py \
  --scenario S1_self_collision \
  --outdir /ws/bdcc_exp/segments/S1_self_collision \
  --duration 50 \
  --start-on-enter \
  --record-diagnostics \
  --notes "Real robot CBF-enabled recording for self-collision segment."

## 测试录制 S2_human_robot
python3 /ws/bdcc_exp/scripts/record/record_skeleton_segment.py \
  --scenario S2_human_robot \
  --outdir /ws/bdcc_exp/segments/S2_human_robot \
  --duration 60\
  --start-on-enter \
  --record-diagnostics \
  --notes "Real robot CBF-enabled recording for human-robot collision segment."


## 检查
ls -lh /ws/bdcc_exp/segments/S1_self_collision
python3 - <<'PY'
import numpy as np
p="/ws/bdcc_exp/segments/S1_self_collision/skeleton_filtered.npz"
d=np.load(p, allow_pickle=True)
print(d.files)
print("t shape:", d["t"].shape)
print("points shape:", d["points"].shape)
PY


## 回放
chmod +x /ws/bdcc_exp/scripts/replay/replay_skeleton_segment.py

## 回放 S1_self_collision
python3 /ws/bdcc_exp/scripts/replay/replay_skeleton_segment.py \
  --segment /ws/bdcc_exp/segments/S1_self_collision \
  --publish-mode filtered \
  --start-delay 3.0 \
  --replay-rate-hz 60 \
  --time-scale 1.0

## 回放 S2_human_robot
python3 /ws/bdcc_exp/scripts/replay/replay_skeleton_segment.py \
  --segment /ws/bdcc_exp/segments/S2_human_robot \
  --publish-mode filtered \
  --start-delay 3.0 \
  --replay-rate-hz 60 \
  --time-scale 1.0

第一次建议不要加 --interpolate-points，这样每帧就是录制时的 filtered skeleton 原值


## 回放记录
chmod +x /ws/bdcc_exp/scripts/log/trial_topic_logger.py


## 回放记录 S1_self_collision
python3 /ws/bdcc_exp/scripts/log/trial_topic_logger.py \
  --platform sim \
  --scenario S1_self_collision \
  --mode cbf \
  --run-id run_001 \
  --outdir /ws/bdcc_exp/runs/sim_default/S1_self_collision/run_001 \
  --duration 55 \
  --record-q-act \
  --record-cbf-diagnostics \
  --rr-safety-distance 0.05 \
  --hr-safety-distance 0.15 \
  --rr-gamma 2.0 \
  --hr-gamma 2.0

python3 /ws/bdcc_exp/scripts/log/trial_topic_logger.py \
  --platform real \
  --scenario S1_self_collision \
  --mode cbf \
  --run-id run_001 \
  --outdir /ws/bdcc_exp/runs/real_default/S1_self_collision/run_001 \
  --duration 65 \
  --record-q-act \
  --record-cbf-diagnostics \
  --rr-safety-distance 0.015 \
  --hr-safety-distance 0.10 \
  --rr-gamma 2.0 \
  --hr-gamma 2.0


## 离线计算
chmod +x /ws/bdcc_exp/scripts/offline/offline_compute_metrics.py

python3 /ws/bdcc_exp/scripts/offline/offline_compute_metrics.py \
  --run-dir /ws/bdcc_exp/runs/sim_default/S1_self_collision/run_001 \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --mode self_collision \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02


==============================================================================================
### 实验section1 sim
### self-collision
## 参数： 
sim_rr_safety_distanc=0.03
sim_rr_gamma=2.0
enable_self_collision: True
enable_human_collision: False

ros2 launch mujoco_g1 bdcc_unified_pipeline.launch.py run_sim:=true run_real:=false use_cbf:=true

python3 /ws/bdcc_exp/scripts/replay/replay_skeleton_segment.py \
  --segment /ws/bdcc_exp/segments/S1_self_collision \
  --publish-mode filtered \
  --start-delay 3.0 \
  --replay-rate-hz 60 \
  --time-scale 1.0

python3 /ws/bdcc_exp/scripts/log/trial_topic_logger.py \
  --platform sim \
  --scenario S1_self_collision \
  --mode cbf \
  --run-id run_001 \
  --outdir /ws/bdcc_exp/runs/sim_default/S1_self_collision/run_001 \
  --duration 55 \
  --record-q-act \
  --record-cbf-diagnostics \
  --rr-safety-distance 0.03 \
  --rr-gamma 2.0


python3 /ws/bdcc_exp/scripts/offline/offline_compute_metrics.py \
  --run-dir /ws/bdcc_exp/runs/sim_default/S1_self_collision/run_001 \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --mode self_collision \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02 \
  --eval-start-sec 8.0 \
  --eval-end-sec 50.0


python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/plot/plot_sim_self_collision.py \
  --run-dir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/sim_default/S1_self_collision/run_001 \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/sim_default/S1_self_collision \
  --smooth-window 3 \
  --angle-unit deg \
  --formats svg png


==============================================================================================
### 实验section1 sim
### human-robot
## 参数： 
sim_hr_safety_distance=0.10
sim_hr_gamma=2.0
enable_self_collision: False
enable_human_collision: True
y_distance: 0.80

ros2 launch mujoco_g1 bdcc_unified_pipeline.launch.py run_sim:=true run_real:=false use_cbf:=true

python3 /ws/bdcc_exp/scripts/replay/replay_skeleton_segment.py \
  --segment /ws/bdcc_exp/segments/S2_human_robot_2 \
  --publish-mode filtered \
  --start-delay 3.0 \
  --replay-rate-hz 60 \
  --time-scale 1.0

python3 /ws/bdcc_exp/scripts/log/trial_topic_logger.py \
  --platform sim \
  --scenario S2_human_robot_2 \
  --mode cbf \
  --run-id run_001 \
  --outdir /ws/bdcc_exp/runs/sim_default/S2_human_robot_2/run_001 \
  --duration 70 \
  --record-q-act \
  --record-cbf-diagnostics \
  --hr-safety-distance 0.10 \
  --hr-gamma 2.0


python3 /ws/bdcc_exp/scripts/offline/offline_compute_metrics.py \
  --run-dir /ws/bdcc_exp/runs/sim_default/S2_human_robot_2/run_001 \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --mode human_robot \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02 \
  --eval-start-sec 10.0 \
  --eval-end-sec 62.0


python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/plot/plot_sim_human_robot.py \
  --run-dir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/sim_default/S2_human_robot_2/run_001 \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/sim_default/S2_human_robot_2 \
  --smooth-window 3 \
  --angle-unit deg \
  --formats svg png


==============================================================================================
### 实验section1 real
### self-collision
## 参数： 
real_rr_safety_distance=0.03
real_rr_gamma=2.0
enable_self_collision: True
enable_human_collision: False

ros2 launch mujoco_g1 bdcc_unified_pipeline.launch.py run_sim:=false run_real:=true use_cbf:=true

python3 /ws/bdcc_exp/scripts/replay/replay_skeleton_segment.py \
  --segment /ws/bdcc_exp/segments/S1_self_collision \
  --publish-mode filtered \
  --start-delay 3.0 \
  --replay-rate-hz 60 \
  --time-scale 1.0

python3 /ws/bdcc_exp/scripts/log/trial_topic_logger.py \
  --platform real \
  --scenario S1_self_collision \
  --mode cbf \
  --run-id run_001 \
  --outdir /ws/bdcc_exp/runs/real_default/S1_self_collision/run_001 \
  --duration 55 \
  --record-q-act \
  --record-cbf-diagnostics \
  --rr-safety-distance 0.03 \
  --rr-gamma 2.0


python3 /ws/bdcc_exp/scripts/offline/offline_compute_metrics.py \
  --run-dir /ws/bdcc_exp/runs/real_default/S1_self_collision/run_001 \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --mode self_collision \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02 \
  --eval-start-sec 8.0 \
  --eval-end-sec 50.0


python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/plot/plot_real_self_collision.py \
  --run-dir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/S1_self_collision/run_001 \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/real_default/S1_self_collision \
  --smooth-window 3 \
  --angle-unit deg \
  --formats svg png


  ## 画表格
  python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/analyze_cbf_runtime_stats.py \
  --input /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/S1_self_collision/run_001/topics.npz \
  --eval-start-sec 8.0 \
  --eval-end-sec 50.0 \
  --rr-safety-distance 0.03 \
  --rr-gamma 2.0 \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/S1_self_collision/run_001 \
  --prefix cbf_runtime_stats

==============================================================================================
### 实验section1 real
### human-robot
## 参数： 
real_hr_safety_distance=0.15
real_hr_gamma=3.0
enable_self_collision: False
enable_human_collision: True
human_capsules_radius: same as sim
y_distance: 0.80

ros2 launch mujoco_g1 bdcc_unified_pipeline.launch.py run_sim:=false run_real:=true use_cbf:=true

python3 /ws/bdcc_exp/scripts/replay/replay_skeleton_segment.py \
  --segment /ws/bdcc_exp/segments/S2_human_robot_2 \
  --publish-mode filtered \
  --start-delay 3.0 \
  --replay-rate-hz 60 \
  --time-scale 1.0

python3 /ws/bdcc_exp/scripts/log/trial_topic_logger.py \
  --platform real \
  --scenario S2_human_robot_2 \
  --mode cbf \
  --run-id run_001 \
  --outdir /ws/bdcc_exp/runs/real_default/S2_human_robot_2/run_001 \
  --duration 70 \
  --record-q-act \
  --record-cbf-diagnostics \
  --hr-safety-distance 0.15 \
  --hr-gamma 3.0


python3 /ws/bdcc_exp/scripts/offline/offline_compute_metrics.py \
  --run-dir /ws/bdcc_exp/runs/real_default/S2_human_robot_2/run_001 \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --mode human_robot \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02 \
  --eval-start-sec 10.0 \
  --eval-end-sec 62.0


python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/plot/plot_real_human_robot.py \
  --run-dir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/S2_human_robot_2/run_001 \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/real_default/S2_human_robot_2 \
  --smooth-window 3 \
  --angle-unit deg \
  --formats svg png


## 画表格
python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/analyze_cbf_runtime_stats.py \
--input /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/S2_human_robot_2/run_001/topics.npz \
--eval-start-sec 10.0 \
--eval-end-sec 62.0 \
--hr-safety-distance 0.15 \
--hr-gamma 3.0 \
--outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/S2_human_robot_2/run_001 \
--prefix cbf_runtime_stats


### 实验section1 real
### human-robot
## 参数： 
real_rr_safety_distance=0.03
real_rr_gamma=2.0
real_hr_safety_distance=0.15
real_hr_gamma=3.0
enable_self_collision: True
enable_human_collision: True
human_capsules_radius: same as sim
y_distance: 0.85

ros2 launch mujoco_g1 bdcc_unified_pipeline.launch.py run_sim:=false run_real:=true use_cbf:=true

python3 /ws/bdcc_exp/scripts/replay/replay_skeleton_segment.py \
  --segment /ws/bdcc_exp/segments/S2_human_robot_2 \
  --publish-mode filtered \
  --start-delay 3.0 \
  --replay-rate-hz 60 \
  --time-scale 1.0

python3 /ws/bdcc_exp/scripts/log/trial_topic_logger.py \
  --platform real \
  --scenario S2_human_robot_2 \
  --mode cbf \
  --run-id run_001 \
  --outdir /ws/bdcc_exp/runs/real_default/both/run_001 \
  --duration 70 \
  --record-q-act \
  --record-cbf-diagnostics \
  --hr-safety-distance 0.15 \
  --hr-gamma 3.0 \
  --rr-safety-distance 0.03 \
  --rr-gamma 2.0


python3 /ws/bdcc_exp/scripts/offline/offline_compute_metrics.py \
  --run-dir /ws/bdcc_exp/runs/real_default/both/run_001 \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --mode both \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02 \
  --eval-start-sec 10.0 \
  --eval-end-sec 62.0


python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/plot/plot_real_both_collision.py \
  --run-dir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/both/run_001 \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/real_default/both \
  --smooth-window 3 \
  --angle-unit deg \
  --top-k-pairs 4 \
  --formats svg png


## 画表格
python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/analyze_cbf_runtime_stats.py \
--input /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/both/run_001/topics.npz \
--eval-start-sec 10.0 \
--eval-end-sec 62.0 \
--rr-safety-distance 0.03 \
--rr-gamma 2.0 \
--hr-safety-distance 0.15 \
--hr-gamma 3.0 \
--outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/runs/real_default/both/run_001 \
--prefix cbf_runtime_stats


==========================================================================================
## 实验section2 real
## phi_grid parameter sweep
## launch参数：
enable_self_collision: True
enable_human_collision: True
human_capsules_radius: same as sim
"extrinsic_ty": 0.95,
CBF "max_velocity": 1.0,
g1 "max_velocity": 1.0,

PHI_RR_VALUES = [0.005, 0.01, 0.02, 0.03, 0.04]
PHI_HR_VALUES = [0.09, 0.12, 0.15, 0.17, 0.19]
GAMMA_RR_VALUES = 2.0
GAMMA_HR_VALUES = 3.0

SCRIPT_ROOT=/ws/bdcc_exp/scripts/sweep
python3 $SCRIPT_ROOT/run_parameter_sweep.py \
  --platform real \
  --sweep-type phi_grid \
  --out-root /ws/bdcc_exp/sweeps/real_merge_phi_grid \
  --repeats 3 \
  --launch-wait-sec 3 \
  --shutdown-wait-sec 10 \
  --duration 70 \
  --eval-start-sec 10 \
  --eval-end-sec 62 \
  --rviz true


python3 /ws/bdcc_exp/scripts/sweep/aggregate_sweep_results.py \
  --sweep-root /ws/bdcc_exp/sweeps/real_merge_phi_grid \
  --compute-missing \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --eval-start-sec 10 \
  --eval-end-sec 62 \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02


python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/sweep/plot_sweep_heatmaps.py \
  --summary-csv /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/sweeps/real_merge_phi_grid/sweep_summary_agg.csv \
  --sweep-type phi_grid \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/sweeps/real_merge_phi_grid \
  --formats png svg \
  --annotate \
  --annotation-fontsize 7 \
  --palette soft_purple \
  --candidate-param-rr 0.01 \
  --candidate-param-hr 0.15 \
  --ctr-vmin 0 \
  --ctr-vmax 2.0


=============================================================================
## gamma_grid parameter sweep
## launch参数：
enable_self_collision: True
enable_human_collision: True
human_capsules_radius: same as sim
"extrinsic_ty": 0.95,
CBF "max_velocity": 1.0,
g1 "max_velocity": 1.0,

PHI_RR_VALUES = 0.03
PHI_HR_VALUES = 0.15
GAMMA_RR_VALUES = [0.5, 1.0, 2.0, 3.0, 3.5]
GAMMA_HR_VALUES = [1.5, 2.0, 3.0, 4.0, 4.5]

SCRIPT_ROOT=/ws/bdcc_exp/scripts/sweep
python3 $SCRIPT_ROOT/run_parameter_sweep.py \
  --platform real \
  --sweep-type gamma_grid \
  --out-root /ws/bdcc_exp/sweeps/real_merge_gamma_grid \
  --repeats 3 \
  --launch-wait-sec 3 \
  --shutdown-wait-sec 10 \
  --duration 70 \
  --eval-start-sec 10 \
  --eval-end-sec 62 \
  --rviz true


python3 /ws/bdcc_exp/scripts/sweep/aggregate_sweep_results.py \
  --sweep-root /ws/bdcc_exp/sweeps/real_merge_gamma_grid \
  --compute-missing \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --eval-start-sec 10 \
  --eval-end-sec 62 \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02


python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/sweep/plot_sweep_heatmaps.py \
  --summary-csv /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/sweeps/real_merge_gamma_grid/sweep_summary_agg.csv \
  --sweep-type gamma_grid \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/sweeps/real_merge_gamma_grid \
  --formats png svg \
  --annotate \
  --annotation-fontsize 7 \
  --palette soft_purple \
  --candidate-param-rr 3.0 \
  --candidate-param-hr 4.0


=============================================================================
## pareto parameter sweep
## launch参数：
enable_self_collision: True
enable_human_collision: True
human_capsules_radius: same as sim
"extrinsic_ty": 0.95,
CBF "max_velocity": 1.0,
g1 "max_velocity": 1.0,

PARETO_CANDIDATES = [
    (0.005, 0.09, 0.5, 1.5),
    (0.01, 0.12, 1.0, 2.0),
    (0.02, 0.15, 2.0, 3.0),
    (0.03, 0.15, 2.0, 3.0),
    (0.04, 0.19, 3.5, 4.5),
    (0.04, 0.17, 2.0, 3.0),
    (0.03, 0.19, 2.0, 4.5),
    (0.005, 0.19, 0.5, 4.5),
    (0.04, 0.09, 3.5, 1.5),
    (0.02, 0.17, 3.0, 4.0),
]

SCRIPT_ROOT=/ws/bdcc_exp/scripts/sweep
python3 $SCRIPT_ROOT/run_parameter_sweep.py \
  --platform real \
  --sweep-type pareto_samples \
  --out-root /ws/bdcc_exp/sweeps/real_merge_pareto_samples \
  --repeats 3 \
  --launch-wait-sec 3 \
  --shutdown-wait-sec 10 \
  --duration 70 \
  --eval-start-sec 10 \
  --eval-end-sec 62 \
  --rviz true


python3 /ws/bdcc_exp/scripts/sweep/aggregate_sweep_results.py \
  --sweep-root /ws/bdcc_exp/sweeps/real_merge_pareto_samples \
  --compute-missing \
  --urdf-path /ws/src/g1_cbf_ros2/g1_description/urdf/g1_29dof.urdf \
  --eval-start-sec 10 \
  --eval-end-sec 62 \
  --sample-rate-hz 50 \
  --max-lag-sec 2.0 \
  --lag-step-sec 0.02


# 画 composite Pareto
python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/sweep/plot_pareto_tradeoff.py \
  --summary-csv /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/sweeps/real_merge_phi_grid/sweep_summary_agg.csv \
  --summary-csv /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/sweeps/real_merge_gamma_grid/sweep_summary_agg.csv \
  --summary-csv /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/sweeps/real_merge_pareto_samples/sweep_summary_agg.csv \
  --pareto-mode composite \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/sweeps/real_merge_pareto \
  --formats png svg pdf \
  --show-errorbars \
  --mark-baseline


# 画 raw Pareto
python3 /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/scripts/sweep/plot_pareto_tradeoff.py \
  --summary-csv /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/sweeps/real_merge_phi_grid/sweep_summary_agg.csv \
  --summary-csv /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/sweeps/real_merge_gamma_grid/sweep_summary_agg.csv \
  --summary-csv /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/sweeps/real_merge_pareto_samples/sweep_summary_agg.csv \
  --pareto-mode raw \
  --outdir /home/wc3059/Projects/Human_Humanoid_Interaction/g1_real_ws/bdcc_exp/figures/sweeps/real_merge_pareto \
  --formats png svg pdf \
  --show-errorbars \
  --mark-baseline
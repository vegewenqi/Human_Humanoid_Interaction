# ROS2 env
source /opt/ros/humble/setup.bash 
source /ws/install/setup.bash 


exec ros2 run mujoco_g1_ik g1_ik_controller --ros-args \
    -p mjcf_path:="/third_party/mujoco_menagerie/unitree_g1/g1_mjx.xml" \
    -p ee_site:="right_palm" \
    -p elbow_body:="right_elbow_link" \
    -p wrist_index:=17 \
    -p elbow_index:=15 \
    -p pelvis_index:=0 \
    -p use_pelvis_relative:=true \
    -p motion_gain:=0.6 \
    -p max_delta_m:=0.35 \
    -p wrist_pos_cost:=1.0 \
    -p elbow_pos_cost:=0.45 \
    -p wrist_ori_cost:=0.08 \
    -p posture_cost:=0.18 \
    -p posture_max_vel:=0.8 \
    -p task_gain:=0.7 \
    -p max_delta_m:=0.35 \
    -p init_avg_frames:=15 \
    -p init_min_conf:=85 \
    -p home_shoulder_pitch:=0.0 \
    -p home_shoulder_roll:=0.0 \
    -p home_shoulder_yaw:=0.0 \
    -p home_elbow:=-0.60 \
    -p home_wrist_roll:=0.00 \
    -p home_wrist_pitch:=0.00 \
    -p home_wrist_yaw:=0.00 \
    -p elbow_avoid_gain:=0.8 \
    -p elbow_avoid_margin_y:=0.18 \
    -p elbow_avoid_margin_x:=0.02
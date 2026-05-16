from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    Command,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # -------- high-level switches --------
    run_estimator = LaunchConfiguration("run_estimator")
    run_mapper = LaunchConfiguration("run_mapper")
    run_sim = LaunchConfiguration("run_sim")
    run_real = LaunchConfiguration("run_real")
    use_cbf = LaunchConfiguration("use_cbf")
    ghost = LaunchConfiguration("ghost")
    rviz = LaunchConfiguration("rviz")

    # -------- shared upstream topics --------
    skeleton_points_topic = LaunchConfiguration("skeleton_points_topic")
    skeleton_points_filtered_topic = LaunchConfiguration("skeleton_points_filtered_topic")
    qdes_nominal_topic = LaunchConfiguration("qdes_nominal_topic")
    unsafe_joint_command_topic = LaunchConfiguration("unsafe_joint_command_topic")

    # -------- sim topics --------
    sim_joint_state_topic = LaunchConfiguration("sim_joint_state_topic")
    sim_safe_joint_command_topic = LaunchConfiguration("sim_safe_joint_command_topic")
    sim_safe_qdes_topic = LaunchConfiguration("sim_safe_qdes_topic")
    sim_human_capsule_topic = LaunchConfiguration("sim_human_capsule_topic")

    # -------- real topics --------
    real_joint_state_topic = LaunchConfiguration("real_joint_state_topic")
    real_safe_joint_command_topic = LaunchConfiguration("real_safe_joint_command_topic")
    real_safe_qdes_topic = LaunchConfiguration("real_safe_qdes_topic")
    real_human_capsule_topic = LaunchConfiguration("real_human_capsule_topic")

    # -------- ghost topic --------
    ghost_joint_state_topic = LaunchConfiguration("ghost_joint_state_topic")

    # -------- misc params --------
    mjcf_path = LaunchConfiguration("mjcf_path")

    # -------- CBF sweep parameters --------
    sim_rr_safety_distance = LaunchConfiguration("sim_rr_safety_distance")
    sim_hr_safety_distance = LaunchConfiguration("sim_hr_safety_distance")
    sim_rr_gamma = LaunchConfiguration("sim_rr_gamma")
    sim_hr_gamma = LaunchConfiguration("sim_hr_gamma")
    real_rr_safety_distance = LaunchConfiguration("real_rr_safety_distance")
    real_hr_safety_distance = LaunchConfiguration("real_hr_safety_distance")
    real_rr_gamma = LaunchConfiguration("real_rr_gamma")
    real_hr_gamma = LaunchConfiguration("real_hr_gamma")

    # -------- CBF diagnostics topics --------
    sim_cbf_diagnostics_topic = LaunchConfiguration("sim_cbf_diagnostics_topic")
    sim_cbf_diagnostics_pair_topic = LaunchConfiguration("sim_cbf_diagnostics_pair_topic")
    real_cbf_diagnostics_topic = LaunchConfiguration("real_cbf_diagnostics_topic")
    real_cbf_diagnostics_pair_topic = LaunchConfiguration("real_cbf_diagnostics_pair_topic")

    # -------- human capsule radii --------
    human_torso_radius_sim = LaunchConfiguration("human_torso_radius_sim")
    human_upper_arm_radius_sim = LaunchConfiguration("human_upper_arm_radius_sim")
    human_forearm_radius_sim = LaunchConfiguration("human_forearm_radius_sim")
    human_thigh_radius_sim = LaunchConfiguration("human_thigh_radius_sim")
    human_shin_radius_sim = LaunchConfiguration("human_shin_radius_sim")
    human_head_radius_sim = LaunchConfiguration("human_head_radius_sim")
    human_torso_radius_real = LaunchConfiguration("human_torso_radius_real")
    human_upper_arm_radius_real = LaunchConfiguration("human_upper_arm_radius_real")
    human_forearm_radius_real = LaunchConfiguration("human_forearm_radius_real")
    human_thigh_radius_real = LaunchConfiguration("human_thigh_radius_real")
    human_shin_radius_real = LaunchConfiguration("human_shin_radius_real")
    human_head_radius_real = LaunchConfiguration("human_head_radius_real")

    # If run_real is true, use the real-human radius profile; otherwise use sim.
    human_torso_radius = PythonExpression([
        human_torso_radius_real, " if '", run_real, "' == 'true' else ", human_torso_radius_sim
    ])
    human_upper_arm_radius = PythonExpression([
        human_upper_arm_radius_real, " if '", run_real, "' == 'true' else ", human_upper_arm_radius_sim
    ])
    human_forearm_radius = PythonExpression([
        human_forearm_radius_real, " if '", run_real, "' == 'true' else ", human_forearm_radius_sim
    ])
    human_thigh_radius = PythonExpression([
        human_thigh_radius_real, " if '", run_real, "' == 'true' else ", human_thigh_radius_sim
    ])
    human_shin_radius = PythonExpression([
        human_shin_radius_real, " if '", run_real, "' == 'true' else ", human_shin_radius_sim
    ])
    human_head_radius = PythonExpression([
        human_head_radius_real, " if '", run_real, "' == 'true' else ", human_head_radius_sim
    ])

    g1_urdf = PathJoinSubstitution([
        FindPackageShare("g1_description"),
        "urdf",
        "g1_29dof.urdf",
    ])

    sim_rviz_config = PathJoinSubstitution([
        FindPackageShare("g1_cbf"),
        "rviz",
        "config_sim.rviz",
    ])
    real_rviz_config = PathJoinSubstitution([
        FindPackageShare("g1_cbf"),
        "rviz",
        "config_real.rviz",
    ])

    # -------- compound conditions --------
    sim_nominal_cond = IfCondition(
        PythonExpression(["'", run_sim, "' == 'true' and '", use_cbf, "' == 'false'"])
    )
    sim_cbf_cond = IfCondition(
        PythonExpression(["'", run_sim, "' == 'true' and '", use_cbf, "' == 'true'"])
    )
    real_nominal_cond = IfCondition(
        PythonExpression(["'", run_real, "' == 'true' and '", use_cbf, "' == 'false'"])
    )
    real_cbf_cond = IfCondition(
        PythonExpression(["'", run_real, "' == 'true' and '", use_cbf, "' == 'true'"])
    )
    ghost_cond = IfCondition(
        PythonExpression([
            "'", run_sim, "' == 'true' and '",
            use_cbf, "' == 'true' and '",
            ghost, "' == 'true'"
        ])
    )
    # only launch rviz if we're running cbf in either sim or real, since that's the main use case for visualization
    rviz_cond_sim = IfCondition(
        PythonExpression(["'", rviz, "' == 'true' and '", use_cbf, "' == 'true' and '", run_sim, "' == 'true'"])
    )  
    rviz_cond_real = IfCondition(
        PythonExpression(["'", rviz, "' == 'true' and '", use_cbf, "' == 'true' and '", run_real, "' == 'true'"])
    )

    return LaunchDescription([
        # ---------------- launch args ----------------
        DeclareLaunchArgument('sigterm_timeout', default_value='15'),
        DeclareLaunchArgument('sigkill_timeout', default_value='10'),

        DeclareLaunchArgument("run_estimator", default_value="true"),
        DeclareLaunchArgument("run_mapper", default_value="true"),
        DeclareLaunchArgument("run_sim", default_value="true"),
        DeclareLaunchArgument("run_real", default_value="false"),
        DeclareLaunchArgument("use_cbf", default_value="true"),
        DeclareLaunchArgument("ghost", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),

        DeclareLaunchArgument("mjcf_path", default_value="/repos/unitree_g1/g1_mjx.xml"),

        DeclareLaunchArgument("skeleton_points_topic", default_value="/skeleton/points"),
        DeclareLaunchArgument("skeleton_points_filtered_topic", default_value="/skeleton/points_filtered"),
        DeclareLaunchArgument("qdes_nominal_topic", default_value="/g1_upperbody_q_des"),
        DeclareLaunchArgument("unsafe_joint_command_topic", default_value="/joint_commands_unsafe"),

        DeclareLaunchArgument("sim_joint_state_topic", default_value="/sim/joint_states"),
        DeclareLaunchArgument("sim_safe_joint_command_topic", default_value="/sim/joint_commands"),
        DeclareLaunchArgument("sim_safe_qdes_topic", default_value="/sim/g1_upperbody_q_des_safe"),
        DeclareLaunchArgument("sim_human_capsule_topic", default_value="/sim/human_capsules_robot"),

        DeclareLaunchArgument("real_joint_state_topic", default_value="/real/joint_states"),
        DeclareLaunchArgument("real_safe_joint_command_topic", default_value="/real/joint_commands"),
        DeclareLaunchArgument("real_safe_qdes_topic", default_value="/real/g1_upperbody_q_des_safe"),
        DeclareLaunchArgument("real_human_capsule_topic", default_value="/real/human_capsules_robot"),

        DeclareLaunchArgument("ghost_joint_state_topic", default_value="/ghost/joint_states"),

        # CBF sweep parameters. Defaults preserve the previous sim/real values.
        DeclareLaunchArgument("sim_rr_safety_distance", default_value="0.03"),
        DeclareLaunchArgument("sim_hr_safety_distance", default_value="0.10"),
        DeclareLaunchArgument("sim_rr_gamma", default_value="2.0"),
        DeclareLaunchArgument("sim_hr_gamma", default_value="2.0"),
        
        DeclareLaunchArgument("real_rr_safety_distance", default_value="0.03"),
        DeclareLaunchArgument("real_hr_safety_distance", default_value="0.10"),
        DeclareLaunchArgument("real_rr_gamma", default_value="2.0"),
        DeclareLaunchArgument("real_hr_gamma", default_value="2.0"),

        # CBF diagnostics topics. The numeric Float32MultiArray layout is documented
        # in g1_cbf_node.py; the pair topic publishes the current min-pair label.
        DeclareLaunchArgument("sim_cbf_diagnostics_topic", default_value="/sim/cbf/diagnostics"),
        DeclareLaunchArgument("sim_cbf_diagnostics_pair_topic", default_value="/sim/cbf/min_control_pair"),
        DeclareLaunchArgument("real_cbf_diagnostics_topic", default_value="/real/cbf/diagnostics"),
        DeclareLaunchArgument("real_cbf_diagnostics_pair_topic", default_value="/real/cbf/min_control_pair"),

        # Human capsule radii. When run_real:=true, the real profile is used;
        # otherwise the sim profile is used.
        # DeclareLaunchArgument("human_torso_radius_real", default_value="0.15"),
        # DeclareLaunchArgument("human_upper_arm_radius_real", default_value="0.09"),
        # DeclareLaunchArgument("human_forearm_radius_real", default_value="0.10"),
        # DeclareLaunchArgument("human_thigh_radius_real", default_value="0.08"),
        # DeclareLaunchArgument("human_shin_radius_real", default_value="0.07"),
        # DeclareLaunchArgument("human_head_radius_real", default_value="0.10"),

        DeclareLaunchArgument("human_torso_radius_sim", default_value="0.15"),
        DeclareLaunchArgument("human_upper_arm_radius_sim", default_value="0.06"),
        DeclareLaunchArgument("human_forearm_radius_sim", default_value="0.05"),
        DeclareLaunchArgument("human_thigh_radius_sim", default_value="0.08"),
        DeclareLaunchArgument("human_shin_radius_sim", default_value="0.07"),
        DeclareLaunchArgument("human_head_radius_sim", default_value="0.10"),


        DeclareLaunchArgument("human_torso_radius_real", default_value="0.15"),
        DeclareLaunchArgument("human_upper_arm_radius_real", default_value="0.06"),
        DeclareLaunchArgument("human_forearm_radius_real", default_value="0.05"),
        DeclareLaunchArgument("human_thigh_radius_real", default_value="0.08"),
        DeclareLaunchArgument("human_shin_radius_real", default_value="0.07"),
        DeclareLaunchArgument("human_head_radius_real", default_value="0.10"),

        # ---------------- shared ZED skeleton points pre-processor ----------------
        Node(
            package="mujoco_g1",
            executable="zed_skeleton_points_preprocessor",
            name="zed_skeleton_points_preprocessor",
            output="screen",
            parameters=[{
                "input_points_topic": skeleton_points_topic,
                "input_conf_topic": "/skeleton/confidence",
                "output_points_topic": skeleton_points_filtered_topic,
                "min_confidence": 60,
                "point_ema_alpha": 0.30,
                "point_max_jump": 2.0,
                "point_max_reject_count": 3,
            }],
        ),

        #---------------- human skeleton capsule ----------------
                
        Node(
            package="mujoco_g1",
            executable="human_skeleton_capsule",
            name="human_skeleton_capsule",
            output="screen",
            parameters=[{
                "input_points_topic": skeleton_points_filtered_topic,
                "input_conf_topic": "/skeleton/confidence",
                "min_confidence": 60,
                "capsule_zed_topic": "/human_capsules_zed",
                "capsule_local_topic": "/human_capsules_local",

                "torso_radius": human_torso_radius,
                "upper_arm_radius": human_upper_arm_radius,
                "forearm_radius": human_forearm_radius,
                "thigh_radius": human_thigh_radius,
                "shin_radius": human_shin_radius,
                "head_radius": human_head_radius,
            }],
        ),

        # ---------------- human capsule transform: sim ----------------
        Node(
            package="mujoco_g1",
            executable="human_capsule_frame_transform",
            name="human_capsule_frame_transform_sim",
            output="screen",
            condition=IfCondition(run_sim),
            parameters=[{
                "mode": "sim",
                "input_topic": "/human_capsules_local",
                "output_topic": sim_human_capsule_topic,
                "marker_topic": "/sim/human_capsules_markers_robot",
                "target_frame": "pelvis",
                # "enable_footprint_marker": False,

                "align_roll_deg": 0.0,
                "align_pitch_deg": 0.0,
                "align_yaw_deg": 180.0,

                "tx": 0.0,
                "ty": 0.80,
                "tz": 0.0,
                "yaw_deg": 0.0,
            }],
        ),

        # ---------------- human capsule transform: real ----------------
        Node(
            package="mujoco_g1",
            executable="human_capsule_frame_transform",
            name="human_capsule_frame_transform_real",
            output="screen",
            condition=IfCondition(run_real),
            parameters=[{
                "mode": "real_quick_cali",
                "input_topic": "/human_capsules_zed",
                "output_topic": real_human_capsule_topic,
                "marker_topic": "/real/human_capsules_markers_robot",
                "target_frame": "pelvis",

                # "extrinsic_tx": 2.118275508612908,
                # "extrinsic_ty": 0.6912551371444147,
                # "extrinsic_tz": -1.094470867712238,
                # "extrinsic_qx": 0.2847651373649267,
                # "extrinsic_qy": 0.12968335366406136,
                # "extrinsic_qz": 0.9467723983663323,
                # "extrinsic_qw": -0.07558485308340035,
                # "extrinsic_qx": 0.0,
                # "extrinsic_qy": 0.0,
                # "extrinsic_qz": 1.0,
                # "extrinsic_qw": 0.0,

                "extrinsic_tx": 0.0,
                "extrinsic_ty": 0.90,
                # "extrinsic_ty": 1.20,
                "extrinsic_tz": 0.61,
                "extrinsic_qx": 0.0,
                "extrinsic_qy": 0.0,
                "extrinsic_qz": 1.0,
                "extrinsic_qw": 0.0,
            }],
        ),

        # ---------------- shared upstream ----------------
        Node(
            package="mujoco_g1",
            executable="human_angle_estimator",
            name="human_angle_estimator",
            output="screen",
            condition=IfCondition(run_estimator),
            parameters=[{
                "input_points_topic": skeleton_points_filtered_topic,
                "input_conf_topic": "/skeleton/confidence",
                "min_confidence": 60,
                "angle_ema_alpha": 0.25,
                "angle_max_rate_deg": 100.0,
                "debug_log": False,
                "debug_log_period_sec": 1.0,
            }],
        ),

        Node(
            package="mujoco_g1",
            executable="g1_joint_mapper",
            name="g1_joint_mapper",
            output="screen",
            condition=IfCondition(run_mapper),
            parameters=[{
                "output_topic": qdes_nominal_topic,
                "unsafe_joint_command_topic": unsafe_joint_command_topic,
                "debug_log": False,
                "debug_log_period_sec": 1.0,
            }],
        ),

        # ============================================================
        # SIMULATION PATH
        # ============================================================

        # ---- sim nominal ----
        Node(
            package="mujoco_g1",
            executable="g1_controller",
            name="g1_controller_nominal",
            output="screen",
            condition=sim_nominal_cond,
            parameters=[{
                "mjcf_path": mjcf_path,
                "qdes_topic": qdes_nominal_topic,
                "joint_state_topic": sim_joint_state_topic,
                "debug_log": False,
                "debug_log_period_sec": 1.0,
                "show_viewer": True,
                "ema_alpha": 1.0,
                "max_rate_deg": 1080.0,
            }],
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher_sim_nominal",
            output="screen",
            condition=sim_nominal_cond,
            parameters=[{
                "robot_description": Command(["cat ", g1_urdf]),
            }],
            remappings=[
                ("joint_states", sim_joint_state_topic),
            ],
        ),

        # ---- sim cbf ----
        Node(
            package="g1_cbf",
            executable="g1_cbf_node",
            name="g1_cbf_node_sim",
            output="screen",
            condition=sim_cbf_cond,
            parameters=[
                {
                    "urdf_path": g1_urdf,
                    "joint_state_topic": sim_joint_state_topic,
                    "unsafe_cmd_topic": unsafe_joint_command_topic,
                    "safe_cmd_topic": sim_safe_joint_command_topic,
                    "human_capsule_topic": sim_human_capsule_topic,
                    "obstacle_topic": "/sim/bbox_3d",
                    "collision_geometry": "capsules",
                    "K": 30.0,
                    "max_velocity": 10.0,
                    "lpf_gain": 0.5,
                    "dt": 1.0/25.0,
                    "rr_safety_distance": sim_rr_safety_distance,
                    "rr_gamma": sim_rr_gamma,
                    "hr_safety_distance": sim_hr_safety_distance,
                    "hr_gamma": sim_hr_gamma,

                    # "use_gpu": True,
                    "enable_self_collision": True,
                    "enable_human_collision": True,

                    "enable_robot_caps_viz": True,
                    "enable_distance_viz": True,
                    "log_summary": True,
                    "summary_period_sec": 1.0,
                    "enable_diagnostics": True,
                    "diagnostics_topic": sim_cbf_diagnostics_topic,
                    "diagnostics_pair_topic": sim_cbf_diagnostics_pair_topic,
                    "enable_coarse_gating": True,
                    "coarse_distance_activate": 0.55,

                    "enable_dynamic_human_cbf": True,
                    "human_velocity_lpf_alpha": 0.5,
                    "human_velocity_max": 2.0,
                    "human_velocity_dt_min": 0.03,
                    "human_velocity_dt_max": 0.15,
                }
            ],
        ),

        Node(
            package="mujoco_g1",
            executable="jointstate_to_array_qdes",
            name="jointstate_to_array_qdes_sim",
            output="screen",
            condition=sim_cbf_cond,
            parameters=[{
                "input_topic": sim_safe_joint_command_topic,
                "output_topic": sim_safe_qdes_topic,
            }],
        ),

        Node(
            package="mujoco_g1",
            executable="g1_controller",
            name="g1_controller_sim",
            output="screen",
            condition=sim_cbf_cond,
            parameters=[{
                "mjcf_path": mjcf_path,
                "qdes_topic": sim_safe_qdes_topic,
                "joint_state_topic": sim_joint_state_topic,
                "debug_log": False,
                "debug_log_period_sec": 1.0,
                "show_viewer": False,
                "ema_alpha": 1.0,
                "max_rate_deg": 1080.0,
            }],
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher_sim",
            output="screen",
            condition=sim_cbf_cond,
            parameters=[{
                "robot_description": Command(["cat ", g1_urdf]),
            }],
            remappings=[
                ("joint_states", sim_joint_state_topic),
            ],
        ),

        # ---- ghost for sim visualization ----
        GroupAction(
            condition=ghost_cond,
            actions=[
                Node(
                    package="g1_cbf",
                    executable="ghost_publisher_node",
                    name="ghost_publisher_node",
                    output="screen",
                    parameters=[{
                        "joint_state_topic": sim_joint_state_topic,
                        "unsafe_topic": unsafe_joint_command_topic,
                        "ghost_topic": ghost_joint_state_topic,
                    }],
                ),

                Node(
                    package="robot_state_publisher",
                    executable="robot_state_publisher",
                    name="ghost_robot_state_publisher",
                    output="screen",
                    parameters=[{
                        "robot_description": Command(["cat ", g1_urdf]),
                        "frame_prefix": "ghost/",
                    }],
                    remappings=[
                        ("joint_states", ghost_joint_state_topic),
                    ],
                ),
            ],
        ),

        # ---- rviz ----
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=rviz_cond_sim,
            arguments=["-d", sim_rviz_config],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=rviz_cond_real,
            arguments=["-d", real_rviz_config],
        ),

        # ============================================================
        # REAL ROBOT PATH
        # ============================================================

        # ---- real nominal ----
        Node(
            package="real_g1",
            executable="g1_arm_sdk_bridge",
            name="g1_arm_sdk_bridge_nominal",
            output="screen",
            condition=real_nominal_cond,
            parameters=[{
                "qdes_topic": qdes_nominal_topic,
                "joint_state_topic": real_joint_state_topic,
                "qdes_in_degrees": False,

                "control_dt": 0.01,
                "ema_alpha": 0.5,
                "max_joint_velocity": 6.0,
                # "topic_timeout_sec": 0.30,
                "home_transition_velocity": 0.20,
                "shutdown_return_velocity": 0.20,
                "weight_acquire_rate": 0.30,
                "weight_release_rate": 0.30,

                "kp_arm": 50.0,
                "kd_arm": 2.0,
                "kp_waist": 100.0,
                "kd_waist": 20.0,
                "dq": 0.0,
                "tau_ff": 0.0,
                "weight_active": 1.0,
                "shutdown_release_sec": 2.0,

                "enable_waist_balance_offset": True,
                "waist_pitch_balance_gain": 0.04,
                "waist_pitch_balance_limit": 0.20,
                "waist_roll_from_pitch_asym_gain": 0.005,
                "waist_roll_from_pitch_asym_limit": 0.05,
                "waist_roll_from_roll_asym_gain": 0.017,
                "waist_roll_from_roll_asym_limit": 0.12,

                "q_home_8": [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708],
                "q_min_8": [-0.52, -0.52, -3.0892, -1.5882, 0, -3.0892, -2.2515, 0],
                "q_max_8": [0.52, 0.52,  2.6704,  2.2515,  2.0944,  2.6704,  1.5882,  2.0944],

                "debug_log": False,
                "debug_log_period_sec": 1.0,
            }],
        ),

        # TODO:
        # If sim and real need to be visualized at the same time,
        # add frame_prefix for the real robot_state_publisher
        # to avoid TF/frame name conflicts with sim.
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher_real_nominal",
            output="screen",
            condition=real_nominal_cond,
            parameters=[{
                "robot_description": Command(["cat ", g1_urdf]),
            }],
            remappings=[
                ("joint_states", real_joint_state_topic),
            ],
        ),

        # ---- real cbf ----
        Node(
            package="g1_cbf",
            executable="g1_cbf_node",
            name="g1_cbf_node_real",
            output="screen",
            condition=real_cbf_cond,
            parameters=[
                {
                    "urdf_path": g1_urdf,
                    "joint_state_topic": real_joint_state_topic,
                    "unsafe_cmd_topic": unsafe_joint_command_topic,
                    "safe_cmd_topic": real_safe_joint_command_topic,
                    "human_capsule_topic": real_human_capsule_topic,
                    "obstacle_topic": "/real/bbox_3d",
                    "collision_geometry": "capsules",
                    "K": 30.0,
                    "max_velocity": 10.0,
                    "lpf_gain": 0.5,
                    "dt": 1.0/25.0,
                    "rr_safety_distance": real_rr_safety_distance,
                    "rr_gamma": real_rr_gamma,
                    "hr_safety_distance": real_hr_safety_distance,
                    "hr_gamma": real_hr_gamma,

                    # "use_gpu": True,
                    "enable_self_collision": True,
                    "enable_human_collision": True,

                    "enable_robot_caps_viz": True,
                    "enable_distance_viz": True,
                    "log_summary": True,
                    "summary_period_sec": 1.0,
                    "enable_diagnostics": True,
                    "diagnostics_topic": real_cbf_diagnostics_topic,
                    "diagnostics_pair_topic": real_cbf_diagnostics_pair_topic,
                    "enable_coarse_gating": True,
                    "coarse_distance_activate": 0.55,

                    "enable_dynamic_human_cbf": True,
                    "human_velocity_lpf_alpha": 0.5,
                    "human_velocity_max": 2.0,
                    "human_velocity_dt_min": 0.03,
                    "human_velocity_dt_max": 0.15,
                }
            ],
        ),

        Node(
            package="mujoco_g1",
            executable="jointstate_to_array_qdes",
            name="jointstate_to_array_qdes_real",
            output="screen",
            condition=real_cbf_cond,
            parameters=[{
                "input_topic": real_safe_joint_command_topic,
                "output_topic": real_safe_qdes_topic,
            }],
        ),

        Node(
            package="real_g1",
            executable="g1_arm_sdk_bridge",
            name="g1_arm_sdk_bridge_real",
            output="screen",
            condition=real_cbf_cond,
            parameters=[{
                "qdes_topic": real_safe_qdes_topic,
                "joint_state_topic": real_joint_state_topic,
                "qdes_in_degrees": False,

                "control_dt": 0.01,
                "ema_alpha": 0.5,
                "max_joint_velocity": 6.0,
                # "topic_timeout_sec": 0.30,
                "home_transition_velocity": 0.20,
                "shutdown_return_velocity": 0.20,
                "weight_acquire_rate": 0.30,
                "weight_release_rate": 0.30,

                "kp_arm": 50.0,
                "kd_arm": 2.0,
                "kp_waist": 100.0,
                "kd_waist": 20.0,
                "dq": 0.0,
                "tau_ff": 0.0,
                "weight_active": 1.0,
                "shutdown_release_sec": 2.0,

                "enable_waist_balance_offset": True,
                "waist_pitch_balance_gain": 0.04,
                "waist_pitch_balance_limit": 0.20,
                "waist_roll_from_pitch_asym_gain": 0.005,
                "waist_roll_from_pitch_asym_limit": 0.05,
                "waist_roll_from_roll_asym_gain": 0.017,
                "waist_roll_from_roll_asym_limit": 0.12,


                "q_home_8": [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708],
                "q_min_8": [-0.52, -0.52, -3.0892, -1.5882, 0, -3.0892, -2.2515, 0],
                "q_max_8": [0.52, 0.52,  2.6704,  2.2515,  2.0944,  2.6704,  1.5882,  2.0944],
                
                "debug_log": False,
                "debug_log_period_sec": 1.0,
            }],
        ),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher_real",
            output="screen",
            condition=real_cbf_cond,
            parameters=[{
                "robot_description": Command(["cat ", g1_urdf]),
            }],
            remappings=[
                ("joint_states", real_joint_state_topic),
            ],
        ),
    ])
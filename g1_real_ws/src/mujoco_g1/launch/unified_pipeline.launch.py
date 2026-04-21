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

    g1_urdf = PathJoinSubstitution([
        FindPackageShare("g1_description"),
        "urdf",
        "g1_29dof.urdf",
    ])

    rviz_config = PathJoinSubstitution([
        FindPackageShare("g1_cbf"),
        "rviz",
        "config.rviz",
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
    rviz_cond = IfCondition(
        PythonExpression(["'", rviz, "' == 'true' and '", use_cbf, "' == 'true'"])
    )  # only launch rviz if we're running cbf in either sim or real, since that's the main use case for visualization

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
                "min_confidence": 40,
                "point_ema_alpha": 0.5,
                "point_max_jump": 1.0,
                "point_max_reject_count": 5,
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
                "min_confidence": 40,
                "capsule_zed_topic": "/human_capsules_zed",
                "capsule_local_topic": "/human_capsules_local",
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

                "align_roll_deg": 0.0,
                "align_pitch_deg": 0.0,
                "align_yaw_deg": 180.0,

                "tx": 0.0,
                "ty": 0.72,
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

                "extrinsic_tx": 0.0,
                "extrinsic_ty": 0.8,
                "extrinsic_tz": 0.15,
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
                "min_confidence": 40,
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
                    "K": 75.0,
                    "max_velocity": 10.0,
                    "lpf_gain": 1.0,
                    "rr_margin_phi": 0.003,
                    "hr_margin_phi": 0.03,
                    "rr_gamma": 2.0,
                    "hr_gamma": 2.0,
                    # "enable_human_collision": False
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
            condition=rviz_cond,
            arguments=["-d", rviz_config],
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

                "control_dt": 0.02,
                "ema_alpha": 0.2,
                "max_joint_velocity": 0.5,
                "home_transition_velocity": 0.10,
                "shutdown_return_velocity": 0.10,
                "weight_acquire_rate": 0.10,
                "weight_release_rate": 0.10,

                "kp_arm": 60.0,
                "kd_arm": 1.5,
                "kp_waist": 80.0,
                "kd_waist": 5.0,
                "dq": 0.0,
                "tau_ff": 0.0,
                "weight_active": 1.0,
                "shutdown_release_sec": 2.0,

                "q_home_8": [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708],
                "q_min_8": [-0.52, -0.52, -3.0892, -1.5882, -1.0472, -3.0892, -2.2515, -1.0472],
                "q_max_8": [0.52, 0.52,  2.6704,  2.2515,  2.0944,  2.6704,  1.5882,  2.0944],
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
                    "K": 75.0,
                    "max_velocity": 10.0,
                    "lpf_gain": 1.0,
                    "rr_margin_phi": 0.003,
                    "hr_margin_phi": 0.03,
                    "rr_gamma": 2.0,
                    "hr_gamma": 2.0,
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

                "control_dt": 0.02,
                "ema_alpha": 0.2,
                "max_joint_velocity": 0.5,
                "home_transition_velocity": 0.10,
                "shutdown_return_velocity": 0.10,
                "weight_acquire_rate": 0.10,
                "weight_release_rate": 0.10,

                "kp_arm": 60.0,
                "kd_arm": 1.5,
                "kp_waist": 80.0,
                "kd_waist": 5.0,
                "dq": 0.0,
                "tau_ff": 0.0,
                "weight_active": 1.0,
                "shutdown_release_sec": 2.0,

                "q_home_8": [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708],
                "q_min_8": [-0.52, -0.52, -3.0892, -1.5882, -1.0472, -3.0892, -2.2515, -1.0472],
                "q_max_8": [0.52, 0.52,  2.6704,  2.2515,  2.0944,  2.6704,  1.5882,  2.0944],
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
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    ghost = LaunchConfiguration("ghost")
    rviz = LaunchConfiguration("rviz")
    use_cbf = LaunchConfiguration("use_cbf")
    urdf_file = LaunchConfiguration("urdf_file")
    mjcf_path = LaunchConfiguration("mjcf_path")

    qdes_nominal_topic = LaunchConfiguration("qdes_nominal_topic")
    qdes_safe_topic = LaunchConfiguration("qdes_safe_topic")
    unsafe_joint_command_topic = LaunchConfiguration("unsafe_joint_command_topic")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    safe_joint_command_topic = LaunchConfiguration("safe_joint_command_topic")
    ghost_joint_state_topic = LaunchConfiguration("ghost_joint_state_topic")

    g1_cbf_params = PathJoinSubstitution([
    FindPackageShare("g1_cbf"),
        "config",
        "params.yaml",
    ])

    urdf_file = PathJoinSubstitution([
        FindPackageShare("g1_description"),
        "urdf",
        "g1_29dof.urdf",
    ])

    rviz_config = PathJoinSubstitution([
        FindPackageShare("g1_cbf"),
        "rviz",
        "config.rviz",
    ])

    mjcf_path = "/repos/unitree_g1/g1_mjx.xml"

    return LaunchDescription([
        DeclareLaunchArgument("ghost", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("use_cbf", default_value="true"),

        DeclareLaunchArgument(
            "urdf_file",
            default_value=urdf_file
        ),
        DeclareLaunchArgument(
            "mjcf_path",
            default_value=mjcf_path
        ),

        # nominal / safe topic layout
        DeclareLaunchArgument(
            "qdes_nominal_topic",
            default_value="/g1_upperbody_q_des"
        ),
        DeclareLaunchArgument(
            "qdes_safe_topic",
            default_value="/g1_upperbody_q_des_safe"
        ),
        DeclareLaunchArgument(
            "unsafe_joint_command_topic",
            default_value="/joint_commands_unsafe"
        ),
        DeclareLaunchArgument(
            "joint_state_topic",
            default_value="/joint_states"
        ),
        DeclareLaunchArgument(
            "safe_joint_command_topic",
            default_value="/joint_commands"
        ),
        DeclareLaunchArgument(
            "ghost_joint_state_topic", 
            default_value="/ghost/joint_states"
        ),

        # 1) Human angle estimator
        Node(
            package="mujoco_g1",
            executable="human_angle_estimator",
            name="human_angle_estimator",
            output="screen",
        ),

        # 2) Joint mapper
        Node(
            package="mujoco_g1",
            executable="g1_joint_mapper",
            name="g1_joint_mapper",
            output="screen",
            parameters=[{
                "output_topic": qdes_nominal_topic,
                "unsafe_joint_command_topic": unsafe_joint_command_topic,
            }],
        ),

        # 3) CBF node
        Node(
            package="g1_cbf",
            executable="g1_cbf_node",
            name="g1_cbf_node",
            output="screen",
            condition=IfCondition(use_cbf),
            parameters=[
                g1_cbf_params,
                {
                    "urdf_path": urdf_file,
                    # keep colleague defaults unless you want to override
                    # "collision_geometry": "capsules",
                    # "dt": 0.02,
                    # "gamma": 5.0,
                    # "K": 5.0,
                    # "max_velocity": 0.5,
                    # "lpf_gain": 0.1,
                }
            ],
        ),

        # 4) Convert safe JointState -> safe 8D array qdes
        Node(
            package="mujoco_g1",
            executable="jointstate_to_array_qdes",
            name="jointstate_to_array_qdes",
            output="screen",
            condition=IfCondition(use_cbf),
            parameters=[{
                "input_topic": safe_joint_command_topic,
                "output_topic": qdes_safe_topic,
            }],
        ),

        # 5) MuJoCo controller
        # If use_cbf=true, controller consumes safe qdes topic.
        # If use_cbf=false, controller consumes nominal qdes topic directly.
        Node(
            package="mujoco_g1",
            executable="g1_controller",
            name="g1_controller",
            output="screen",
            parameters=[{
                "mjcf_path": mjcf_path,
                "qdes_topic": qdes_safe_topic,
                "joint_state_topic": joint_state_topic,
            }],
            condition=IfCondition(use_cbf),
        ),

        Node(
            package="mujoco_g1",
            executable="g1_controller",
            name="g1_controller_nominal",
            output="screen",
            parameters=[{
                "mjcf_path": mjcf_path,
                "qdes_topic": qdes_nominal_topic,
                "joint_state_topic": joint_state_topic,
            }],
            condition=UnlessCondition(use_cbf),
        ),

        # 6) Ghost robot for nominal trajectory visualization
        GroupAction(
            condition=IfCondition(ghost),
            actions=[
                # Full robot_state_publisher for real robot
                Node(
                    package="robot_state_publisher",
                    executable="robot_state_publisher",
                    name="robot_state_publisher",
                    output="screen",
                    parameters=[{
                        "robot_description": Command(["cat ", urdf_file]),
                    }],
                    remappings=[
                        ("joint_states", joint_state_topic),
                    ],
                ),
                # Ghost robot publisher for nominal (unsafe) commands
                Node(
                    package="g1_cbf",
                    executable="ghost_publisher_node",
                    name="ghost_publisher_node",
                    output="screen",
                    parameters=[{
                        "joint_state_topic": joint_state_topic,
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
                        "robot_description": Command(["cat ", urdf_file]),
                        "frame_prefix": "ghost/",
                    }],
                    remappings=[
                        ("joint_states", ghost_joint_state_topic),
                    ],
                ),
            ],
        ),

        # 7) Optional RViz
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            condition=IfCondition(rviz),
            arguments=["-d", rviz_config],
        ),
    ])

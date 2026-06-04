from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    skeleton_points_filtered_topic = LaunchConfiguration("skeleton_points_filtered_topic")
    hand_finger_angles_topic = LaunchConfiguration("hand_finger_angles_topic")
    run_inspire_bridge = LaunchConfiguration("run_inspire_bridge")
    enable_inspire_motion = LaunchConfiguration("enable_inspire_motion")
    inspire_command_topic = LaunchConfiguration("inspire_command_topic")
    inspire_state_topic = LaunchConfiguration("inspire_state_topic")
    inspire_controlled_side = LaunchConfiguration("inspire_controlled_side")
    zed_output_layout = LaunchConfiguration("zed_output_layout")

    return LaunchDescription([
        DeclareLaunchArgument(
            "skeleton_points_filtered_topic",
            default_value="/skeleton/points_filtered",
        ),
        DeclareLaunchArgument(
            "hand_finger_angles_topic",
            default_value="/hand_finger_angles",
        ),
        DeclareLaunchArgument(
            "zed_output_layout",
            default_value="finger10",
        ),
        DeclareLaunchArgument(
            "run_inspire_bridge",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "enable_inspire_motion",
            default_value="true",
        ),
        DeclareLaunchArgument(
            "inspire_command_topic",
            default_value="/inspire/cmd",
        ),
        DeclareLaunchArgument(
            "inspire_state_topic",
            default_value="/inspire/state",
        ),
        DeclareLaunchArgument(
            "inspire_controlled_side",
            default_value="both",
        ),
        Node(
            package="mujoco_g1",
            executable="zed_skeleton_points_preprocessor",
            name="zed_skeleton_points_preprocessor",
            output="screen",
            parameters=[{
                "input_points_topic": "/skeleton/points",
                "input_conf_topic": "/skeleton/confidence",
                "output_points_topic": skeleton_points_filtered_topic,
                "min_confidence": 60,
                "point_ema_alpha": 0.30,
                "point_max_jump": 0.6,
                "point_max_reject_count": 3,
            }],
        ),

        Node(
            package="mujoco_g1",
            executable="zed_hand_finger_angles",
            name="zed_hand_finger_angles",
            output="screen",
            parameters=[{
                "input_points_topic": skeleton_points_filtered_topic,
                "input_conf_topic": "/skeleton/confidence",
                "output_topic": hand_finger_angles_topic,
                "output_layout": zed_output_layout,
                "min_confidence": 40,
                "timeout_state": 1.0,
                "hold_last_on_timeout": True,
                "debug_log": True,
            }],
        ),

        Node(
            package="real_g1",
            executable="inspire_hand_bridge",
            name="inspire_hand_bridge",
            output="screen",
            condition=IfCondition(run_inspire_bridge),
            parameters=[{
                "finger_angles_topic": hand_finger_angles_topic,
                "command_topic": inspire_command_topic,
                "state_topic": inspire_state_topic,
                "controlled_side": inspire_controlled_side,
                "enable_motion": ParameterValue(enable_inspire_motion, value_type=bool),
                "input_layout": zed_output_layout,
                "publish_both_hands": True,
                "debug_log": True,
            }],
        ),
    ])

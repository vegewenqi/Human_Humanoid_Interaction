from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    finger_angles_topic = LaunchConfiguration("finger_angles_topic")
    command_topic = LaunchConfiguration("command_topic")
    state_topic = LaunchConfiguration("state_topic")
    controlled_side = LaunchConfiguration("controlled_side")
    enable_motion = LaunchConfiguration("enable_motion")
    input_layout = LaunchConfiguration("input_layout")

    return LaunchDescription([
        DeclareLaunchArgument(
            "finger_angles_topic",
            default_value="/hand_finger_angles",
        ),
        DeclareLaunchArgument(
            "command_topic",
            default_value="/inspire/cmd",
        ),
        DeclareLaunchArgument(
            "state_topic",
            default_value="/inspire/state",
        ),
        DeclareLaunchArgument(
            "controlled_side",
            default_value="right",
        ),
        DeclareLaunchArgument(
            "enable_motion",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "input_layout",
            default_value="auto",
        ),
        Node(
            package="real_g1",
            executable="inspire_hand_bridge",
            name="inspire_hand_bridge",
            output="screen",
            parameters=[{
                "finger_angles_topic": finger_angles_topic,
                "command_topic": command_topic,
                "state_topic": state_topic,
                "controlled_side": controlled_side,
                "enable_motion": ParameterValue(enable_motion, value_type=bool),
                "input_layout": input_layout,
                "publish_both_hands": True,
                "debug_log": True,
            }],
        ),
    ])

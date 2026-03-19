from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mujoco_g1',
            executable='human_angle_estimator',
            name='human_angle_estimator',
            output='screen'
        ),
        Node(
            package='mujoco_g1',
            executable='g1_joint_mapper',
            name='g1_joint_mapper',
            output='screen'
        ),
        Node(
            package='mujoco_g1',
            executable='g1_controller',
            name='g1_controller',
            output='screen'
        ),
    ])
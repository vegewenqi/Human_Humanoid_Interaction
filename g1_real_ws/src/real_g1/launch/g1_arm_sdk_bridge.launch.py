from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='real_g1',
            executable='g1_arm_sdk_bridge',
            name='g1_arm_sdk_bridge',
            output='screen',
            parameters=[{
                'network_interface': 'enx98ded0145852',
                'qdes_topic': '/g1_upperbody_q_des',
                'qdes_in_degrees': False,
                'control_dt': 0.02,
                'ema_alpha': 0.20,
                'max_joint_velocity': 0.50,
                'topic_timeout_sec': 0.30,
                'kp_arm': 60.0,
                'kd_arm': 1.5,
                'kp_waist': 40.0,
                'kd_waist': 1.5,
                'dq': 0.0,
                'tau_ff': 0.0,
                'weight_active': 1.0,
                'weight_release_rate': 0.2,
                'shutdown_release_sec': 2.0,
                'q_home_6': [0.0, 0.0, 0.0, 1.5708, 0.0, 1.5708],
                'q_min_6': [-0.52, -0.52, 0.0, -1.0472, -2.2515, -1.0472],
                'q_max_6': [0.52, 0.52, 2.2515, 2.0944, 0.0, 2.0944],
            }]
        )
    ])
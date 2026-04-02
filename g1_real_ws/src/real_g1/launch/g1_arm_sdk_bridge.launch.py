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

                'qdes_topic': '/g1_upperbody_q_des_safe',
                'joint_state_topic': '/joint_states',
                'qdes_in_degrees': False,

                'control_dt': 0.02,
                'ema_alpha': 0.2,
                'max_joint_velocity': 0.5,
                'home_transition_velocity': 0.05,
                'home_hold_sec': 5.0,
                'track_entry_blend_sec': 3.0,
                'shutdown_return_velocity': 0.05,
                'shutdown_hold_sec': 1.5,
                'weight_acquire_rate': 0.15,
                'weight_release_rate': 0.10,
                'topic_timeout_sec': 0.30,

                'kp_arm': 60.0,
                'kd_arm': 1.5,
                'kp_waist': 35.0,
                'kd_waist': 1.5,
                'dq': 0.0,
                'tau_ff': 0.0,
                'weight_active': 1.0,
                'shutdown_release_sec': 2.0,

                # 8-DoF input order:
                # [waist_roll, waist_pitch, l_sh_pitch, l_sh_roll, l_elbow, r_sh_pitch, r_sh_roll, r_elbow]
                'q_home_8': [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708],
                'q_min_8': [-0.52, -0.52, -3.0892, -1.5882, -1.0472, -3.0892, -2.2515, -1.0472],
                'q_max_8': [0.52, 0.52,  2.6704,  2.2515,  2.0944,  2.6704,  1.5882,  2.0944],
            }]
        )
    ])

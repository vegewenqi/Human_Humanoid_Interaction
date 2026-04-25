from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    urdf_path = '/ws/src/g1_cbf_ros2/g1_cbf/g1_29dof.urdf'

    return LaunchDescription([
        Node(
            package='real_g1',
            executable='g1_arm_sdk_bridge',
            name='g1_arm_sdk_bridge_calib',
            output='screen',
            parameters=[{
                'qdes_topic': '/calib_upperbody_q_des',
                'joint_state_topic': '/joint_states_calib',
                'qdes_in_degrees': False,

                "control_dt": 0.01,
                "ema_alpha": 0.25,
                "max_joint_velocity": 0.7,
                "home_transition_velocity": 0.20,
                "shutdown_return_velocity": 0.20,
                "weight_acquire_rate": 0.30,
                "weight_release_rate": 0.30,

                "kp_arm": 50.0,
                "kd_arm": 2.0,
                "kp_waist": 100.0,
                "kd_waist": 10.0,
                "dq": 0.0,
                "tau_ff": 0.0,
                "weight_active": 1.0,
                "shutdown_release_sec": 2.0,

                "enable_waist_balance_offset": False,

                "q_home_8": [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 1.5708],
                "q_min_8": [-0.52, -0.52, -3.0892, -1.5882, -1.0472, -3.0892, -2.2515, -1.0472],
                "q_max_8": [0.52, 0.52,  2.6704,  2.2515,  2.0944,  2.6704,  1.5882,  2.0944],
                
                "debug_log": False,
                "debug_log_period_sec": 1.0,
            }],
        ),

        Node(
            package='g1_cbf',
            executable='calibration_sampler_node',
            name='calibration_sampler_node',
            output='screen',
            parameters=[{
                'urdf_path': urdf_path,
                'tag_topic': '/tag_center_zed_world',
                'lowstate_topic': '/lowstate',
                'qdes_topic': '/calib_upperbody_q_des',

                'csv_path': '/ws/calibration/g1_tag_calibration_samples.csv',
                'result_path': '/ws/calibration/g1_tag_extrinsic_result.json',

                'tag_frame': 'torso_link',
                'tag_offset_x': 0.08,
                'tag_offset_y': 0.0,
                'tag_offset_z': 0.125,

                'settle_sec': 2.5,
                'sample_sec': 0.8,
                'roll_amp': 0.08,
                'pitch_amp': 0.08,
            }],
        ),
    ])
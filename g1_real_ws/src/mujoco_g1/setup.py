from setuptools import setup
from glob import glob
import os

package_name = 'mujoco_g1'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wenqi Cai',
    maintainer_email='wenqicai.97@gmail.com',
    description='MuJoCo G1 controller (no IK)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'skeleton_listener = mujoco_g1.skeleton_listener:main',
            'human_angle_estimator = mujoco_g1.human_angle_estimator:main',
            'g1_joint_mapper = mujoco_g1.g1_joint_mapper:main',
            'g1_controller = mujoco_g1.g1_controller:main',
            'jointstate_to_array_qdes = mujoco_g1.jointstate_to_array_qdes:main',
        ],
    },
)
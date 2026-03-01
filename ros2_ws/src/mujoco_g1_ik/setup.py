from setuptools import setup

package_name = 'mujoco_g1_ik'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Wenqi Cai',
    maintainer_email='wenqicai.97@gmail.com',
    description='MuJoCo G1 IK controller (ROS2 subscriber; mink)',
    license='MIT',
    entry_points={
        'console_scripts': [
            'skeleton_listener = mujoco_g1_ik.skeleton_listener:main',
            'g1_ik_controller = mujoco_g1_ik.g1_ik_controller:main',
        ],
    },
)

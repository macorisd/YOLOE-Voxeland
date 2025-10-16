from setuptools import setup

package_name = 'yoloe_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Macorís Decena Giménez',
    maintainer_email='macorisd@gmail.com',
    description='ROS2 package for YOLOE',
    license='MIT',
    entry_points={
        'console_scripts': [
            'yoloe_node = yoloe_ros2.yoloe_ros2_node:main',
        ],
    },
)

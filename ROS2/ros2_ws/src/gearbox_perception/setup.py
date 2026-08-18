from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'gearbox_perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/perception_pipeline.launch.py']),
        (os.path.join('share', package_name, 'resources'),
            [f for f in glob('resource/*') if f != 'resource/gearbox_perception']),
        (os.path.join('share', package_name, 'utils'), glob('gearbox_perception/utils/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_bridge = gearbox_perception.nodes.camera_bridge_node:main',
            'gearbox_registration = gearbox_perception.nodes.gearbox_registration_node:main',
        ],
    },
)

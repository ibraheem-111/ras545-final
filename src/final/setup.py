from setuptools import find_packages, setup

package_name = 'final'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ibraheem',
    maintainer_email='ibraheem@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_publisher = final.image_publisher:main',
            'image_subscriber = final.image_subscriber:main',
            'object_finder = final.object_finder:main',
            'robot_control = final.robot_control:main',
            'orchestrator = final.orchestrator:main',
        ],
    },
)

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    eye_node = Node(
        package='xarm6_intercept',
        executable='eye',
        name='eye_node',
        output='screen',
    )

    brain_node = Node(
        package='xarm6_intercept',
        executable='brain',
        name='brain',
        output='screen',
    )

    arm_node = Node(
        package='xarm6_intercept',
        executable='arm',
        name='arm',
        output='screen',
    )

    return LaunchDescription([
        eye_node,
        brain_node,
        arm_node,
    ])

import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml
from launch.actions import IncludeLaunchDescription, ExecuteProcess,RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.event_handlers import OnProcessExit
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Define the 'ig_lio' package directory
    ig_lio_dir = get_package_share_directory('ig_lio')
    ig_lio_reloc_dir = get_package_share_directory('ig_lio_relocalization')
    
    # Define the path to your parameter file
    param_path = os.path.join(ig_lio_dir, 'config', 'livox.yaml')
    reloc_param_path = os.path.join(ig_lio_reloc_dir, 'params', 'reloc.yaml')

    return LaunchDescription([
        Node(
            package='ig_lio_relocalization',
            executable='ig_lio_relocalization_node',
            name='ig_lio_relocalization_node',
            output='screen',
            parameters=[reloc_param_path],  # Pass the parameter file path directly
        ),
        Node(
            package='ig_lio_relocalization',
            executable='ig_lio_tf_fusion_node',
            name='ig_lio_tf_fusion_node',
            output='screen',
            # parameters=[param_path],  # Pass the parameter file path directly
        ),
        Node(
            package='ig_lio',
            executable='ig_lio_node',
            name='ig_lio_node',
            output='screen',
            parameters=[param_path],  # Pass the parameter file path directly
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz',
            arguments=['-d', os.path.join(ig_lio_reloc_dir, 'rviz', 'reloc_show.rviz')],
            output='screen'
        ),
    ])

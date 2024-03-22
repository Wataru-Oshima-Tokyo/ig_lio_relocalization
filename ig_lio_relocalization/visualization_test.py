import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d

class PointCloudVisualizerNode(Node):
    def __init__(self):
        super().__init__('point_cloud_visualizer_node')
        # Create an instance of the Visualizer class
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("ROS 2 Random Point Cloud Visualization", width=800, height=600)
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3) - 0.5)
        self.vis.add_geometry(self.pcd)
        
        # Create a timer to update the point cloud every 0.1 seconds (10 Hz)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        # Generate a random point cloud
        self.pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3) - 0.5)
        
        # Update the visualizer
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

def main(args=None):
    rclpy.init(args=args)
    visualizer_node = PointCloudVisualizerNode()
    
    # Using try-except to properly handle shutdown
    try:
        rclpy.spin(visualizer_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        visualizer_node.vis.destroy_window()
        visualizer_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

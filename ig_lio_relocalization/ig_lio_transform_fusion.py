#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry
import numpy as np
import tf_transformations
from tf2_ros import TransformBroadcaster
import threading
import time
from .library.color_print import COLOR_PRINT
import copy 
from geometry_msgs.msg import TransformStamped
class TransformFusionNode(Node):
    def __init__(self):
        super().__init__('transform_fusion_node')
        self.declare_parameter('freq_pub_localization', 50)
        self.freq_pub_localization = self.get_parameter('freq_pub_localization').value

        self.cur_odom_to_baselink = None
        self.cur_map_to_odom = None
        self.color_print = COLOR_PRINT(self)
        self.sub_odom = self.create_subscription(
            Odometry,
            '/lio_odom',
            self.cb_save_cur_odom,
            10)


        self.sub_map_to_odom = self.create_subscription(
            Odometry,
            '/map_to_odom',
            self.cb_save_map_to_odom,
            10)

        self.pub_localization = self.create_publisher(
            Odometry,
            '/localization',
            10)

        self.br = TransformBroadcaster(self)

        threading.Thread(target=self.transform_fusion).start()

    def pose_to_mat(self, pose_msg):
        position = pose_msg.pose.pose.position
        orientation = pose_msg.pose.pose.orientation
        return np.matmul(
            tf_transformations.translation_matrix((position.x, position.y, position.z)),
            tf_transformations.quaternion_matrix((orientation.x, orientation.y, orientation.z, orientation.w)),
        )

    def transform_fusion(self):
        while rclpy.ok():
            time.sleep(1 / self.freq_pub_localization)
            cur_odom = copy.copy(self.cur_odom_to_baselink)
            if self.cur_map_to_odom is not None:
                T_map_to_odom = self.pose_to_mat(self.cur_map_to_odom)
            else:
                T_map_to_odom = np.eye(4)

            # Transformation broadcasting
            # Note: Update to use tf2_ros for ROS2
            # This section needs to be adapted based on your TF broadcasting needs in ROS2
            if cur_odom is not None:
                self.broadcast_transform(T_map_to_odom)
                # Publish global localization odometry
                # Adapt this section to your needs for publishing the localization information

    def rotation_matrix_to_quaternion(self, rotation_matrix):
        quaternion = tf_transformations.quaternion_from_matrix(rotation_matrix)
        return quaternion

    def broadcast_transform(self, transformation_matrix):
        t = TransformStamped()

        # Fill header information
        t.header.stamp = self.cur_odom_to_baselink.header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        
        # Extract translation from the transformation matrix
        t.transform.translation.x = transformation_matrix[0, 3]
        t.transform.translation.y = transformation_matrix[1, 3]
        t.transform.translation.z = transformation_matrix[2, 3]
        
        # Convert the rotation matrix to a quaternion
        # self.color_print.print_in_green("Transformation is: ")
        # self.color_print.print_in_green(transformation_matrix)
        q = self.rotation_matrix_to_quaternion(transformation_matrix)
        # self.color_print.print_in_green("Done transforming q")
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        # Broadcast the transformation
        self.br.sendTransform(t)
        # self.get_logger().info(f'Broadcasted map to odom transform with confidence {confidence}')

    def cb_save_cur_odom(self, odom_msg):
        self.cur_odom_to_baselink = odom_msg

    def cb_save_map_to_odom(self, odom_msg):
        self.cur_map_to_odom = odom_msg

def main(args=None):
    rclpy.init(args=args)
    transform_fusion_node = TransformFusionNode()
    rclpy.spin(transform_fusion_node)
    transform_fusion_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

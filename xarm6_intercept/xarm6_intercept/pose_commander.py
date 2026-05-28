#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from xarm_msgs.srv import PlanPose, PlanExec


class XArmPoseCommander(Node):
    def __init__(self):
        super().__init__('xarm_pose_commander')

        self.pose_client = self.create_client(PlanPose, '/xarm_pose_plan')
        self.exec_client = self.create_client(PlanExec, '/xarm_exec_plan')

        self.sub = self.create_subscription(
            Pose,
            '/target_pose',
            self.pose_callback,
            10
        )

        self.get_logger().info('Waiting for xArm planner services...')
        self.pose_client.wait_for_service()
        self.exec_client.wait_for_service()
        self.get_logger().info('Ready. Publish geometry_msgs/Pose to /target_pose')

    def pose_callback(self, pose: Pose):
        self.get_logger().info(
            f'Received pose: x={pose.position.x}, y={pose.position.y}, z={pose.position.z}'
        )

        req = PlanPose.Request()
        req.target = pose

        future = self.pose_client.call_async(req)
        future.add_done_callback(self.plan_done)

    def plan_done(self, future):
        result = future.result()
        if result is None or not result.success:
            self.get_logger().error('Pose planning failed')
            return

        self.get_logger().info('Planning succeeded. Executing...')

        exec_req = PlanExec.Request()
        exec_req.wait = True

        exec_future = self.exec_client.call_async(exec_req)
        exec_future.add_done_callback(self.exec_done)

    def exec_done(self, future):
        result = future.result()
        self.get_logger().info(f'Execution result: {result}')

        if result is None or not result.success:
            self.get_logger().error('Execution failed')
        else:
            self.get_logger().info('Execution finished')


def main():
    rclpy.init()
    node = XArmPoseCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

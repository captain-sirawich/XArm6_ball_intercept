#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from xarm_msgs.srv import PlanJoint, PlanExec


TARGET = [
    -1.570802092552185,
    0.5235917568206787,
    -0.52359938621521,
    -3.1416022777557373,
    1.5707982778549194,
    -0.08727966994047165,
]


class ResetInitialPositionStepwise(Node):
    def __init__(self):
        super().__init__('reset_initial_position_stepwise')

        self.current_joints = None
        self.step = 0

        self.joint_client = self.create_client(PlanJoint, '/xarm_joint_plan')
        self.exec_client = self.create_client(PlanExec, '/xarm_exec_plan')

        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.get_logger().info('Waiting for planner services...')
        self.joint_client.wait_for_service()
        self.exec_client.wait_for_service()

        self.get_logger().info('Waiting for current joint state...')

    def joint_state_callback(self, msg):
        if self.current_joints is not None:
            return

        joint_map = dict(zip(msg.name, msg.position))

        try:
            self.current_joints = [
                joint_map['joint1'],
                joint_map['joint2'],
                joint_map['joint3'],
                joint_map['joint4'],
                joint_map['joint5'],
                joint_map['joint6'],
            ]
        except KeyError:
            self.get_logger().error(f'Joint names not found: {msg.name}')
            return

        self.get_logger().info(f'Current joints: {self.current_joints}')

        first_target = self.current_joints.copy()
        first_target[0] = TARGET[0]

        self.get_logger().info('Step 1: moving joint1 only')
        self.plan_joint_target(first_target)

    def plan_joint_target(self, joints):
        req = PlanJoint.Request()
        req.target = joints

        future = self.joint_client.call_async(req)
        future.add_done_callback(self.plan_done)

    def plan_done(self, future):
        result = future.result()
        self.get_logger().info(f'Planning result: {result}')

        if result is None or not result.success:
            self.get_logger().error('Planning failed')
            rclpy.shutdown()
            return

        exec_req = PlanExec.Request()
        exec_req.wait = True

        future = self.exec_client.call_async(exec_req)
        future.add_done_callback(self.exec_done)

    def exec_done(self, future):
        result = future.result()
        self.get_logger().info(f'Execution result: {result}')

        if result is None or not result.success:
            self.get_logger().error('Execution failed')
            rclpy.shutdown()
            return

        if self.step == 0:
            self.step = 1
            self.get_logger().info('Step 1 complete. Step 2: moving remaining joints')
            self.plan_joint_target(TARGET)
        else:
            self.get_logger().info('Reset complete')
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ResetInitialPositionStepwise()
    rclpy.spin(node)


if __name__ == '__main__':
    main()

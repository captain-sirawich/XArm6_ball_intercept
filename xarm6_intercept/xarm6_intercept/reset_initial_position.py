#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from xarm_msgs.srv import PlanJoint, PlanExec


TARGET = [
    # -1.570802092552185,
    # 0.5235917568206787,
    # -0.52359938621521,
    # -3.1416022777557373,
    # 1.5707982778549194,
    # -0.08727966994047165,
    -1.5708001852035522,
    0.34906306862831116,
    -0.6108618378639221,
    -3.1415984630584717,
    1.134453535079956,
    -0.08727200329303741

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
        # We only need to catch the joint state once to initialize the sequence
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

        # ---------------------------------------------------------
        # STEP 1: Set Joint 3 to -1.500 (Retract elbow)
        # ---------------------------------------------------------
        self.get_logger().info('Step 1: Setting joint 3 to -1.500 to avoid collision')
        
        target_step1 = self.current_joints.copy()
        target_step1[2] = -1.500  # Index 2 corresponds to Joint 3
        
        # Save this position state so Step 2 builds off of it
        self.current_joints = target_step1.copy()

        self.plan_joint_target(target_step1)

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

        # ---------------------------------------------------------
        # State Machine for Step Execution
        # ---------------------------------------------------------
        if self.step == 0:
            self.step = 1
            self.get_logger().info('Step 1 complete. Step 2: Moving only joint 1 to target')
            
            # STEP 2: Move Joint 1 to TARGET[0], keeping Joint 3 retracted
            target_step2 = self.current_joints.copy()
            target_step2[0] = TARGET[0] # Index 0 corresponds to Joint 1
            
            # Save this position state so Step 3 builds off of it natively
            self.current_joints = target_step2.copy()
            
            self.plan_joint_target(target_step2)
            
        elif self.step == 1:
            self.step = 2
            self.get_logger().info('Step 2 complete. Step 3: Moving remaining joints to target')
            
            # STEP 3: Move all joints to the final TARGET pose
            self.plan_joint_target(TARGET)
            
        else:
            self.get_logger().info('Reset complete. Arm safely reached initial position.')
            rclpy.shutdown()


def main():
    rclpy.init()
    node = ResetInitialPositionStepwise()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
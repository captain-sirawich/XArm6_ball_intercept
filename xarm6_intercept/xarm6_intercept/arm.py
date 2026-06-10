#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32
from xarm_msgs.msg import MoveVelocity
from xarm_msgs.srv import SetInt16, SetInt16ById


class XArmVelocityDriver(Node):
    def __init__(self):
        super().__init__('xarm_velocity_driver')

        self.motion_enable_client = self.create_client(
            SetInt16ById,
            '/xarm/motion_enable'
        )
        self.set_mode_client = self.create_client(
            SetInt16,
            '/xarm/set_mode'
        )
        self.set_state_client = self.create_client(
            SetInt16,
            '/xarm/set_state'
        )

        self.get_logger().info('Waiting for xArm services...')
        self.wait_for_service(self.motion_enable_client, '/xarm/motion_enable')
        self.wait_for_service(self.set_mode_client, '/xarm/set_mode')
        self.wait_for_service(self.set_state_client, '/xarm/set_state')

        self.vel_pub = self.create_publisher(
            MoveVelocity,
            '/xarm/vc_set_cartesian_velocity',
            1
        )

        self.vel_sub = self.create_subscription(
            Float32,
            '/xarm/x_velocity_cmd',
            self.velocity_callback,
            1
        )

        self.enable_velocity_mode()

        self.get_logger().info(
            'Velocity driver ready. Listening to /xarm/x_velocity_cmd'
        )

    def wait_for_service(self, client, name):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(f'Waiting for service: {name}')

    def call_service(self, client, request, name, timeout=5.0):
        self.get_logger().info(f'Calling {name}...')
        future = client.call_async(request)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=timeout
        )

        if future.done():
            result = future.result()
            self.get_logger().info(f'{name} response: {result}')
            return result

        self.get_logger().error(f'{name} timed out')
        return None

    def enable_velocity_mode(self):
        self.get_logger().info('Enabling xArm velocity mode...')

        req1 = SetInt16ById.Request()
        req1.id = 8
        req1.data = 1
        self.call_service(
            self.motion_enable_client,
            req1,
            '/xarm/motion_enable'
        )

        req2 = SetInt16.Request()
        req2.data = 5
        self.call_service(
            self.set_mode_client,
            req2,
            '/xarm/set_mode'
        )

        req3 = SetInt16.Request()
        req3.data = 0
        self.call_service(
            self.set_state_client,
            req3,
            '/xarm/set_state'
        )

        self.get_logger().info('xArm velocity mode setup finished')

    def velocity_callback(self, msg: Float32):
        vx_m_s = msg.data
        vx_mm_s = vx_m_s * 1000.0

        arm_msg = MoveVelocity()

        # [vx, vy, vz, rx, ry, rz]
        arm_msg.speeds = [
            float(vx_mm_s),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0
        ]

        arm_msg.is_tool_coord = False
        arm_msg.duration = 0.08

        self.vel_pub.publish(arm_msg)

        self.get_logger().info(
            f'Sent xArm velocity: {vx_mm_s:.1f} mm/s'
        )

    def stop(self):
        arm_msg = MoveVelocity()
        arm_msg.speeds = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        arm_msg.is_tool_coord = False
        arm_msg.duration = 0.05
        self.vel_pub.publish(arm_msg)

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = XArmVelocityDriver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping velocity driver...')
        node.stop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

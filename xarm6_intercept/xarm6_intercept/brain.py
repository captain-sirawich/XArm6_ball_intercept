#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener


class XArmXBrain(Node):
    def __init__(self):
        super().__init__('xarm_x_brain')

        self.ball_sub = self.create_subscription(
            PointStamped,
            '/eye/ball_global',
            self.ball_callback,
            1
        )

        self.vel_pub = self.create_publisher(
            Float32,
            '/xarm/x_velocity_cmd',
            1
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_frame = 'link_base'
        self.tcp_frame = 'link6'

        self.latest_target_x = None
        self.last_ball_time = None

        self.fixed_speed_m_s = 1.0
        self.tolerance = 0.03
        self.dt = 0.07
        self.target_timeout = 0.5

        self.min_x = -0.4
        self.max_x = 0.4
        self.max_velocity = 1.0

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info('Brain ready. Listening to /eye/ball_global.')

    def ball_callback(self, msg: PointStamped):
        target_x = msg.point.x

        if target_x < self.min_x or target_x > self.max_x:
            self.get_logger().warn(
                f'Ignored ball target_x={target_x:.4f}. '
                f'Allowed range: [{self.min_x:.4f}, {self.max_x:.4f}]'
            )
            return

        self.latest_target_x = target_x
        self.last_ball_time = self.get_clock().now()

    def get_current_x(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                rclpy.time.Time()
            )
            return tf.transform.translation.x
        except Exception as e:
            self.get_logger().warn(f'Cannot get current X: {e}')
            return None

    def ball_target_is_stale(self):
        if self.last_ball_time is None:
            return True

        age = self.get_clock().now() - self.last_ball_time
        return age.nanoseconds / 1e9 > self.target_timeout

    def control_loop(self):
        if self.latest_target_x is None or self.ball_target_is_stale():
            self.publish_velocity(0.0)
            return

        current_x = self.get_current_x()

        if current_x is None:
            self.publish_velocity(0.0)
            return

        error = self.latest_target_x - current_x

        if abs(error) <= self.tolerance:
            vx_m_s = 0.0
        else:
            # direction = 1.0 if error > 0 else -1.0
            # vx_m_s = direction * self.fixed_speed_m_s * math.sqrt(abs(error))
            vx_m_s = self.fixed_speed_m_s*error


        vx_m_s = max(-self.max_velocity, min(self.max_velocity, vx_m_s))

        self.publish_velocity(vx_m_s)

        self.get_logger().info(
            f'current_x={current_x:.4f}, '
            f'ball_x={self.latest_target_x:.4f}, '
            f'error={error:.4f}, '
            f'cmd_vx={vx_m_s * 1000:.1f} mm/s'
        )

    def publish_velocity(self, vx_m_s):
        msg = Float32()
        msg.data = float(vx_m_s)
        self.vel_pub.publish(msg)

    def destroy_node(self):
        self.publish_velocity(0.0)
        super().destroy_node()


def main():
    rclpy.init()
    node = XArmXBrain()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping brain node...')
        node.publish_velocity(0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

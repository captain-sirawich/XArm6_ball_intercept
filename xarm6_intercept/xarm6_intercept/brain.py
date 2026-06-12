#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from xarm_msgs.srv import SetInt16


class XArmXBrain(Node):
    def __init__(self):
        super().__init__('xarm_x_brain')

        self.ball_sub = self.create_subscription(
            PointStamped,
            '/eye/ball_global',
            self.ball_callback,
            1
        )

        self.gripper_sub = self.create_subscription(
            PointStamped,
            '/eye/gripper_global',
            self.gripper_callback,
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

        self.latest_ball_global_x = None
        self.last_ball_time = None

        self.x_offset = None

        self.fixed_speed_m_s = 2.0
        self.tolerance = 0.03
        self.dt = 0.03
        self.target_timeout = 1.0

        self.min_x = -0.4
        self.max_x = 0.4
        self.max_velocity = self.fixed_speed_m_s

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            'Brain ready. Calibrating global X to local arm X.'
        )

    def ball_callback(self, msg: PointStamped):
        self.latest_ball_global_x = msg.point.x
        self.last_ball_time = self.get_clock().now()

    def gripper_callback(self, msg: PointStamped):
        if self.x_offset is not None:
            return

        current_local_x = self.get_current_x()

        if current_local_x is None:
            self.get_logger().warn(
                'Cannot calibrate yet because local arm X is unavailable.'
            )
            return

        gripper_global_x = msg.point.x

        self.x_offset = current_local_x - gripper_global_x

        self.get_logger().info(
            f'X calibration complete: '
            f'local_x={current_local_x:.4f}, '
            f'gripper_global_x={gripper_global_x:.4f}, '
            f'x_offset={self.x_offset:.4f}'
        )

    def get_current_x(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                rclpy.time.Time()
            )
            return tf.transform.translation.x
        except Exception as e:
            self.get_logger().warn(f'Cannot get current local X: {e}')
            return None

    def ball_target_is_stale(self):
        if self.last_ball_time is None:
            return True

        age = self.get_clock().now() - self.last_ball_time
        return age.nanoseconds / 1e9 > self.target_timeout

    def control_loop(self):
        if self.x_offset is None:
            self.publish_velocity(0.0)
            self.get_logger().warn('Waiting for X calibration...')
            return

        if self.latest_ball_global_x is None or self.ball_target_is_stale():
            self.publish_velocity(0.0)
            return

        current_local_x = self.get_current_x()

        if current_local_x is None:
            self.publish_velocity(0.0)
            return

        target_local_x = self.latest_ball_global_x + self.x_offset

        if target_local_x < self.min_x or target_local_x > self.max_x:
            self.get_logger().warn(
                f'Ignored target_local_x={target_local_x:.4f}. '
                f'Allowed range: [{self.min_x:.4f}, {self.max_x:.4f}]'
            )
            self.publish_velocity(0.0)
            return

        error = target_local_x - current_local_x

        if abs(error) <= self.tolerance:
            vx_m_s = 0.0
            # self.snap_stop_arm()
        else:
            direction = 1.0 if error > 0 else -1.0
            vx_m_s = direction * self.fixed_speed_m_s

        vx_m_s = max(-self.max_velocity, min(self.max_velocity, vx_m_s))

        self.publish_velocity(vx_m_s)

        self.get_logger().info(
            f'current_local_x={current_local_x:.4f}, '
            f'ball_global_x={self.latest_ball_global_x:.4f}, '
            f'target_local_x={target_local_x:.4f}, '
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

    # def snap_stop_arm(self):
    #     """Instantly halts the arm at the hardware level."""
    #     self.get_logger().warn('INITIATING SNAP STOP!')
        
    #     # Create the request object
    #     state_req = SetInt16.Request()
    #     state_req.data = 4  # 4 is the xArm code for "Stop"
        
    #     # Call the service asynchronously
    #     self.state_client.call_async(state_req)


def main(args=None):
    rclpy.init(args=args)
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

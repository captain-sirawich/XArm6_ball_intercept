#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32, Empty
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

        # --- NEW: Reset Command Subscriber ---
        self.reset_sub = self.create_subscription(
            Empty,
            '/xarm/reset_home',
            self.reset_callback,
            1
        )

        self.vel_pub = self.create_publisher(
            Float32,
            '/xarm/x_velocity_cmd',
            1
        )

        self.state_client = self.create_client(
            SetInt16,
            '/xarm/set_state'
        )

        while not self.state_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for /xarm/set_state service...')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.base_frame = 'link_base'
        self.tcp_frame = 'link6'

        self.latest_ball_global_x = None
        self.last_ball_time = None
        self.x_offset = None

        self.fixed_speed_m_s = 10.0
        self.tolerance = 0.03
        self.dt = 0.05
        self.target_timeout = 1.0

        self.min_x = -0.4
        self.max_x = 0.4
        self.max_velocity = self.fixed_speed_m_s

        self.was_moving = False
        self.snap_stop_enabled = True
        
        # --- NEW: Reset State Variables ---
        self.returning_home = False
        self.home_local_x = 0.0  # Set this to your exact initial local X coordinate

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            'Brain ready. Calibrating global X to local arm X.'
        )

    # --- NEW: Reset Callback ---
    def reset_callback(self, msg: Empty):
        self.get_logger().info('Reset command received! Taking over control to return home.')
        self.returning_home = True

    def ball_callback(self, msg: PointStamped):
        self.latest_ball_global_x = msg.point.x
        self.last_ball_time = self.get_clock().now()

        # If a new ball target comes in, cancel the return home sequence
        if self.returning_home:
            self.get_logger().info('New ball detected! Canceling reset.')
            self.returning_home = False

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
            self.stop_command()
            self.get_logger().warn('Waiting for X calibration...')
            return

        current_local_x = self.get_current_x()

        if current_local_x is None:
            self.stop_command()
            return

        # --- NEW: Determine Target based on State ---
        if self.returning_home:
            target_local_x = self.home_local_x
        else:
            if self.latest_ball_global_x is None or self.ball_target_is_stale():
                self.stop_command()
                return
            target_local_x = self.latest_ball_global_x + self.x_offset

        if target_local_x < self.min_x or target_local_x > self.max_x:
            self.get_logger().warn(
                f'Ignored target_local_x={target_local_x:.4f}. '
                f'Allowed range: [{self.min_x:.4f}, {self.max_x:.4f}]'
            )
            self.stop_command()
            # If home is somehow out of bounds, cancel the sequence
            if self.returning_home:
                self.returning_home = False
            return

        error = target_local_x - current_local_x

        if abs(error) <= self.tolerance:
            self.stop_command()

            # --- NEW: Check if we just arrived home ---
            if self.returning_home:
                self.get_logger().info('Successfully returned to initial position.')
                self.returning_home = False
            else:
                self.get_logger().info(
                    f'Target reached. '
                    f'current_local_x={current_local_x:.4f}, '
                    f'target_local_x={target_local_x:.4f}, '
                    f'error={error:.4f}'
                )
            return

        direction = 1.0 if error > 0 else -1.0
        vx_m_s = direction * self.fixed_speed_m_s

        vx_m_s = max(
            -self.max_velocity,
            min(self.max_velocity, vx_m_s)
        )

        self.publish_velocity(vx_m_s)
        self.was_moving = True

        state_msg = 'RETURNING HOME' if self.returning_home else 'TRACKING BALL'
        self.get_logger().info(
            f'[{state_msg}] current_local_x={current_local_x:.4f}, '
            f'target_local_x={target_local_x:.4f}, '
            f'error={error:.4f}, '
            f'cmd_vx={vx_m_s * 1000:.1f} mm/s'
        )

    def stop_command(self):
        self.publish_velocity(0.0)

        if self.was_moving:
            self.was_moving = False

            if self.snap_stop_enabled:
                self.snap_stop_arm()

    def publish_velocity(self, vx_m_s):
        msg = Float32()
        msg.data = float(vx_m_s)
        self.vel_pub.publish(msg)

    def snap_stop_arm(self):
        self.get_logger().warn('INITIATING SNAP STOP!')

        if not self.state_client.service_is_ready():
            self.get_logger().warn('/xarm/set_state service is not ready.')
            return

        state_req = SetInt16.Request()
        state_req.data = 4

        self.state_client.call_async(state_req)

        state_req = SetInt16.Request()
        state_req.data = 0

        self.state_client.call_async(state_req)

    def destroy_node(self):
        self.publish_velocity(0.0)
        super().destroy_node()


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

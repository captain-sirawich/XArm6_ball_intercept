#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from collections import deque
from geometry_msgs.msg import PointStamped, Twist
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

        # Upgraded to Twist for multi-axis velocity
        self.vel_pub = self.create_publisher(
            Twist,
            '/xarm/velocity_cmd',
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

        # --- PREDICTION VARIABLES ---
        self.ball_history = deque(maxlen=15) 
        self.last_ball_time = None
        self.intercept_y = 0.0
        self.lookback_time = 0.2
        # ----------------------------

        # --- SOFT CATCH VARIABLES ---
        self.home_y = None          # Captured during calibration
        self.y_reach_trigger = 0.5  # Ball distance to reach out
        self.y_catch_trigger = 0.1  # Ball distance to retract
        self.fixed_vy_m_s = 0.05    # Safe Y-axis speed
        self.max_y_travel = 0.05    # +/- 5cm bounding box
        # ----------------------------

        self.x_offset = None

        self.fixed_speed_m_s = 10.0
        self.tolerance = 0.03
        self.dt = 0.07
        self.target_timeout = 1.0

        self.min_x = -0.4
        self.max_x = 0.4
        self.max_velocity = self.fixed_speed_m_s

        self.was_moving = False
        self.snap_stop_enabled = True

        self.timer = self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            'Brain ready. Calibrating global X and caching home Y...'
        )

    def ball_callback(self, msg: PointStamped):
        self.last_ball_time = self.get_clock().now()
        cam_time = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
        self.ball_history.append((msg.point.x, msg.point.y, cam_time))

    def gripper_callback(self, msg: PointStamped):
        if self.x_offset is not None:
            return

        coords = self.get_current_xy()

        if coords[0] is None:
            self.get_logger().warn(
                'Cannot calibrate yet because local arm TF is unavailable.'
            )
            return

        current_local_x, current_local_y = coords

        # Calibrate X offset
        gripper_global_x = msg.point.x
        self.x_offset = current_local_x - gripper_global_x
        
        # Cache the current Y as the neutral home line
        self.home_y = current_local_y

        self.get_logger().info(
            f'Calibration complete: '
            f'local_x={current_local_x:.4f}, '
            f'x_offset={self.x_offset:.4f}, '
            f'home_y={self.home_y:.4f}'
        )

    def get_current_xy(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.tcp_frame,
                rclpy.time.Time()
            )
            return tf.transform.translation.x, tf.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f'Cannot get current local transform: {e}')
            return None, None

    def ball_target_is_stale(self):
        if self.last_ball_time is None:
            return True

        age = self.get_clock().now() - self.last_ball_time
        return age.nanoseconds / 1e9 > self.target_timeout

    def get_predicted_x(self):
        if len(self.ball_history) < 2:
            return self.ball_history[0][0] if self.ball_history else None

        current_x, current_y, current_time = self.ball_history[-1]

        old_x, old_y, old_time = self.ball_history[0]
        for x, y, t in reversed(self.ball_history):
            if current_time - t >= self.lookback_time:
                old_x, old_y, old_time = x, y, t
                break

        dt = current_time - old_time
        
        if dt <= 0.01:
            return current_x

        vx = (current_x - old_x) / dt
        vy = (current_y - old_y) / dt

        if abs(vy) < 0.01:
            return current_x

        time_to_intercept = (self.intercept_y - current_y) / vy

        if time_to_intercept < 0 or time_to_intercept > 5.0:
            return current_x

        predicted_x = current_x + (vx * time_to_intercept)
        return predicted_x

    def control_loop(self):
        if self.x_offset is None or self.home_y is None:
            self.stop_command()
            self.get_logger().warn('Waiting for calibration...')
            return

        if not self.ball_history or self.ball_target_is_stale():
            self.stop_command()
            return

        coords = self.get_current_xy()
        if coords[0] is None:
            self.stop_command()
            return
            
        current_local_x, current_local_y = coords
        current_ball_y = self.ball_history[-1][1]

        # ----------------------------------------------------
        # 1. LATERAL X-AXIS PREDICTION LOGIC
        # ----------------------------------------------------
        predicted_global_x = self.get_predicted_x()
        
        if predicted_global_x is None:
            self.stop_command()
            return

        target_local_x = predicted_global_x + self.x_offset
        error = target_local_x - current_local_x
        vx_m_s = 0.0

        if self.min_x <= target_local_x <= self.max_x:
            if abs(error) > self.tolerance:
                direction = 1.0 if error > 0 else -1.0
                vx_m_s = direction * self.fixed_speed_m_s
                vx_m_s = max(-self.max_velocity, min(self.max_velocity, vx_m_s))

        # ----------------------------------------------------
        # 2. DEPTH Y-AXIS SOFT CATCH LOGIC
        # ----------------------------------------------------
        vy_m_s = 0.0
        
        if current_ball_y > self.y_reach_trigger:
            # Reaching out: move forward, hard stop at +5cm limit
            if current_local_y < (self.home_y + self.max_y_travel):
                vy_m_s = self.fixed_vy_m_s
                
        elif current_ball_y < self.y_catch_trigger:
            # Retracting: pull back, hard stop at -5cm limit
            if current_local_y > (self.home_y - self.max_y_travel):
                vy_m_s = -self.fixed_vy_m_s

        # ----------------------------------------------------
        # 3. COMMAND EXECUTION
        # ----------------------------------------------------
        if vx_m_s == 0.0 and vy_m_s == 0.0:
            self.stop_command()
        else:
            self.publish_velocity(vx_m_s, vy_m_s)
            self.was_moving = True

            self.get_logger().info(
                f'CMD | X: {vx_m_s*1000:.1f} mm/s (err={error:.4f}) | '
                f'Y: {vy_m_s*1000:.1f} mm/s (ball_y={current_ball_y:.2f})'
            )

    def stop_command(self):
        self.publish_velocity(0.0, 0.0)

        if self.was_moving:
            self.was_moving = False
            if self.snap_stop_enabled:
                self.snap_stop_arm()

    def publish_velocity(self, vx_m_s, vy_m_s):
        msg = Twist()
        msg.linear.x = float(vx_m_s)
        msg.linear.y = float(vy_m_s)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.vel_pub.publish(msg)

    def snap_stop_arm(self):
        self.get_logger().warn('INITIATING SNAP STOP!')

        if not self.state_client.service_is_ready():
            return

        state_req = SetInt16.Request()
        state_req.data = 4
        self.state_client.call_async(state_req)

        state_req = SetInt16.Request()
        state_req.data = 0
        self.state_client.call_async(state_req)

    def destroy_node(self):
        self.publish_velocity(0.0, 0.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = XArmXBrain()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping brain node...')
        node.publish_velocity(0.0, 0.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
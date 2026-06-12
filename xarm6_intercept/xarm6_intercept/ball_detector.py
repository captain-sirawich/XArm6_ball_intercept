#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool


class EyeNode(Node):
    def __init__(self):
        super().__init__('eye_node')

        self.declare_parameter('rgb_topic', '/camera/rgb/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image')
        self.declare_parameter('camera_info_topic', '/camera/rgb/camera_info')
        self.declare_parameter('global_frame', 'eye_global')
        self.declare_parameter('debug_view', True)

        self.declare_parameter('h_min', 18)
        self.declare_parameter('h_max', 35)

        self.declare_parameter('s_min', 45)
        self.declare_parameter('v_min', 40)

        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_max', 255)

        self.declare_parameter('marker_h_min', 160)
        self.declare_parameter('marker_h_max', 179)
        self.declare_parameter('marker_s_min', 35)
        self.declare_parameter('marker_s_max', 255)
        self.declare_parameter('marker_v_min', 40)
        self.declare_parameter('marker_v_max', 255)

        self.declare_parameter('min_ball_area', 250.0)
        self.declare_parameter('min_marker_area', 150.0)
        self.declare_parameter('blur_kernel', 5)
        self.declare_parameter('morph_kernel', 9)

        self.bridge = CvBridge()

        self.latest_depth = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.initialized = False
        self.initial_gripper_camera = None

        self.global_frame = self.get_parameter('global_frame').value

        self.ball_pub = self.create_publisher(PointStamped, '/eye/ball_global', 10)
        self.gripper_pub = self.create_publisher(PointStamped, '/eye/gripper_global', 10)
        self.camera_pub = self.create_publisher(PointStamped, '/eye/camera_global', 10)

        self.ball_detected_pub = self.create_publisher(Bool, '/eye/ball_detected', 10)
        self.gripper_detected_pub = self.create_publisher(Bool, '/eye/gripper_detected', 10)
        self.initialized_pub = self.create_publisher(Bool, '/eye/initialized', 10)

        self.create_subscription(
            Image,
            self.get_parameter('rgb_topic').value,
            self.rgb_callback,
            10,
        )

        self.create_subscription(
            Image,
            self.get_parameter('depth_topic').value,
            self.depth_callback,
            10,
        )

        self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').value,
            self.camera_info_callback,
            10,
        )

        self.get_logger().info('Eye node started.')
        self.get_logger().info('Initial gripper position becomes global (0, 0, 0).')
        self.get_logger().info('Global XY axes are rotated 90 degrees anti-clockwise.')

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='passthrough'
            )
        except Exception as exc:
            self.get_logger().error(f'Failed to convert depth image: {exc}')

    def rgb_callback(self, msg: Image):
        if self.latest_depth is None or self.fx is None:
            self.get_logger().warn('Waiting for depth image and camera info...')
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert RGB image: {exc}')
            return

        ball = self.detect_ball(frame)
        gripper = self.detect_marker(frame)

        self.ball_detected_pub.publish(Bool(data=ball is not None))
        self.gripper_detected_pub.publish(Bool(data=gripper is not None))
        self.initialized_pub.publish(Bool(data=self.initialized))

        debug_frame = frame.copy()

        if gripper is not None:
            gx, gy, gsize = gripper
            gripper_camera = self.pixel_to_3d(gx, gy)

            if gripper_camera is not None:
                if not self.initialized:
                    self.initial_gripper_camera = gripper_camera.copy()
                    self.initialized = True

                    self.get_logger().info(
                        'Initialized eye global frame. '
                        'Initial gripper is now global origin: '
                        '(0.000, 0.000, 0.000)'
                    )

                    self.get_logger().info(
                        f'Initial gripper in camera frame: '
                        f'x={gripper_camera[0]:.3f}, '
                        f'y={gripper_camera[1]:.3f}, '
                        f'z={gripper_camera[2]:.3f}'
                    )

                gripper_global_raw = gripper_camera - self.initial_gripper_camera
                gripper_global = self.rotate_xy_90_ccw(gripper_global_raw)

                self.publish_point(self.gripper_pub, gripper_global, msg.header.stamp)

                cv2.circle(debug_frame, (gx, gy), 5, (255, 0, 255), -1)
                cv2.putText(
                    debug_frame,
                    f'Gripper G=({gripper_global[0]:.2f},'
                    f'{gripper_global[1]:.2f},'
                    f'{gripper_global[2]:.2f})m',
                    (max(0, gx - 100), max(20, gy - 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 255),
                    2,
                )

        if self.initialized:
            camera_global_raw = -self.initial_gripper_camera
            camera_global = self.rotate_xy_90_ccw(camera_global_raw)
            self.publish_point(self.camera_pub, camera_global, msg.header.stamp)

        if ball is not None and self.initialized:
            bx, by, radius = ball
            ball_camera = self.pixel_to_3d(bx, by)

            if ball_camera is not None:
                ball_global_raw = ball_camera - self.initial_gripper_camera
                ball_global = self.rotate_xy_90_ccw(ball_global_raw)

                self.publish_point(self.ball_pub, ball_global, msg.header.stamp)

                cv2.circle(debug_frame, (bx, by), radius, (0, 255, 0), 2)
                cv2.circle(debug_frame, (bx, by), 4, (0, 0, 255), -1)
                cv2.putText(
                    debug_frame,
                    f'Ball G=({ball_global[0]:.2f},'
                    f'{ball_global[1]:.2f},'
                    f'{ball_global[2]:.2f})m',
                    (max(0, bx - 100), max(20, by - radius - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        if self.get_parameter('debug_view').value:
            cv2.imshow('eye_node', debug_frame)
            cv2.waitKey(1)

    def rotate_xy_90_ccw(self, point):
        return np.array([
            -point[1],
            point[0],
            point[2],
        ], dtype=np.float64)

    def publish_point(self, publisher, xyz, stamp):
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.global_frame
        msg.point.x = float(xyz[0])
        msg.point.y = float(xyz[1])
        msg.point.z = float(xyz[2])
        publisher.publish(msg)

    def pixel_to_3d(self, u, v):
        h, w = self.latest_depth.shape[:2]

        if u < 0 or v < 0 or u >= w or v >= h:
            return None

        depth_value = self.latest_depth[v, u]

        if isinstance(depth_value, np.ndarray):
            depth_value = depth_value[0]

        z = float(depth_value)

        if self.latest_depth.dtype == np.uint16:
            z = z / 1000.0

        if z <= 0.0 or np.isnan(z) or np.isinf(z):
            z = self.search_valid_depth(u, v)

        if z is None:
            return None

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return np.array([x, y, z], dtype=np.float64)

    def search_valid_depth(self, u, v, window=5):
        h, w = self.latest_depth.shape[:2]
        values = []

        for dy in range(-window, window + 1):
            for dx in range(-window, window + 1):
                px = u + dx
                py = v + dy

                if px < 0 or py < 0 or px >= w or py >= h:
                    continue

                value = float(self.latest_depth[py, px])

                if self.latest_depth.dtype == np.uint16:
                    value = value / 1000.0

                if value > 0.0 and not np.isnan(value) and not np.isinf(value):
                    values.append(value)

        if not values:
            return None

        return float(np.median(values))

    def build_hsv_mask(self, frame, lower, upper):
        blur_kernel = int(self.get_parameter('blur_kernel').value)
        morph_kernel = int(self.get_parameter('morph_kernel').value)

        if blur_kernel % 2 == 0:
            blur_kernel += 1

        if morph_kernel % 2 == 0:
            morph_kernel += 1

        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def detect_ball(self, frame):
        lower = np.array([
            self.get_parameter('h_min').value,
            self.get_parameter('s_min').value,
            self.get_parameter('v_min').value,
        ])

        upper = np.array([
            self.get_parameter('h_max').value,
            self.get_parameter('s_max').value,
            self.get_parameter('v_max').value,
        ])

        mask = self.build_hsv_mask(frame, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        min_area = float(self.get_parameter('min_ball_area').value)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)

            if radius < 10:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0

            if perimeter > 0:
                circularity = 4.0 * np.pi * area / (perimeter * perimeter)

            if circularity < 0.4:
                continue

            score = area * circularity
            candidates.append((
                score,
                int(round(x)),
                int(round(y)),
                int(round(radius))
            ))

        if not candidates:
            return None

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        _, x, y, radius = candidates[0]

        return x, y, radius

    def detect_marker(self, frame):
        lower = np.array([
            self.get_parameter('marker_h_min').value,
            self.get_parameter('marker_s_min').value,
            self.get_parameter('marker_v_min').value,
        ])

        upper = np.array([
            self.get_parameter('marker_h_max').value,
            self.get_parameter('marker_s_max').value,
            self.get_parameter('marker_v_max').value,
        ])

        mask = self.build_hsv_mask(frame, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        min_area = float(self.get_parameter('min_marker_area').value)
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            rect = cv2.minAreaRect(contour)
            (x, y), (width, height), _ = rect

            if width < 8 or height < 8:
                continue

            size = int(round(max(width, height)))
            score = area

            candidates.append((
                score,
                int(round(x)),
                int(round(y)),
                size
            ))

        if not candidates:
            return None

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        _, x, y, size = candidates[0]

        return x, y, size


def main(args=None):
    rclpy.init(args=args)
    node = EyeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Pose2D
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        self.declare_parameter('image_topic', '/camera/rgb/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image')
        self.declare_parameter('camera_info_topic', '/camera/depth/camera_info')
        self.declare_parameter('ball_topic', '/ball/pose2d')
        self.declare_parameter('ball_point_topic', '/ball/point3d')
        self.declare_parameter('debug_view', True)
        self.declare_parameter('detect_mode', 'multi')
        self.declare_parameter('draw_all_detections', True)
        self.declare_parameter('publish_3d', True)
        self.declare_parameter('depth_window', 5)

        self.declare_parameter('h_min', 35)
        self.declare_parameter('s_min', 20)
        self.declare_parameter('v_min', 10)
        self.declare_parameter('h_max', 90)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_max', 255)
        self.declare_parameter('dark_v_max', 85)
        self.declare_parameter('background_threshold', 32.0)

        self.declare_parameter('min_radius', 20)
        self.declare_parameter('max_radius', 140)
        self.declare_parameter('min_area', 500.0)
        self.declare_parameter('min_circularity', 0.45)
        self.declare_parameter('min_fill_ratio', 0.35)
        self.declare_parameter('blur_kernel', 5)
        self.declare_parameter('morph_kernel', 9)
        self.declare_parameter('roi_x', 0)
        self.declare_parameter('roi_y', 0)
        self.declare_parameter('roi_width', 0)
        self.declare_parameter('roi_height', 0)

        image_topic = self.get_parameter('image_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        ball_topic = self.get_parameter('ball_topic').value
        ball_point_topic = self.get_parameter('ball_point_topic').value

        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_depth_header = None
        self.camera_info = None

        self.publisher = self.create_publisher(Pose2D, ball_topic, 10)
        self.point_publisher = self.create_publisher(PointStamped, ball_point_topic, 10)
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )
        self.depth_subscription = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10,
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.get_logger().info(f'Subscribing to {image_topic}')
        self.get_logger().info(f'Subscribing to {depth_topic}')
        self.get_logger().info(f'Subscribing to {camera_info_topic}')
        self.get_logger().info(f'Publishing ball center to {ball_topic}')
        self.get_logger().info(f'Publishing ball 3D point to {ball_point_topic}')

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_header = msg.header
        except Exception as exc:
            self.get_logger().error(f'Failed to convert depth image: {exc}')

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert ROS image: {exc}')
            return

        roi, roi_offset = self.crop_roi(frame)
        mask = self.build_mask(roi)
        detections = self.detect_balls(mask, roi_offset)

        debug_frame = frame.copy()
        self.draw_roi(debug_frame, roi_offset, roi.shape)
        if detections:
            best_circle = detections[0]
            x, y, radius = best_circle
            pose = Pose2D()
            pose.x = float(x)
            pose.y = float(y)
            pose.theta = float(radius)
            self.publisher.publish(pose)
            point = self.compute_3d_point(x, y, frame.shape)
            if point is not None:
                self.point_publisher.publish(point)

            circles_to_draw = detections if self.get_parameter('draw_all_detections').value else [best_circle]
            for draw_x, draw_y, draw_radius in circles_to_draw:
                cv2.circle(debug_frame, (draw_x, draw_y), draw_radius, (0, 255, 0), 2)
                cv2.circle(debug_frame, (draw_x, draw_y), 3, (0, 0, 255), -1)

            cv2.putText(
                debug_frame,
                f'x={x} y={y} r={radius}',
                (max(0, x - radius), max(20, y - radius - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if point is not None:
                cv2.putText(
                    debug_frame,
                    f'X={point.point.x:.3f} Y={point.point.y:.3f} Z={point.point.z:.3f}m',
                    (max(0, x - radius), max(45, y - radius - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        if self.get_parameter('debug_view').value:
            cv2.imshow('ball_detector', debug_frame)
            cv2.imshow('ball_detector_mask', mask)
            cv2.waitKey(1)

    def crop_roi(self, frame):
        height, width = frame.shape[:2]
        x = int(self.get_parameter('roi_x').value)
        y = int(self.get_parameter('roi_y').value)
        roi_width = int(self.get_parameter('roi_width').value)
        roi_height = int(self.get_parameter('roi_height').value)

        if roi_width <= 0:
            roi_width = width - x
        if roi_height <= 0:
            roi_height = height - y

        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        x2 = max(x + 1, min(x + roi_width, width))
        y2 = max(y + 1, min(y + roi_height, height))
        return frame[y:y2, x:x2], (x, y)

    def compute_3d_point(self, image_x, image_y, image_shape):
        if not self.get_parameter('publish_3d').value:
            return None
        if self.latest_depth is None or self.camera_info is None:
            return None

        depth = self.latest_depth
        depth_height, depth_width = depth.shape[:2]
        image_height, image_width = image_shape[:2]
        depth_x = int(round(image_x * depth_width / float(image_width)))
        depth_y = int(round(image_y * depth_height / float(image_height)))
        depth_x = max(0, min(depth_x, depth_width - 1))
        depth_y = max(0, min(depth_y, depth_height - 1))

        depth_m = self.sample_depth_meters(depth, depth_x, depth_y)
        if depth_m is None:
            return None

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]
        if fx == 0.0 or fy == 0.0:
            return None

        point = PointStamped()
        point.header = self.latest_depth_header
        point.point.z = float(depth_m)
        point.point.x = (float(depth_x) - cx) * point.point.z / fx
        point.point.y = (float(depth_y) - cy) * point.point.z / fy
        return point

    def sample_depth_meters(self, depth, x, y):
        window = int(self.get_parameter('depth_window').value)
        if window % 2 == 0:
            window += 1
        half = max(0, window // 2)
        y1 = max(0, y - half)
        y2 = min(depth.shape[0], y + half + 1)
        x1 = max(0, x - half)
        x2 = min(depth.shape[1], x + half + 1)

        patch = depth[y1:y2, x1:x2].astype(np.float32)
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None

        depth_value = float(np.median(valid))
        if depth.dtype == np.uint16:
            depth_value *= 0.001
        return depth_value

    @staticmethod
    def draw_roi(frame, offset, roi_shape):
        x, y = offset
        roi_height, roi_width = roi_shape[:2]
        cv2.rectangle(
            frame,
            (x, y),
            (x + roi_width - 1, y + roi_height - 1),
            (255, 180, 0),
            1,
        )

    def build_mask(self, frame):
        blur_kernel = self.get_parameter('blur_kernel').value
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        morph_kernel = self.get_parameter('morph_kernel').value
        if morph_kernel % 2 == 0:
            morph_kernel += 1

        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        detect_mode = self.get_parameter('detect_mode').value
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

        if detect_mode == 'hsv':
            mask = cv2.inRange(hsv, lower, upper)
        elif detect_mode == 'background':
            mask = self.build_background_mask(blurred)
        elif detect_mode == 'multi':
            mask = self.build_multi_color_mask(hsv)
        else:
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            colored_mask = cv2.inRange(
                saturation,
                int(self.get_parameter('s_min').value),
                int(self.get_parameter('s_max').value),
            )
            bright_enough = cv2.inRange(
                value,
                int(self.get_parameter('v_min').value),
                int(self.get_parameter('v_max').value),
            )
            dark_mask = cv2.inRange(
                value,
                0,
                int(self.get_parameter('dark_v_max').value),
            )
            mask = cv2.bitwise_or(cv2.bitwise_and(colored_mask, bright_enough), dark_mask)

        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return self.fill_mask_holes(mask)

    def build_background_mask(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        h, w = lab.shape[:2]
        border = max(8, min(h, w) // 20)
        samples = np.concatenate((
            lab[:border, :, :].reshape(-1, 3),
            lab[-border:, :, :].reshape(-1, 3),
            lab[:, :border, :].reshape(-1, 3),
            lab[:, -border:, :].reshape(-1, 3),
        ))
        background = np.median(samples, axis=0)
        delta = lab.astype(np.float32) - background.astype(np.float32)
        distance = np.sqrt(
            0.7 * delta[:, :, 0] * delta[:, :, 0]
            + delta[:, :, 1] * delta[:, :, 1]
            + delta[:, :, 2] * delta[:, :, 2]
        )
        threshold = float(self.get_parameter('background_threshold').value)
        return np.where(distance > threshold, 255, 0).astype(np.uint8)

    def build_multi_color_mask(self, hsv):
        s_min = int(self.get_parameter('s_min').value)
        v_min = int(self.get_parameter('v_min').value)
        dark_v_max = int(self.get_parameter('dark_v_max').value)

        ranges = [
            ((18, max(35, s_min), max(35, v_min)), (34, 255, 255)),   # yellow
            ((30, s_min, v_min), (95, 255, 255)),                     # green
            ((90, s_min, v_min), (140, 255, 255)),                    # blue
        ]

        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lower), np.array(upper)))

        dark_mask = cv2.inRange(hsv[:, :, 2], 0, dark_v_max)
        return cv2.bitwise_or(mask, dark_mask)

    @staticmethod
    def fill_mask_holes(mask):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        filled = np.zeros_like(mask)
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        return filled

    def detect_balls(self, mask, roi_offset):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return []

        min_area = float(self.get_parameter('min_area').value)
        min_radius = int(self.get_parameter('min_radius').value)
        max_radius = int(self.get_parameter('max_radius').value)
        min_circularity = float(self.get_parameter('min_circularity').value)
        min_fill_ratio = float(self.get_parameter('min_fill_ratio').value)

        valid_candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if radius < min_radius or radius > max_radius:
                continue

            circle_area = np.pi * radius * radius
            fill_ratio = area / circle_area if circle_area > 0.0 else 0.0
            if fill_ratio < min_fill_ratio:
                continue

            perimeter = cv2.arcLength(contour, True)
            circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0.0 else 0.0
            if circularity < min_circularity:
                continue

            score = area * (0.7 + circularity) * (0.7 + fill_ratio)
            valid_candidates.append((score, x, y, radius))

        if not valid_candidates:
            return []

        offset_x, offset_y = roi_offset
        valid_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [
            (int(round(x + offset_x)), int(round(y + offset_y)), int(round(radius)))
            for _, x, y, radius in valid_candidates
        ]


def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
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

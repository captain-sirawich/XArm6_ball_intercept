import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose2D
from rclpy.node import Node
from sensor_msgs.msg import Image


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector')

        self.declare_parameter('image_topic', '/camera/rgb/image_raw')
        self.declare_parameter('ball_topic', '/Ball_Pose2D')
        self.declare_parameter('marker_topic', '/Hand_Pose2D')
        self.declare_parameter('debug_view', True)
        self.declare_parameter('draw_all_detections', True)

        self.declare_parameter('h_min', 35)
        self.declare_parameter('s_min', 20)
        self.declare_parameter('v_min', 10)
        self.declare_parameter('h_max', 90)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_max', 255)

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

        self.declare_parameter('detect_marker', True)
        self.declare_parameter('marker_h_min', 160)
        self.declare_parameter('marker_h_max', 179)
        self.declare_parameter('marker_s_min', 35)
        self.declare_parameter('marker_s_max', 255)
        self.declare_parameter('marker_v_min', 40)
        self.declare_parameter('marker_v_max', 255)
        self.declare_parameter('marker_min_area', 350.0)
        self.declare_parameter('marker_min_width', 12)
        self.declare_parameter('marker_min_height', 12)
        self.declare_parameter('marker_min_fill_ratio', 0.45)

        image_topic = self.get_parameter('image_topic').value
        ball_topic = self.get_parameter('ball_topic').value
        marker_topic = self.get_parameter('marker_topic').value

        self.bridge = CvBridge()

        self.publisher = self.create_publisher(Pose2D, ball_topic, 10)
        self.marker_publisher = self.create_publisher(Pose2D, marker_topic, 10)
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(f'Subscribing to {image_topic}')
        self.get_logger().info(f'Publishing ball 2D pose to {ball_topic}')
        self.get_logger().info(f'Publishing hand marker 2D pose to {marker_topic}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'Failed to convert ROS image: {exc}')
            return

        roi, roi_offset = self.crop_roi(frame)
        mask = self.build_mask(roi)
        detections = self.detect_balls(mask, roi_offset)

        marker_mask = None
        marker_detections = []
        if self.get_parameter('detect_marker').value:
            marker_mask = self.build_marker_mask(roi)
            marker_detections = self.detect_markers(marker_mask, roi_offset)

        debug_view = self.get_parameter('debug_view').value
        debug_frame = frame.copy() if debug_view else None
        if debug_view:
            self.draw_roi(debug_frame, roi_offset, roi.shape)

        if detections:
            best_circle = detections[0]
            x, y, radius = best_circle
            pose = Pose2D()
            pose.x = float(x)
            pose.y = float(y)
            pose.theta = float(radius)
            self.publisher.publish(pose)

            if debug_view:
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

        if marker_detections:
            marker = marker_detections[0]
            mx, my, marker_size, marker_box = marker
            marker_pose = Pose2D()
            marker_pose.x = float(mx)
            marker_pose.y = float(my)
            marker_pose.theta = float(marker_size)
            self.marker_publisher.publish(marker_pose)

            if debug_view:
                cv2.drawContours(debug_frame, [marker_box], -1, (255, 0, 255), 2)
                cv2.circle(debug_frame, (mx, my), 3, (255, 0, 255), -1)
                cv2.putText(
                    debug_frame,
                    f'marker x={mx} y={my} s={marker_size}',
                    (max(0, mx - marker_size), max(20, my - marker_size - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        if debug_view:
            cv2.imshow('ball_detector', debug_frame)
            cv2.imshow('ball_detector_mask', mask)
            if marker_mask is not None:
                cv2.imshow('gripper_marker_mask', marker_mask)
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
        return self.build_hsv_mask(frame, lower, upper)

    def build_marker_mask(self, frame):
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
        return self.build_hsv_mask(frame, lower, upper)

    def build_hsv_mask(self, frame, lower, upper):
        blur_kernel = self.get_parameter('blur_kernel').value
        if blur_kernel % 2 == 0:
            blur_kernel += 1

        morph_kernel = self.get_parameter('morph_kernel').value
        if morph_kernel % 2 == 0:
            morph_kernel += 1

        blurred = cv2.GaussianBlur(frame, (blur_kernel, blur_kernel), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return self.fill_mask_holes(mask)

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

        min_area = float(self.get_parameter('min_area').value)
        min_radius = int(self.get_parameter('min_radius').value)
        max_radius = int(self.get_parameter('max_radius').value)
        min_circularity = float(self.get_parameter('min_circularity').value)
        min_fill_ratio = float(self.get_parameter('min_fill_ratio').value)

        valid_candidates = []
        offset_x, offset_y = roi_offset
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

        valid_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        valid_candidates = self.remove_duplicate_candidates(valid_candidates)
        return [
            (int(round(x + offset_x)), int(round(y + offset_y)), int(round(radius)))
            for _, x, y, radius in valid_candidates
        ]

    def detect_markers(self, mask, roi_offset):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        min_area = float(self.get_parameter('marker_min_area').value)
        min_width = int(self.get_parameter('marker_min_width').value)
        min_height = int(self.get_parameter('marker_min_height').value)
        min_fill_ratio = float(self.get_parameter('marker_min_fill_ratio').value)

        candidates = []
        offset_x, offset_y = roi_offset
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            rect = cv2.minAreaRect(contour)
            (x, y), (width, height), _ = rect
            if width < min_width or height < min_height:
                continue

            rect_area = width * height
            fill_ratio = area / rect_area if rect_area > 0.0 else 0.0
            if fill_ratio < min_fill_ratio:
                continue

            box = cv2.boxPoints(rect)
            box[:, 0] += offset_x
            box[:, 1] += offset_y
            box = np.round(box).astype(np.int32)
            marker_size = int(round(max(width, height) * 0.5))
            score = area * fill_ratio
            candidates.append((
                score,
                int(round(x + offset_x)),
                int(round(y + offset_y)),
                max(1, marker_size),
                box,
            ))

        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [
            (x, y, marker_size, box)
            for _, x, y, marker_size, box in candidates
        ]

    @staticmethod
    def remove_duplicate_candidates(candidates):
        filtered = []
        for candidate in candidates:
            _, x, y, radius = candidate
            duplicate = False
            for _, kept_x, kept_y, kept_radius in filtered:
                center_distance = np.hypot(x - kept_x, y - kept_y)
                if center_distance < min(radius, kept_radius) * 0.55:
                    duplicate = True
                    break
            if not duplicate:
                filtered.append(candidate)
        return filtered


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

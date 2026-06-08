#!/usr/bin/env python3
# 0507 detector node: dot center publish + inspection_state
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import time
import json

A3_W_MM = 420.0
A3_H_MM = 297.0
BUSY_FLAG_PATH = '/tmp/robot_auto_busy.flag'


class IntegratedDetector(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.bridge = CvBridge()
        self.model = YOLO('/media/wonseok/새 볼륨/project/0507_hard/FINAL.pt')

        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10
        )
        self.det_pub = self.create_publisher(String, '/detected_object', 10)
        self.inspection_pub = self.create_publisher(String, '/inspection_state', 10)

        self.clicked_pts = []
        self.H = None
        self.h_ready = False

        self.prev_cx = None
        self.prev_cy = None
        self.last_center_px = None
        self.stationary_start_time = None
        self.sent_flag = False

        self.smooth_alpha = 0.70
        self.stationary_pixel_threshold = 4.0
        self.stationary_hold_sec = 0.9
        self.move_reset_threshold = 12.0

        self.final_lock_pixel_threshold = 2.0
        self.final_lock_frames_required = 8
        self.final_lock_count = 0

        self.win_name = 'Real-time Inspection'
        cv2.namedWindow(self.win_name)
        cv2.setMouseCallback(self.win_name, self.on_mouse)

        self.get_logger().info(f'model classes: {self.model.names}')
        self.get_logger().info('시스템 시작: A3 모서리 4곳을 클릭하세요.')

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.h_ready:
            self.clicked_pts.append((x, y))
            self.get_logger().info(f'클릭: ({x}, {y}) - {len(self.clicked_pts)}/4')
            if len(self.clicked_pts) == 4:
                src = np.array(self.clicked_pts, dtype=np.float32)
                dst = np.array([
                    [0.0, 0.0],
                    [A3_W_MM, 0.0],
                    [A3_W_MM, A3_H_MM],
                    [0.0, A3_H_MM]
                ], dtype=np.float32)
                self.H = cv2.getPerspectiveTransform(src, dst)
                self.h_ready = True
                self.get_logger().info('★ mm 변환 준비 완료!')

    def pixel_to_mm(self, px, py):
        if not self.h_ready:
            return None, None
        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def is_inside_workspace_mm(self, x_mm, y_mm):
        return (0.0 <= x_mm <= A3_W_MM) and (0.0 <= y_mm <= A3_H_MM)

    def smooth_center(self, cx, cy):
        if self.prev_cx is None:
            self.prev_cx = float(cx)
            self.prev_cy = float(cy)
            return float(cx), float(cy)
        self.prev_cx = self.smooth_alpha * self.prev_cx + (1.0 - self.smooth_alpha) * float(cx)
        self.prev_cy = self.smooth_alpha * self.prev_cy + (1.0 - self.smooth_alpha) * float(cy)
        return self.prev_cx, self.prev_cy

    def is_stationary(self, cx, cy):
        now = time.time()
        if self.last_center_px is None:
            self.last_center_px = (cx, cy)
            self.stationary_start_time = now
            return False, 0.0

        dist = float(np.sqrt((cx - self.last_center_px[0]) ** 2 + (cy - self.last_center_px[1]) ** 2))

        if dist <= self.stationary_pixel_threshold:
            hold_time = now - self.stationary_start_time
            return (hold_time >= self.stationary_hold_sec), dist
        else:
            self.last_center_px = (cx, cy)
            self.stationary_start_time = now
            self.final_lock_count = 0
            return False, dist

    def reset_publish_state(self):
        self.sent_flag = False
        self.final_lock_count = 0

    def robot_is_busy(self):
        return os.path.exists(BUSY_FLAG_PATH)

    def publish_inspection_state(self, has_car_part=False, has_defect=False, defect_count=0):
        msg = String()
        if has_car_part and has_defect:
            judge = 'NG'
        elif has_car_part:
            judge = 'OK'
        else:
            judge = 'NO_OBJECT'

        payload = {
            'has_car_part': bool(has_car_part),
            'has_defect': bool(has_defect),
            'defect_count': int(defect_count),
            'judge': judge,
            'stamp': time.time(),
        }
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.inspection_pub.publish(msg)

    def listener_callback(self, data):
        frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')

        results = self.model(frame, conf=0.25 , iou=0.45, verbose=False)
        result = results[0]
        annotated_frame = result.plot(conf=False)

        boxes = result.boxes
        if len(boxes) == 0:
            self.publish_inspection_state(False, False, 0)
            self.reset_publish_state()
            cv2.imshow(self.win_name, annotated_frame)
            cv2.waitKey(1)
            return

        has_car_part = False
        has_dot = False
        has_defect = False
        defect_count = 0
        best_dot_box = None
        best_dot_conf = -1.0
        best_dot_label = None

        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[class_id]
            label_l = str(label).strip().lower()

            # 검사 상태용: 기존 car_part가 있으면 그대로 기록
            if label_l == 'car_part':
                has_car_part = True

            # 흡착 목표점: dot 중심을 사용
            if label_l == 'dot':
                has_dot = True
                if conf > best_dot_conf:
                    best_dot_conf = conf
                    best_dot_box = box
                    best_dot_label = label

            # 결함 판정은 기존과 동일하게 defect 누적 시간으로 판단
            if label_l == 'defect':
                has_defect = True
                defect_count += 1

        # /inspection_state는 다면 검사에서 defect 시간 누적용으로 사용한다.
        # 모델이 dot+defect 구조라 car_part가 없을 수도 있으므로 has_car_part에는 has_dot도 반영한다.
        self.publish_inspection_state(has_car_part or has_dot, has_defect, defect_count)

        judge = None
        if has_dot and has_defect:
            judge = 'NG'
        elif has_dot:
            judge = 'OK'

        if best_dot_box is None:
            self.reset_publish_state()
            cv2.putText(
                annotated_frame,
                f'NO DOT / defect:{defect_count}',
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
        else:
            x1, y1, x2, y2 = map(int, best_dot_box.xyxy[0].tolist())
            raw_cx, raw_cy = (x1 + x2) // 2, (y1 + y2) // 2
            smooth_cx, smooth_cy = self.smooth_center(raw_cx, raw_cy)

            draw_cx = int(round(smooth_cx))
            draw_cy = int(round(smooth_cy))
            cv2.circle(annotated_frame, (draw_cx, draw_cy), 4, (0, 0, 255), 1)

            if judge is not None:
                judge_color = (0, 255, 0) if judge == 'OK' else (0, 0, 255)
                cv2.putText(
                    annotated_frame,
                    f'judge:{judge}',
                    (x1, max(35, y1 - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    judge_color,
                    2
                )

            if self.h_ready:
                mx, my = self.pixel_to_mm(smooth_cx, smooth_cy)
                stationary_ok, move_dist = self.is_stationary(smooth_cx, smooth_cy)
                lock_dist = float(np.sqrt((raw_cx - smooth_cx) ** 2 + (raw_cy - smooth_cy) ** 2))
                inside_workspace = self.is_inside_workspace_mm(mx, my)
                busy = self.robot_is_busy()

                if move_dist > self.move_reset_threshold:
                    self.reset_publish_state()

                if stationary_ok and lock_dist <= self.final_lock_pixel_threshold and inside_workspace and not busy:
                    self.final_lock_count += 1
                else:
                    self.final_lock_count = 0

                publish_ready = (
                    stationary_ok
                    and self.final_lock_count >= self.final_lock_frames_required
                    and not self.sent_flag
                    and judge in {'OK', 'NG'}
                    and inside_workspace
                    and not busy
                )

                if publish_ready:
                    msg = String()
                    msg.data = f'{best_dot_label},{mx:.1f},{my:.1f},0.0,0,{judge}'
                    self.det_pub.publish(msg)
                    self.get_logger().info(f'★ 안정화된 dot 좌표 전송: {msg.data}')
                    self.sent_flag = True

                if busy:
                    status_text = 'ROBOT BUSY'
                    txt_color = (255, 0, 255)
                elif not inside_workspace:
                    status_text = 'OUTSIDE WORKSPACE'
                    txt_color = (0, 0, 255)
                elif self.sent_flag:
                    status_text = 'PUBLISHED'
                    txt_color = (255, 200, 0)
                elif stationary_ok:
                    status_text = 'LOCKING'
                    txt_color = (0, 255, 0)
                else:
                    status_text = 'STABILIZING'
                    txt_color = (0, 255, 255)

                cv2.putText(
                    annotated_frame,
                    f'{mx:.1f}, {my:.1f} mm',
                    (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    txt_color,
                    2
                )
                if status_text == 'OUTSIDE WORKSPACE':
                    cv2.putText(
                        annotated_frame,
                        status_text,
                        (x1, y2 + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        txt_color,
                        2
                    )

        for pt in self.clicked_pts:
            cv2.circle(annotated_frame, pt, 6, (0, 255, 255), -1)

        cv2.imshow(self.win_name, annotated_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

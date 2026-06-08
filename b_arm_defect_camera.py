import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class BArmDefectCamera(Node):
    def __init__(self):
        super().__init__('b_arm_defect_camera')

        self.declare_parameter(
            'model_path',
            '/media/wonseok/새 볼륨/yolo_project/runs/board_plastic/weights/FINAL.pt'
        )
        self.declare_parameter('image_topic', '/b_camera/image_raw')
        self.declare_parameter('state_topic', '/b_inspection_state')
        self.declare_parameter('window_name', 'B Arm Defect Camera')
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.45)

        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.state_topic = self.get_parameter('state_topic').value
        self.window_name = self.get_parameter('window_name').value
        self.conf = float(self.get_parameter('conf').value)
        self.iou = float(self.get_parameter('iou').value)

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.listener_callback,
            10
        )

        self.state_pub = self.create_publisher(String, self.state_topic, 10)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        self.get_logger().info(f'B arm defect camera started')
        self.get_logger().info(f'image_topic: {self.image_topic}')
        self.get_logger().info(f'state_topic: {self.state_topic}')
        self.get_logger().info(f'model_path: {self.model_path}')
        self.get_logger().info(f'model classes: {self.model.names}')

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        results = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)
        result = results[0]

        boxes = result.boxes

        has_car_part = False
        has_defect = False
        defect_count = 0

        view = frame.copy()

        for box in boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = self.model.names[class_id]

            label_lower = label.lower()

            if label_lower == 'car_part':
                has_car_part = True
                continue

            if label_lower == 'defect':
                has_defect = True
                defect_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                cv2.rectangle(
                    view,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    view,
                    f'defect {conf:.2f}',
                    (x1, max(25, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        if has_car_part and has_defect:
            judge = 'NG'
        elif has_car_part:
            judge = 'OK'
        elif has_defect:
            judge = 'NG'
        else:
            judge = 'NO_OBJECT'

        state = {
            'camera': 'b_arm',
            'has_car_part': has_car_part,
            'has_defect': has_defect,
            'defect_count': defect_count,
            'judge': judge
        }

        out_msg = String()
        out_msg.data = json.dumps(state, ensure_ascii=False)
        self.state_pub.publish(out_msg)

        cv2.imshow(self.window_name, view)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = BArmDefectCamera()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

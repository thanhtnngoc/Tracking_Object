#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import cv2
from cv_bridge import CvBridge
from threading import Thread

class VideoStream:
    """USB camera stream in separate thread."""
    def __init__(self, cam_index=0, resolution=(640,480)):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera!")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

class YOLOPublisher(Node):
    """Publish YOLO bbox and cropped ROI as sensor_msgs/Image."""
    def __init__(self):
        super().__init__('yolo_publisher')

        self.stream = VideoStream()
        self.model = YOLO("yolov8n.pt")
        self.bridge = CvBridge()

        # ROS2 publisher (QoS default supports intra-process)
        self.crop_pub = self.create_publisher(Image, '/yolo_crop_image', 10)

        # Timer ~30ms
        self.create_timer(0.03, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.stream.read()
        if not ret or frame is None:
            return

        # YOLO detect only person (class 0)
        results = self.model(frame, classes=[0], conf=0.6, verbose=False)
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            idx = int(confs.argmax())
            x1, y1, x2, y2 = map(int, xyxy[idx])

            # Crop ROI
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                try:
                    msg = self.bridge.cv2_to_imgmsg(roi, "bgr8")
                    self.crop_pub.publish(msg)
                except Exception as e:
                    self.get_logger().error(f"CvBridge error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLOPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stream.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import cv2
import math
from ultralytics import YOLO
from threading import Thread
import supervision as sv  # ⚡ Bộ tracking ổn định (ByteTrack)


class VideoStream_UsbCam:
    """Luồng video USB chạy trên thread riêng."""
    def __init__(self, cam_index=0, resolution=(640, 480)):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Không mở được USB camera!")

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


class ObjectTracker:
    """Quản lý tracking ID ổn định qua nhiều frame."""
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections):
        """Nhận các detection mới từ YOLO và trả về detection có ID ổn định."""
        return self.tracker.update_with_detections(detections)


class YOLOPersonScanner(Node):
    """ROS2 Node phát hiện người + tracking ổn định."""

    def __init__(self):
        super().__init__('yolo_person_scanner')
        self.get_logger().info("🧠 YOLOv8 + ByteTrack Person Scanner node started")

        # --- Mở camera ---
        self.stream = VideoStream_UsbCam()

        # --- Nạp model YOLO ---
        self.model = YOLO("yolov8n.pt")  # model nhỏ gọn, detect người

        # --- Tracker ---
        self.tracker = ObjectTracker()

        # --- Publisher ---
        self.publisher = self.create_publisher(Int32MultiArray, '/yolo_person_result', 10)

        # --- Timer 30ms ---
        self.timer = self.create_timer(0.03, self.timer_callback)

    # ------------------------
    # Hàm hiển thị (tách riêng)
    # ------------------------
    def display_frame(self, frame):
        """Hiển thị khung hình (tách riêng ra cho dễ tái sử dụng)."""
        cv2.imshow("YOLOv8 + ByteTrack", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            rclpy.shutdown()

    def timer_callback(self):
        ret, frame = self.stream.read()
        if not ret or frame is None:
            print("No frame received from camera")
            return

        h, w, _ = frame.shape
        cx_frame = w // 2
        cy_frame = h // 2

        # --- YOLO detect ---
        results = self.model(frame, classes=[0], conf=0.6, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # --- Cập nhật tracker ---
        tracked = self.tracker.update(detections)

        annotated_frame = frame.copy()
        cv2.circle(annotated_frame, (cx_frame, cy_frame), 6, (255, 0, 0), -1)  # tâm khung hình

        # --- Vẽ bounding box có ID ---
        for xyxy, track_id in zip(tracked.xyxy, tracked.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"ID {int(track_id)}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)

        # --- Tìm người gần tâm nhất ---
        distances = []
        for xyxy, track_id in zip(tracked.xyxy, tracked.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
            dx = cx - cx_frame
            dy = cy - cy_frame
            dist = math.sqrt(dx**2 + dy**2)
            distances.append((dist, dx, dy, int(track_id), (cx, cy)))

        if distances:
            dist_val, dx, dy, track_id, (cx, cy) = min(distances, key=lambda x: x[0])
            cv2.line(annotated_frame, (cx_frame, cy_frame), (cx, cy), (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Closest ID={track_id} dx={dx}, dy={dy}",
                        (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            msg = Int32MultiArray()
            msg.data = [dx, dy, track_id]
            self.publisher.publish(msg)

        # --- Hiển thị ---
        self.display_frame(annotated_frame)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOPersonScanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stream.stop()
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

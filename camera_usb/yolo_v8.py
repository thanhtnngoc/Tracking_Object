#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
import cv2
import math
from ultralytics import YOLO
from threading import Thread
import sys


class VideoStream_UsbCam:
    """USB video stream running in a separate thread."""
    def __init__(self, cam_index=0, resolution=(640, 480)):
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Failed to open USB camera!")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        """Continuously update frames in the background."""
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        """Return the latest frame."""
        return self.ret, self.frame

    def stop(self):
        """Stop the video stream and release the camera."""
        self.stopped = True
        self.cap.release()


class YOLOPersonScanner(Node):
    """ROS2 node for person detection using YOLOv8 (built-in bounding boxes)."""

    def __init__(self, show_display=False):
        super().__init__('yolo_person_scanner')
        self.get_logger().info("🧠 YOLOv8 Person Scanner node started")

        self.show_display = show_display

        # --- Initialize camera stream ---
        self.stream = VideoStream_UsbCam()

        # --- Load YOLOv8 model ---
        self.model = YOLO("yolov8n.pt")  # Small YOLO model

        # --- ROS2 Publisher ---
        self.publisher = self.create_publisher(Int32MultiArray, '/yolo_person_result', 10)

        # --- Timer callback every 30 ms ---
        self.timer = self.create_timer(0.03, self.timer_callback)

    def display_frame(self, frame):
        """Display frame only if enabled."""
        if not self.show_display:
            return  # 🚫 Bỏ qua nếu tắt hiển thị
        cv2.imshow("YOLOv8 Person Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            self.get_logger().info("🛑 ESC pressed — shutting down gracefully...")
            self.stream.stop()                 # 🧹 Giải phóng camera
            cv2.destroyAllWindows()    # 🧹 Đóng cửa sổ
            rclpy.shutdown()           # 🧹 Dừng ROS2 node
            sys.exit(0)
            return    

    def timer_callback(self):
        """Main callback for detection and publishing results."""
        ret, frame = self.stream.read()
        if not ret or frame is None:
            print("No frame received from camera")
            return
        h, w, _ = frame.shape
        cx_frame = w // 2
        cy_frame = h // 2

        # --- Run YOLO inference (detect only 'person') ---
        results = self.model(frame, classes=[0], conf=0.6, verbose=False)
        boxes = results[0].boxes

        # --- Lấy khung hình có bounding box, label, confidence do YOLO vẽ ---
        annotated_frame = results[0].plot()
        cv2.circle(annotated_frame, (cx_frame, cy_frame), 6, (255, 0, 0), -1)  # Draw frame center

        # --- Tính dx, dy giữa người gần trung tâm nhất ---
        distances = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                dx = cx - cx_frame
                dy = cy - cy_frame
                dist = math.sqrt(dx**2 + dy**2)
                distances.append((dist, dx, dy, (cx, cy)))

            # --- Lấy person gần trung tâm nhất ---
            dist_val, dx, dy, (cx, cy) = min(distances, key=lambda x: x[0])
            if self.show_display:
                cv2.circle(annotated_frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.line(annotated_frame, (cx_frame, cy_frame), (cx, cy), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"dx={dx}, dy={dy}", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # --- Publish dx, dy ---
            msg = Int32MultiArray()
            msg.data = [dx, dy]
            self.publisher.publish(msg)
            self.get_logger().info(f"dx = {dx}, dy = {dy}")
        # else:
        #     cv2.putText(annotated_frame, "No person detected", (20, 40),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # --- Display YOLO-rendered frame ---
        self.display_frame(annotated_frame)


def main(args=None):
    """Initialize and spin the ROS2 node."""
    rclpy.init(args=args)
    node = YOLOPersonScanner(show_display=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stream.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

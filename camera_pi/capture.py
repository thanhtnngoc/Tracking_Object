import cv2
import numpy as np
import os
import time
from threading import Thread
from picamera2 import Picamera2, Preview

class VideoStream:
    def __init__(self):
        self.picam2 = Picamera2()
        self.picam2.configure(
            self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (1640, 1232)}
            )
        )
        self.picam2.start()
        self.frame = self.picam2.capture_array()
        self.stopped = False
        Thread(target=self.update, args=(), daemon=True).start()

    def update(self):
        while not self.stopped:
            self.frame = self.picam2.capture_array()

    def read(self):
        return True, self.frame

    def stop(self):
        self.stopped = True
        self.picam2.stop()

class ImageCapture:
    def __init__(self, stream: VideoStream, save_dir, size, interval, max_img, mode):
        self.stream = stream
        self.save_dir = save_dir
        self.size = size
        self.interval = interval
        self.max_img = max_img
        self.mode = mode
        self.stopped = False
        self.i = 0

        os.makedirs(save_dir, exist_ok=True)

    def start(self):
        Thread(target=self.capture_loop, daemon=True).start()
        print("📸 Bắt đầu lưu ảnh...")
    
    def capture_loop(self):
        last_time = time.time()
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                print("Không đọc được frame.")
                continue
            frame = cv2.resize(frame, self.size)
            current_time = time.time()
            if current_time - last_time >= self.interval:
                filename = os.path.join(self.save_dir, f"img_{self.i:03d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Đã lưu {filename}")
                self.i += 1
                last_time = current_time

                if self.mode == "1":
                    if self.show_window: 
                        cv2.imshow("Live Capture", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:  # ESC
                            print("Đóng cửa sổ hiển thị (vẫn chụp bình thường).")
                            self.show_window = False

                    if self.max_img is not None and self.i >= self.max_img:
                        print("Đã chụp đủ số ảnh.")
                        self.stop()
                        break
    
        cv2.destroyAllWindows()

    def stop(self):
        self.stopped = True
        print("Dừng lưu ảnh.")

if __name__ == "__main__":
    auto = True

    if not auto:
        mode = input("1: Calib, 2: Capture: ").strip()
    else:
        mode = "2"

    if mode == "1":
        max = 30
        interval = 1.0
        save_dir = "/home/thanh/ros2_ws/src/p_detect_object/images_calib"
    else:
        max = None
        interval = 0.2
        save_dir = "/home/thanh/ros2_ws/src/p_detect_object/images_object"

    stream = VideoStream()
    w, h = 800, 600
    capture = ImageCapture(stream, save_dir, size=(w,h), interval=interval, max_img=max, mode=mode)

    try:
        capture.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        capture.stop()
        stream.stop()
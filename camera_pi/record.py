import cv2
import os
import time
from threading import Thread
from picamera2 import Picamera2

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


# --- Thư mục lưu video ---
save_dir = "/home/thanh/ros2_ws/src/p_detect_object/record"
os.makedirs(save_dir, exist_ok=True)

# --- Tên file video ---
video_path = os.path.join(save_dir, "output.mp4")

# --- Bắt đầu stream ---
stream = VideoStream()

# --- Định nghĩa thông số video ---
fps = 30.0                      # số khung hình/giây
frame_size = (800, 600)         # kích thước khung
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec cho .mp4

# --- Tạo đối tượng ghi video ---
out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)

print(f"🎥 Đang quay video... Nhấn Ctrl+C để dừng.")
print(f"📁 File lưu tại: {video_path}")

try:
    while True:
        ret, frame = stream.read()
        if not ret:
            print("Không đọc được frame.")
            continue

        # Resize cho đúng kích thước video
        frame = cv2.resize(frame, frame_size)

        # Ghi vào file
        out.write(frame)

        # Hiển thị nếu muốn xem trực tiếp
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nĐã dừng quay.")

finally:
    stream.stop()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Video đã lưu thành công!")

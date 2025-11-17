# import cv2

# # Mở camera USB (thường là /dev/video0)
# cap = cv2.VideoCapture(0)  # 0 = camera đầu tiên, đổi thành 1 nếu bạn có nhiều camera

# if not cap.isOpened():
#     print("❌ Không mở được camera!")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Không nhận được khung hình!")
#         break

#     # Hiển thị hình ảnh
#     cv2.imshow("USB Camera", frame)

#     # Nhấn ESC để thoát
#     key = cv2.waitKey(1)
#     if key == 27:  # 27 là mã ASCII của phím ESC
#         break

# # Giải phóng tài nguyên
# cap.release()
# cv2.destroyAllWindows()


import cv2
from ultralytics import YOLO
import math

# === 1. Load model YOLOv8 pretrained (COCO có class "person") ===
model = YOLO("yolov8n.pt")  # bạn có thể đổi sang yolov8s.pt, yolov8m.pt, v.v.

# === 2. Mở camera USB ===
cap = cv2.VideoCapture(0)  # 0: camera đầu tiên (/dev/video0)
if not cap.isOpened():
    print("❌ Không mở được camera!")
    exit()

print("✅ Camera đang chạy... Nhấn ESC để thoát.")

# === 3. Vòng lặp đọc khung hình & detect người ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được khung hình!")
        break

    h, w, _ = frame.shape
    cx_frame = w // 2
    cy_frame = h // 2

    # === 4. Dò người trong khung hình ===
    results = model.predict(source=frame, classes=[0], conf=0.5, verbose=False)

    # === 5. Hiển thị kết quả lên khung hình ===
    annotated_frame = results[0].plot()  # Vẽ bounding boxes

    cv2.circle(annotated_frame, (cx_frame, cy_frame), 6, (255, 0, 0), -1)

    MIN_AREA = 5000
    boxes = results[0].boxes.xyxy
    for box in boxes:
        
        x1, y1, x2, y2 = map(int, box[:4])  
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        width = x2 - x1
        height = y2 - y1
        area = width*height

        if area < MIN_AREA:
            continue

        # Vẽ tâm người (màu đỏ)
        cv2.circle(annotated_frame, (cx, cy), 6, (0, 0, 255), -1)

        # Vẽ đường nối
        cv2.line(annotated_frame, (cx_frame, cy_frame), (cx, cy), (0, 255, 255), 2)

        # === 7. Tính khoảng cách từ tâm ảnh đến tâm người ===
        dx = cx - cx_frame
        dy = cy - cy_frame
        distance = math.sqrt(dx**2 + dy**2)

        # Hiển thị thông tin
        cv2.putText(annotated_frame, f"dx={dx}, dy={dy}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(annotated_frame, f"dist={int(distance)} px", (cx + 10, cy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("YOLOv8 Person Detection", annotated_frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) == 27:
        break

# === 6. Giải phóng tài nguyên ===
cap.release()
cv2.destroyAllWindows()
print("🛑 Đã dừng camera.")

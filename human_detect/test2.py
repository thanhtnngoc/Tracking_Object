from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera!")
    exit()

print("✅ Camera đang chạy... Nhấn ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, classes=[0], conf=0.5, verbose=False)

    for result in results:
        boxes = result.boxes  # đối tượng Boxes
        xyxy = boxes.xyxy.cpu().numpy()       # shape = (N,4)
        confs = boxes.conf.cpu().numpy()      # shape = (N,)
        classes = boxes.cls.cpu().numpy()     # shape = (N,)

        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].astype(int)
            conf = float(confs[i])
            cls = int(classes[i])

            if conf < 0.5:
                continue

            # Tâm bounding box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Vẽ khung, tâm, đường từ tâm ảnh -> tâm box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
            cv2.line(frame, (center_x, center_y), (cx, cy), (255, 255, 0), 2)

            # Hiển thị thông tin
            dx, dy = cx - center_x, cy - center_y
            cv2.putText(frame, f"dx={dx}, dy={dy}, conf={conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 - Person Tracking", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

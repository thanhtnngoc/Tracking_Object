import cv2
import mediapipe as mp

# === 1. Khởi tạo MediaPipe Pose ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(  # bạn có thể tinh chỉnh các tham số này
    static_image_mode=False,      # False = dùng cho video stream
    model_complexity=1,           # 0 = nhanh hơn, 2 = chính xác hơn
    enable_segmentation=False,    # True nếu muốn mask người
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === 2. Mở camera USB ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không mở được camera!")
    exit()

print("✅ Camera đang chạy... Nhấn ESC để thoát.")

# === 3. Vòng lặp đọc frame & detect người ===
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không nhận được khung hình!")
        break

    # MediaPipe cần ảnh RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # === 4. Chạy nhận diện pose ===
    results = pose.process(rgb_frame)

    # === 5. Nếu có người ===
    if results.pose_landmarks:
        # Vẽ skeleton lên ảnh
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2)
        )

        # Lấy ví dụ: tọa độ mũi (landmark[0])
        nose = results.pose_landmarks.landmark[0]
        h, w, _ = frame.shape
        cx = int(nose.x * w)
        cy = int(nose.y * h)
        cv2.circle(frame, (cx, cy), 6, (255, 0, 255), -1)
        cv2.putText(frame, "Person detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No person", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # === 6. Hiển thị ===
    cv2.imshow("MediaPipe Pose - Person Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("Đã dừng camera.")

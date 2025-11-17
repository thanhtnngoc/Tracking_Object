import cv2

# --- Mở camera USB ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Không mở được USB camera!")

# Kích thước khung "A4 tham chiếu" (ước lượng pixel)
a4_width_px = 420    # ~ chiều ngang
a4_height_px = 300   # ~ chiều dọc

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình!")
        break

    h, w, _ = frame.shape
    # Tính toạ độ để khung nằm giữa
    x0 = (w - a4_width_px) // 2
    y0 = (h - a4_height_px) // 2
    x1 = x0 + a4_width_px
    y1 = y0 + a4_height_px

    # Vẽ khung chữ nhật (giả lập A4)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

    # Ghi chữ số pixel trên cạnh
    font = cv2.FONT_HERSHEY_SIMPLEX
    

    cv2.putText(frame, f"Width: {a4_width_px}px", (x0 + 10, y0 - 10),
                font, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Height: {a4_height_px}px", (x1 + 10, y0 + a4_height_px // 2),
                font, 0.7, (0, 255, 255), 2)

    # Hiển thị video
    cv2.imshow("USB Cam - A4 Reference", frame)

    # Nhấn ESC để thoát
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

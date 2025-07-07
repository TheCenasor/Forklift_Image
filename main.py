from ultralytics import YOLO
import cv2
import time

# Eğitilmiş modeli yükle
model = YOLO("Yolo11_Forklift_Model.pt")

# Video dosyasını aç
video_path = r"Video\A_fixed_overhead_202507061706_zumxo.mp4"
cap = cv2.VideoCapture(video_path)

# Video boyutu ve FPS bilgisi al
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Çıktı videosunu ayarla
out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

# FPS ölçümü için zaman başlat
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # FPS hesapla
    current_time = time.time()
    elapsed_time = current_time - prev_time
    prev_time = current_time
    current_fps = 1 / elapsed_time if elapsed_time > 0 else 0

    # YOLO ile tahmin yap
    results = model.predict(frame)

    for result in results:
        boxes = result.boxes

        if len(boxes) > 0:
            # Sadece ilk kutuyu al
            box = boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Kare çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        break  # Sadece bir nesne göster

    # 🔢 FPS’yi sağ üst köşeye yaz
    fps_text = f'FPS: {current_fps:.2f}'
    cv2.putText(frame, fps_text, (10, 30),  # sol üst köşe (x=10, y=30)
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Görüntüyü göster ve yaz
    cv2.imshow('YOLOv8 Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Evaluation der Full-Frame-Gesichtserkennung als Vergleich zur ROI-basierten Gesichtserkennung

import time

import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis


VIDEO_PATH = "path/to/input_video.mp4"


# ---------------------------------------------------------
# Modelle laden
# ---------------------------------------------------------

# Personendetektion
model = YOLO("yolov8m.pt")

# Gesichtserkennung mit SCRFD
face_detector = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"]
)

face_detector.prepare(
    ctx_id=-1,
    det_size=(320, 320),
    det_thresh=0.3
)


# ---------------------------------------------------------
# Video öffnen
# ---------------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Das Eingabevideo konnte nicht geöffnet werden: {VIDEO_PATH}"
    )


# ---------------------------------------------------------
# Einzelnen Frame verarbeiten
# ---------------------------------------------------------

def process_frame(frame, model, face_detector):
    # YOLO + ByteTrack wie bei der ROI-Variante
    model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

    # Gesichtserkennung trotzdem auf dem vollständigen Frame
    faces = face_detector.get(frame)

    return frame, len(faces)


# ---------------------------------------------------------
# Hauptschleife und Laufzeitmessung
# ---------------------------------------------------------

# Evaluationsvariablen
processed_frames = 0
total_processing_time = 0.0
full_frame_face_detections = 0      # Insgesamt erkannte Gesichter
frames_with_full_frame_face = 0     # Frames mit mindestens einer Erkennung


while True:
    ret, frame = cap.read()

    if not ret:
        break

    start_time = time.perf_counter()

    processed_frame, face_count = process_frame(frame, model, face_detector)

    processing_time = time.perf_counter() - start_time
    total_processing_time += processing_time

    processed_frames += 1
    full_frame_face_detections += face_count

    if face_count > 0:
        frames_with_full_frame_face += 1


cap.release()

cv2.destroyAllWindows()


# ---------------------------------------------------------
# Laufzeit ausgeben
# ---------------------------------------------------------

if processed_frames > 0:
    average_processing_time = total_processing_time / processed_frames
    processing_fps = 1 / average_processing_time

    print(f"\nVerarbeitete Frames: {processed_frames}")
    print(f"Durchschnittliche Zeit pro Frame: {average_processing_time:.4f} s")
    print(f"Verarbeitungsgeschwindigkeit: {processing_fps:.2f} FPS")
    print(f"Full-Frame-Gesichtsdetektionen: {full_frame_face_detections}")
    print(f"Frames mit mindestens einem erkannten Gesicht: {frames_with_full_frame_face}")

print("Verarbeitung abgeschlossen.")
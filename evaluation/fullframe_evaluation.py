import os
import time

import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ---------------------------------------------------------
# Einstellungen
# ---------------------------------------------------------

VIDEO_PATH = "Testvideos/MOT17-09.mp4"
OUTPUT_DIRECTORY = "Ausgabevideos"

SAVE_VIDEO = False
SHOW_VIDEO = False

# Für einen fairen Vergleich zunächst denselben Wert wie bisher verwenden.
OUTPUT_FPS = 50.0


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

video_name = os.path.splitext(
    os.path.basename(VIDEO_PATH)
)[0]

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

input_fps = cap.get(cv2.CAP_PROP_FPS)

print(f"FPS des Eingabevideos: {input_fps}")


# ---------------------------------------------------------
# VideoWriter vorbereiten
# ---------------------------------------------------------

video_writer = None

if SAVE_VIDEO:
    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIRECTORY,
        f"{video_name}_scrfd_gesamtframe.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        output_path,
        fourcc,
        OUTPUT_FPS,
        (frame_width, frame_height)
    )

    if not video_writer.isOpened():
        raise RuntimeError(
            "Das Ausgabevideo konnte nicht erstellt werden."
        )

    print(f"Ausgabevideo: {output_path}")


# ---------------------------------------------------------
# Farben für Track-IDs
# ---------------------------------------------------------

ID_COLORS = [
    (255, 204, 204),  # Pastellrosa
    (204, 255, 204),  # Pastellgrün
    (204, 204, 255),  # Pastellblau
    (255, 255, 204),  # Pastellgelb
    (255, 204, 255),  # Pastelllila
    (204, 255, 255),  # Pastelltürkis
    (230, 216, 173),  # Sand
    (221, 204, 255),  # Lavendel
    (204, 230, 255),  # Himmelblau
    (204, 255, 230),  # Mint
]


def get_color_for_id(track_id):
    """
    Ordnet jeder Track-ID reproduzierbar eine Farbe zu.
    """

    return ID_COLORS[track_id % len(ID_COLORS)]


# ---------------------------------------------------------
# Gesichtsanonymisierung
# ---------------------------------------------------------

def anonymize_face(frame, face_box):
    """
    Verkleinert die erkannte Gesichtsregion, wendet einen
    Gaussian Blur an und skaliert sie anschließend wieder
    auf die ursprüngliche Größe.
    """

    x1, y1, x2, y2 = face_box

    current_frame_height, current_frame_width = frame.shape[:2]

    # Gesichtsbox auf Bildgrenzen beschränken
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(current_frame_width, x2)
    y2 = min(current_frame_height, y2)

    if x2 <= x1 or y2 <= y1:
        return

    face_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0:
        return

    face_height, face_width = face_roi.shape[:2]

    # Gesichtsgröße näherungsweise bestimmen
    largest_side = max(
        face_width,
        face_height
    )

    # Dynamische Skalierung
    if largest_side >= 160:
        scale_factor = 0.15

    elif largest_side >= 80:
        scale_factor = 0.20

    else:
        scale_factor = 0.25

    small_width = max(
        1,
        int(face_width * scale_factor)
    )

    small_height = max(
        1,
        int(face_height * scale_factor)
    )

    # Gesicht verkleinern
    small_face = cv2.resize(
        face_roi,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA
    )

    smallest_side = min(
        small_width,
        small_height
    )

    # Zu kleine Regionen können nicht sinnvoll geblurrt werden
    if smallest_side < 3:
        return

    # Größtmöglichen ungeraden Kernel bestimmen
    kernel_size = smallest_side

    if kernel_size % 2 == 0:
        kernel_size -= 1

    blurred_small_face = cv2.GaussianBlur(
        small_face,
        (kernel_size, kernel_size),
        0
    )

    # Anonymisierte Region auf ursprüngliche Größe bringen
    blurred_face = cv2.resize(
        blurred_small_face,
        (face_width, face_height),
        interpolation=cv2.INTER_LINEAR
    )

    # Ursprüngliche Gesichtsregion ersetzen
    frame[y1:y2, x1:x2] = blurred_face


# ---------------------------------------------------------
# Einzelnen Frame verarbeiten
# ---------------------------------------------------------

def process_frame(frame, model, face_detector):
    # YOLO + ByteTrack wie bei der ROI-Variante
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Gesichtserkennung trotzdem auf dem vollständigen Frame
    faces = face_detector.get(frame)

    return frame, len(faces)


# ---------------------------------------------------------
# Hauptschleife und Laufzeitmessung
# ---------------------------------------------------------

processed_frames = 0
total_processing_time = 0.0

full_frame_face_detections = 0
frames_with_full_frame_face = 0


while True:
    ret, frame = cap.read()

    if not ret:
        break

    start_time = time.perf_counter()

    processed_frame, face_count = process_frame(
        frame,
        model,
        face_detector
    )

    processing_time = time.perf_counter() - start_time
    total_processing_time += processing_time

    processed_frames += 1
    full_frame_face_detections += face_count

    if face_count > 0:
        frames_with_full_frame_face += 1




# ---------------------------------------------------------
# Ressourcen freigeben
# ---------------------------------------------------------

cap.release()

if video_writer is not None:
    video_writer.release()

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
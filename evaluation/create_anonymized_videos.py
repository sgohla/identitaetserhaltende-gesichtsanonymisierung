# Pipeline mit Erweiterungen: Selektives Tracking(alte Version), Ausfall von Gesichtserkennung (Kalman-Filter und Auswahl des besten Gesichts)

import time
import os
import cv2
from ultralytics import YOLO
import numpy as np
from insightface.app import FaceAnalysis

# YOLO-Modell laden
model = YOLO("yolov8m.pt")

#SCRFD-Modell laden
face_detector = FaceAnalysis(
    name="buffalo_l",
    allowed_modules=["detection"],
    providers=["CPUExecutionProvider"]
)

# Kleinere Eingabegröße:
# Das gesamte Bild wird stärker verkleinert, wodurch sehr große Gesichter für das Modell kleiner erscheinen.
face_detector.prepare(
    ctx_id=-1,
    det_size=(320, 320),
    det_thresh=0.4
)


# Eingabevideo
video_path = "Testvideos/MOT17-09.mp4"

cap = cv2.VideoCapture(video_path)

# Dateiname ohne Ordner und Endung
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Videoeigenschaften abrufen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"fps: {fps}")

# VideoWriter-Objekt erstellen, um das Ergebnisvideo zu speichern
save_video = True
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None
OUTPUT_FPS = 45.0

if save_video:
    
    os.makedirs("Ausgabevideos", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        f"Ausgabevideos/{video_name}_anonymized.mp4",
        fourcc,
        fps,
        (frame_width, frame_height)
    )


# Randbereich für die Personenbox, um zu erkennen, ob sie am Rand des Frames liegt

FRAME_BORDER_MARGIN = 10

def touches_frame_border(person_box, frame):

    x1, y1, x2, y2 = person_box

    frame_height, frame_width = frame.shape[:2]

    return (
        x1 <= FRAME_BORDER_MARGIN
        or y1 <= FRAME_BORDER_MARGIN
        or x2 >= frame_width - FRAME_BORDER_MARGIN
        or y2 >= frame_height - FRAME_BORDER_MARGIN)



# Variablen für Kalman-Filter, wenn Gesicht nicht erkannt wird

face_kalman_filters = {}
face_missing_frames = {}
last_face_sizes = {}
face_detection_counts = {}

MAX_MISSING_FRAMES = 4
MIN_FACE_DETECTIONS_FOR_KALMAN = 4


# Kalman-Filter für die Gesichtsposition erstellen
def create_face_kalman_filter(face_box):
    x1, y1, x2, y2 = face_box

    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    kalman_filter = cv2.KalmanFilter(4, 2)

    kalman_filter.transitionMatrix = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    
    kalman_filter.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kalman_filter.errorCovPost = np.eye(4, dtype=np.float32)

    initial_state = np.array([
        [center_x],
        [center_y],
        [0],
        [0]
    ], dtype=np.float32)

    kalman_filter.statePost = initial_state
    kalman_filter.statePre = initial_state.copy()

    return kalman_filter


def update_face_kalman_filter(kalman_filter, face_box):
    x1, y1, x2, y2 = face_box

    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width / 2
    center_y = y1 + height / 2

    measurement = np.array([
        [center_x],
        [center_y]
    ], dtype=np.float32)

    kalman_filter.predict()
    kalman_filter.correct(measurement)

    return (width, height)
    

def predict_face_box(kalman_filter, face_size, frame):
    prediction = kalman_filter.predict()

    center_x = float(prediction[0, 0])
    center_y = float(prediction[1, 0])

    width, height = face_size

    frame_height, frame_width = frame.shape[:2]

    # Größe absichern
    width = min(width, frame_width)
    height = min(height, frame_height)

    if width <= 0 or height <= 0:
        return None

    x1 = max(0, int(center_x - width / 2))
    y1 = max(0, int(center_y - height / 2))
    x2 = min(frame_width, int(center_x + width / 2))
    y2 = min(frame_height, int(center_y + height / 2))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)
    

def calculate_face_distance(face, x1, y1, last_center_x, last_center_y):

    local_x1, local_y1, local_x2, local_y2 = face.bbox.astype(int)

    center_x = x1 + (local_x1 + local_x2) / 2
    center_y = y1 + (local_y1 + local_y2) / 2

    return np.hypot(
        center_x - last_center_x,
        center_y - last_center_y
    )
    
def calculate_initial_face_score(face, roi_x1, roi_y1, person_box):
    person_x1, person_y1, person_x2, person_y2 = person_box

    local_x1, local_y1, local_x2, local_y2 = face.bbox.astype(int)

    face_center_x = roi_x1 + (local_x1 + local_x2) / 2
    face_center_y = roi_y1 + (local_y1 + local_y2) / 2

    person_width = person_x2 - person_x1
    person_height = person_y2 - person_y1

    expected_x = (person_x1 + person_x2) / 2
    expected_y = person_y1 + 0.25 * person_height

    distance = np.hypot(
        face_center_x - expected_x,
        face_center_y - expected_y
    )

    normalized_distance = distance / max(person_width, person_height, 1)

    return face.det_score - normalized_distance


last_face_boxes = {}

def detect_face(frame, person_box, face_detector, track_id):
    x1, y1, x2, y2 = person_box

    frame_height, frame_width = frame.shape[:2]

    if touches_frame_border(person_box, frame):
        person_width = x2 - x1
        person_height = y2 - y1

        x1 -= int(person_width * 0.20)
        x2 += int(person_width * 0.20)
        y1 -= int(person_height * 0.20)
        y2 += int(person_height * 0.05)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    person_roi = frame[y1:y2, x1:x2]

    if person_roi.size == 0:
        return None

    faces = face_detector.get(person_roi)

    if not faces:
        return None

    if len(faces) == 1:
        best_face = faces[0]

    elif track_id not in last_face_boxes:
        best_face = max(faces, key=lambda face: calculate_initial_face_score(face, x1, y1, person_box))

    else:
        last_x1, last_y1, last_x2, last_y2 = last_face_boxes[track_id]

        last_center_x = (last_x1 + last_x2) / 2
        last_center_y = (last_y1 + last_y2) / 2
        last_width = last_x2 - last_x1
        last_height = last_y2 - last_y1


        best_face = min(faces, key= lambda face: calculate_face_distance(face, x1, y1, last_center_x, last_center_y))
        best_distance = calculate_face_distance(best_face, x1, y1, last_center_x, last_center_y)

        max_distance = 2.5 * max(last_width, last_height)

        if best_distance > max_distance:
            return None

    face_x1, face_y1, face_x2, face_y2 = best_face.bbox.astype(int)

    face_x1 += x1
    face_y1 += y1
    face_x2 += x1
    face_y2 += y1

    face_x1 = max(0, face_x1)
    face_y1 = max(0, face_y1)
    face_x2 = min(frame_width, face_x2)
    face_y2 = min(frame_height, face_y2)

    face_box = (face_x1, face_y1, face_x2, face_y2)
    last_face_boxes[track_id] = face_box

    return face_box


def anonymize_face(frame, face_box):

    x1, y1, x2, y2 = face_box

    frame_height, frame_width = frame.shape[:2]

    # Gesichtsbox auf Bildgrenzen beschränken
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    face_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0:
        return

    face_height, face_width = face_roi.shape[:2]

    # Dynamische Skalierung abhängig von der Gesichtsgröße
    largest_side = max(face_width, face_height)

    # Je nach Gesichtsgröße unterschiedlich stark verkleinern
    if largest_side >= 160:
        scale_factor = 0.15
    elif largest_side >= 80:
        scale_factor = 0.20
    else:
        scale_factor = 0.25

    # Größe der verkleinerten Gesichtsregion berechnen
    small_width = max(1, int(face_width * scale_factor))
    small_height = max(1, int(face_height * scale_factor))

    # Gesicht verkleinern
    small_face = cv2.resize(face_roi, (small_width, small_height), interpolation=cv2.INTER_AREA)

    # Kernel an die verkleinerte Region anpassen
    smallest_side = min(small_width, small_height)

    # Zu kleine Regionen können nicht sinnvoll weichgezeichnet werden
    if smallest_side < 3:
        return

    # Größtmöglichen ungeraden Kernel bestimmen
    kernel_size = smallest_side

    if kernel_size % 2 == 0:
        kernel_size -= 1

    # Gaussian Blur anwenden
    blurred_small_face = cv2.GaussianBlur(small_face, (kernel_size, kernel_size), 0) 

    # Wieder auf ursprüngliche Größe vergrößern
    blurred_face = cv2.resize(blurred_small_face, (face_width, face_height), interpolation=cv2.INTER_LINEAR)

    # Anonymisierte Gesichtsregion in den Frame zurückschreiben
    frame[y1:y2, x1:x2] = blurred_face
    

track_last_seen = {}
MAX_TRACK_MEMORY_FRAMES = 30


def process_frame(frame, model, face_detector, frame_idx):
    
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    boxes = results[0].boxes
    
    
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int) # Koordinaten der Box
        cls = int(box.cls[0]) # Klasse als Integer
        conf = float(box.conf[0])# Konfidenz als Float
        
        if cls != 0: # Klasse 0 entspricht "person" im COCO-Datensatz
            continue    
        
        if box.id is None: #Objekte ohne Track-ID überspringen
            continue
        
        track_id = int(box.id[0])
        
        # Zeitpunkt speichern, zu dem der Track zuletzt sichtbar war
        track_last_seen[track_id] = frame_idx
        
        person_box = (x1, y1, x2, y2)
        
        
        # Gesichtsdetektion und Anonymisierung
        face_box = detect_face(frame, person_box, face_detector, track_id)

        if face_box is not None:
            face_detection_counts[track_id] = (face_detection_counts.get(track_id, 0) + 1)
            x1_face, y1_face, x2_face, y2_face = face_box
            face_size = (x2_face - x1_face, y2_face - y1_face)

            if track_id not in face_kalman_filters:
                face_kalman_filters[track_id] = create_face_kalman_filter(face_box)
            else:
                face_size = update_face_kalman_filter(face_kalman_filters[track_id],face_box)

            last_face_sizes[track_id] = face_size
            face_missing_frames[track_id] = 0

            anonymize_face(frame, face_box)

        else:
            missing_frames = face_missing_frames.get(track_id, 0)
            detection_count = face_detection_counts.get(track_id, 0)

            if (
                track_id in face_kalman_filters
                and track_id in last_face_sizes
                and detection_count >= MIN_FACE_DETECTIONS_FOR_KALMAN
                and missing_frames < MAX_MISSING_FRAMES
            ):
                predicted_face_box = predict_face_box(face_kalman_filters[track_id], last_face_sizes[track_id], frame)

                if predicted_face_box is not None:
                    anonymize_face(frame, predicted_face_box)

                    # Nur zum Testen
                    # cv2.rectangle(
                    #     frame,
                    #     (predicted_face_box[0], predicted_face_box[1]),
                    #     (predicted_face_box[2], predicted_face_box[3]),
                    #     (0, 165, 255),
                    #     2
                    # )

                face_missing_frames[track_id] = missing_frames + 1
            
        
        
     # Nicht mehr benötigte Daten alter Tracks entfernen
    old_track_ids = [
        track_id
        for track_id, last_seen in track_last_seen.items()
        if frame_idx - last_seen > MAX_TRACK_MEMORY_FRAMES
    ]

    for track_id in old_track_ids:
        track_last_seen.pop(track_id, None)
        face_kalman_filters.pop(track_id, None)
        face_missing_frames.pop(track_id, None)
        last_face_sizes.pop(track_id, None)
        last_face_boxes.pop(track_id, None)
        face_detection_counts.pop(track_id, None)

    return frame
    


processed_frames = 0
total_processing_time = 0.0

frame_idx = 0


while True:

    ret, frame = cap.read()

    if not ret:
        break
    
    frame_idx += 1

    # Zeitmessung für die Verarbeitung eines Frames starten
    start_time = time.perf_counter()

    processed_frame = process_frame(frame, model, face_detector, frame_idx)

    # Verarbeitungszeit dieses Frames berechnen
    processing_time = time.perf_counter() - start_time

    total_processing_time += processing_time
    processed_frames += 1

    if save_video:
        video_writer.write(processed_frame)



cap.release()

if save_video:
    video_writer.release()
    
cv2.destroyAllWindows()

if processed_frames > 0:

    average_processing_time = (total_processing_time / processed_frames)

    processing_fps = (1.0 / average_processing_time)

    print(f"\nVerarbeitete Frames: {processed_frames}")

    print("Durchschnittliche Zeit pro Frame: "f"{average_processing_time:.4f} s")

    print("Verarbeitungsgeschwindigkeit: "f"{processing_fps:.2f} FPS")

       
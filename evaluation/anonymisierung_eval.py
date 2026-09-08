# Evaluation der Gesichtserkennung und Stabilisierung der Anonymisierung
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
    det_thresh=0.3
)


# Eingabevideo
video_path = "path/to/input_video.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Video konnte nicht geöffnet werden: {video_path}")

# Dateiname ohne Ordner und Endung
video_name = os.path.splitext(os.path.basename(video_path))[0]



MULTI_FACE_OUTPUT_DIR = f"Evaluation/multi_face/{video_name}"
os.makedirs(MULTI_FACE_OUTPUT_DIR, exist_ok=True)

MULTI_FACE_SAVE_INTERVAL = 4
last_multi_face_saved = {}


DET_THRESH = 0.3

LOW_CONF_OUTPUT_DIR = f"Evaluation/low_conf/{video_name}_thresh_{DET_THRESH}_überarbeitet"
KALMAN_OUTPUT_DIR = f"Evaluation/kalman/{video_name}_thresh_{DET_THRESH}_überarbeitet"

os.makedirs(LOW_CONF_OUTPUT_DIR, exist_ok=True)
os.makedirs(KALMAN_OUTPUT_DIR, exist_ok=True)

LOW_CONF_UPPER_LIMIT = 0.4


# Videoeigenschaften abrufen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"fps: {fps}")

# VideoWriter-Objekt erstellen, um das Ergebnisvideo zu speichern
save_video = False 
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None

if save_video:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        f"Ausgabevideos/{video_name}_erweiterung.mp4",
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
kalman_active = {}

MAX_MISSING_FRAMES = 4
MIN_FACE_DETECTIONS_FOR_KALMAN = 5


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


def save_multi_face_case(frame, person_box, faces, best_face, roi_x1, roi_y1, frame_idx, track_id, selection_method, baseline_face):
    last_saved = last_multi_face_saved.get(track_id)

    if last_saved is not None and frame_idx - last_saved < MULTI_FACE_SAVE_INTERVAL:
        return

    debug_frame = frame.copy()

    # Personenbox markieren
    px1, py1, px2, py2 = person_box
    cv2.rectangle(debug_frame, (px1, py1), (px2, py2), (255, 255, 255), 2)

    for i, face in enumerate(faces):
        fx1, fy1, fx2, fy2 = face.bbox.astype(int)

        fx1 += roi_x1
        fx2 += roi_x1
        fy1 += roi_y1
        fy2 += roi_y1

        if face is best_face:
            color = (0, 255, 0)
            label = f"OURS {i} conf={face.det_score:.2f}"
        elif face is baseline_face:
            color = (255, 0, 0)
            label = f"CONF {i} conf={face.det_score:.2f}"
        else:
            color = (0, 0, 255)
            label = f"{i} conf={face.det_score:.2f}"

        cv2.rectangle(debug_frame, (fx1, fy1), (fx2, fy2), color, 2)
        cv2.putText(debug_frame, label, (fx1, max(fy1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    filename = f"{video_name}_frame_{frame_idx:05d}_track_{track_id}.jpg"
    cv2.imwrite(os.path.join(MULTI_FACE_OUTPUT_DIR, filename), debug_frame)

    last_multi_face_saved[track_id] = frame_idx


last_face_boxes = {}

# -------------------------------------------------
# Evaluationsvariablen
# -------------------------------------------------

# Mehrfachgesichtsauswahl
face_selection_cases = 0          # Frames mit mindestens einem Gesichtskandidaten
multi_face_cases = 0              # Frames mit mehreren Gesichtskandidaten
multi_face_initial_selection = 0  # Auswahl ohne vorherige Gesichtsposition
multi_face_history_selection = 0  # Auswahl anhand vorheriger Gesichtsposition
multi_face_rejected = 0           # Abgelehnte Auswahl wegen zu großer Distanz
multi_face_same_as_conf = 0       # Eigene Auswahl entspricht höchster SCRFD-Konfidenz
multi_face_different_from_conf = 0  # Eigene Auswahl weicht von höchster Konfidenz ab

# Gesichtserkennung und Kalman-Überbrückung
face_detection_failures = 0       # Frames ohne gültige direkte Gesichtserkennung
kalman_predictions = 0            # Erfolgreich erzeugte Kalman-Vorhersagen
kalman_not_possible = 0           # Ausfälle, bei denen keine Überbrückung möglich war
kalman_eligible_failures = 0      # Ausfälle, bei denen Kalman grundsätzlich eingesetzt werden durfte
kalman_prediction_failures = 0    # Kalman-Vorhersagen ohne gültige Gesichtsbox
face_detected_frames = 0          # Direkte SCRFD-Gesichtserkennungen
kalman_used_frames = 0            # Tatsächlich zur Anonymisierung verwendete Kalman-Frames

# Niedrigkonfidente Gesichtserkennungen für die Schwellenwertanalyse
low_conf_detections = 0

def detect_face(frame, person_box, face_detector, track_id, frame_idx):
    global face_selection_cases
    global multi_face_cases
    global multi_face_initial_selection
    global multi_face_history_selection
    global multi_face_rejected
    global multi_face_same_as_conf
    global multi_face_different_from_conf
    
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
        return None, None

    faces = face_detector.get(person_roi)

    if not faces:
        return None, None
    
    face_selection_cases += 1
    baseline_face = max(faces, key=lambda face: face.det_score) if len(faces) > 1 else None

    if len(faces) == 1:
        best_face = faces[0]
        selection_method = "single"

    elif track_id not in last_face_boxes:
        
        #Evaluation
        multi_face_cases += 1
        multi_face_initial_selection += 1
        selection_method = "initial"
        
        best_face = max(faces, key=lambda face: calculate_initial_face_score(face, x1, y1, person_box))

    else:
        
        #Evaluation
        multi_face_cases += 1
        multi_face_history_selection += 1
        selection_method = "history"
        
        last_x1, last_y1, last_x2, last_y2 = last_face_boxes[track_id]

        last_center_x = (last_x1 + last_x2) / 2
        last_center_y = (last_y1 + last_y2) / 2
        last_width = last_x2 - last_x1
        last_height = last_y2 - last_y1


        best_face = min(faces, key= lambda face: calculate_face_distance(face, x1, y1, last_center_x, last_center_y))
        best_distance = calculate_face_distance(best_face, x1, y1, last_center_x, last_center_y)

        max_distance = 2.5 * max(last_width, last_height)

        if best_distance > max_distance:
            multi_face_rejected += 1
    

            save_multi_face_case(frame, person_box, faces, best_face, x1, y1, frame_idx, track_id, "history_rejected", baseline_face)


            return None , None
        
    if len(faces) > 1:

        if best_face is baseline_face:
            multi_face_same_as_conf += 1
        else:
            multi_face_different_from_conf += 1
            save_multi_face_case(frame, person_box, faces, best_face, x1, y1, frame_idx, track_id, selection_method, baseline_face)

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

    return face_box, float(best_face.det_score)


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
    
    global face_detection_failures
    global kalman_predictions
    global kalman_not_possible
    global kalman_eligible_failures
    global kalman_prediction_failures
    global face_detected_frames
    global kalman_used_frames
    global low_conf_detections

    
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    boxes = results[0].boxes
    
    detection_frame = frame.copy()
    
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
        face_box, face_conf = detect_face(detection_frame, person_box, face_detector, track_id, frame_idx)

        if face_box is not None:
            face_detected_frames += 1
            if face_conf < LOW_CONF_UPPER_LIMIT:
                low_conf_detections += 1

                debug_frame = frame.copy()
                fx1, fy1, fx2, fy2 = face_box

                cv2.rectangle(debug_frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(debug_frame, f"SCRFD conf={face_conf:.3f}", (fx1, max(fy1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.imwrite(os.path.join(LOW_CONF_OUTPUT_DIR, f"{video_name}_frame_{frame_idx:05d}_track_{track_id}_conf_{face_conf:.3f}.jpg"), debug_frame)
            
            face_detection_counts[track_id] = (face_detection_counts.get(track_id, 0) + 1)
            kalman_active[track_id] = False
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
            face_detection_failures += 1
            missing_frames = face_missing_frames.get(track_id, 0)
            detection_count = face_detection_counts.get(track_id, 0)
            
            # Beim ersten Ausfall prüfen, ob zuvor mindestens 5 aufeinanderfolgende Gesichtserkennungen vorlagen
            if missing_frames == 0:
                kalman_active[track_id] = (detection_count >= MIN_FACE_DETECTIONS_FOR_KALMAN)
                
            face_detection_counts[track_id] = 0
            
            if (
                track_id in face_kalman_filters
                and track_id in last_face_sizes
                and kalman_active.get(track_id, False)
                and missing_frames < MAX_MISSING_FRAMES
            ):
                kalman_eligible_failures += 1
                predicted_face_box = predict_face_box(face_kalman_filters[track_id], last_face_sizes[track_id], frame)

                if predicted_face_box is not None:
                    kalman_predictions += 1
                    kalman_used_frames += 1
                    
                    debug_frame = frame.copy()
                    kx1, ky1, kx2, ky2 = predicted_face_box

                    cv2.rectangle(debug_frame, (kx1, ky1), (kx2, ky2), (0, 255, 0), 2)
                    cv2.putText(debug_frame, "Kalman", (kx1, max(ky1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    cv2.imwrite(os.path.join(KALMAN_OUTPUT_DIR, f"{video_name}_frame_{frame_idx:05d}_track_{track_id}.jpg"), debug_frame)
                    
                    anonymize_face(frame, predicted_face_box)
                else:
                    kalman_not_possible += 1
                    kalman_prediction_failures += 1

                face_missing_frames[track_id] = missing_frames + 1
                
            else:
                kalman_not_possible += 1
        
        
     # Nicht mehr benötigte Daten alter Tracks entfernen
    old_track_ids = [track_id
        for track_id, last_seen in track_last_seen.items()
        if frame_idx - last_seen > MAX_TRACK_MEMORY_FRAMES]

    for track_id in old_track_ids:
        track_last_seen.pop(track_id, None)
        face_kalman_filters.pop(track_id, None)
        face_missing_frames.pop(track_id, None)
        last_face_sizes.pop(track_id, None)
        last_face_boxes.pop(track_id, None)
        face_detection_counts.pop(track_id, None)
        kalman_active.pop(track_id, None)

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

if processed_frames > 0:

    average_processing_time = (total_processing_time / processed_frames)

    processing_fps = (1.0 / average_processing_time)

    print(f"\nVerarbeitete Frames: {processed_frames}")

    print("Durchschnittliche Zeit pro Frame: "f"{average_processing_time:.4f} s")

    print("Verarbeitungsgeschwindigkeit: "f"{processing_fps:.2f} FPS")
    
    
print("\n--- Stabilisierungsevaluation ---")
print(f"Video: {video_name}")

print("\nMehrfachgesichtsauswahl:")
print(f"Frames mit mindestens einem Gesichtskandidaten: {face_selection_cases}")
print(f"Frames mit mehreren Gesichtskandidaten: {multi_face_cases}")

if face_selection_cases > 0:
    multi_face_rate = multi_face_cases / face_selection_cases * 100
    print(f"Anteil mit erforderlicher Mehrfachauswahl: {multi_face_rate:.2f} %")

print(f"Gleiche Auswahl wie höchste Confidence: {multi_face_same_as_conf}")
print(f"Andere Auswahl als höchste Confidence: {multi_face_different_from_conf}")

if multi_face_cases > 0:
    difference_rate = multi_face_different_from_conf / multi_face_cases * 100
    print(f"Anteil abweichender Entscheidungen: {difference_rate:.2f} %")

print("\nGesichtserkennung und Kalman:")
print(f"Direkte SCRFD-Detektionen: {face_detected_frames}")
print(f"Niedrigkonfidente Detektionen (< {LOW_CONF_UPPER_LIMIT}): {low_conf_detections}")
print(f"Kalman-Überbrückungen: {kalman_used_frames}")
print(f"Anonymisierungen insgesamt: {face_detected_frames + kalman_used_frames}")

       
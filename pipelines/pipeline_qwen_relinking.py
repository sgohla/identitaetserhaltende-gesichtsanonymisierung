# Modell mit Qwen
# + neue Logik fürs Selektive Tracking (JSON Datei erstellen und separater video viewer)

import time
import os
import cv2
from ultralytics import YOLO
import numpy as np
import torch
import tempfile
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor)
from qwen_vl_utils import process_vision_info
from insightface.app import FaceAnalysis
import json

# -------------------------------------------------
# Konfiguration
# -------------------------------------------------
video_path = "Testvideos/durcheinander.mp4"
SAVE_VIDEO = False

MIN_HISTOGRAM_SIMILARITY = 0.60
MIN_FRAMES_BEFORE_RELINKING = 5

# -------------------------------------------------
# Qwen-Modell
# -------------------------------------------------

QWEN_MODEL_NAME = "/fshpc/sgohla/bachelorarbeit/models/Qwen2.5-VL-7B-Instruct"

qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    local_files_only=True
)

qwen_processor = AutoProcessor.from_pretrained(
    QWEN_MODEL_NAME,
    local_files_only=True
)

qwen_model.eval()

# -------------------------------------------------
# Modelle
# -------------------------------------------------

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


# -------------------------------------------------
# Videoein- und -ausgabe
# -------------------------------------------------

cap = cv2.VideoCapture(video_path)

# Dateiname ohne Ordner und Endung
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Videoeigenschaften abrufen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"fps: {fps}")

# VideoWriter-Objekt erstellen, um das Ergebnisvideo zu speichern
video_writer = None
OUTPUT_FPS = 45.0

if SAVE_VIDEO:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        f"Ausgabevideos/{video_name}_qwen_anonymized.mp4",
        fourcc,
        fps,
        (frame_width, frame_height)
    )
    


# -------------------------------------------------
# Randüberprüfung
# -------------------------------------------------

FRAME_BORDER_MARGIN = 10


def touches_frame_border(person_box, frame):
    """Prüft, ob eine Personenbox den definierten Bildrand berührt."""

    x1, y1, x2, y2 = person_box

    frame_height, frame_width = frame.shape[:2]

    return (
        x1 <= FRAME_BORDER_MARGIN
        or y1 <= FRAME_BORDER_MARGIN
        or x2 >= frame_width - FRAME_BORDER_MARGIN
        or y2 >= frame_height - FRAME_BORDER_MARGIN
    )

#----------------------------
# VLM Referenzframes
#----------------------------    

reference_frames = {}

def calculate_reference_score(person_box, confidence, frame):
    """Bewertet die Eignung eines Personenausschnitts als Referenzframe."""
    
    x1, y1, x2, y2 = person_box

    person_height = y2 - y1

    score = confidence

    # Person am Bildrand ist als Referenz eher ungeeignet
    if touches_frame_border(person_box, frame):
        score -= 0.5

    # Sehr kleine Personen etwas abwerten
    if person_height < 100:
        score -= 0.2

    return score


def update_reference_frames(track_id, frame, person_box, confidence, frame_idx):
    """Speichert bis zu zwei geeignete Referenzframes eines Tracks."""
    
    x1, y1, x2, y2 = person_box

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return

    score = calculate_reference_score(person_box, confidence, frame)

    candidate = {
        "frame_idx": frame_idx,
        "crop": crop.copy(),
        "score": score
    }

    if track_id not in reference_frames:
        reference_frames[track_id] = [candidate]
        return

    stored = reference_frames[track_id]

    # Zweite Referenz nur mit ausreichendem zeitlichem Abstand speichern
    if len(stored) == 1:
        if abs(frame_idx - stored[0]["frame_idx"]) >= 8:
            stored.append(candidate)

        elif score > stored[0]["score"]:
            stored[0] = candidate

        return

    # Von den bereits gespeicherten Referenzen die schlechtere suchen
    worst_index = min(range(len(stored)),key=lambda i: stored[i]["score"])

    other_index = 1 - worst_index

    # Ersatz nur, wenn der neue Frame besser ist undngenügend Abstand zur anderen Referenz besitzt
    if (score > stored[worst_index]["score"] and abs(frame_idx - stored[other_index]["frame_idx"]) >= 8):
        stored[worst_index] = candidate


def compare_reference_frames_with_qwen(old_frames, new_frames):

    if len(old_frames) < 1 or len(new_frames) < 1:
        return None

    selected_old = old_frames[:2]
    selected_new = new_frames[:2]

    temp_paths = []

    try:

        # Alte Referenzen
        for reference in selected_old:

            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

            temp_file.close()

            cv2.imwrite(temp_file.name, reference["crop"])

            temp_paths.append(temp_file.name)

        # Neue Referenzen
        for reference in selected_new:

            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)

            temp_file.close()

            cv2.imwrite(temp_file.name, reference["crop"])

            temp_paths.append(temp_file.name)

        num_old = len(selected_old)
        num_new = len(selected_new)

        prompt = f"""
                You are performing a person re-identification verification.

                The first {num_old} image(s) show the SAME physical person
                from a previously lost track.

                The following {num_new} image(s) show a newly detected track.

                The candidate has already been selected because a color-histogram
                comparison found sufficiently high similarity.

                Therefore, similar clothing colors are already expected and must
                NOT be the main reason for deciding SAME.

                Compare additional clearly visible person-specific characteristics:
                - hair and hairstyle
                - body build and proportions
                - clothing design and cut
                - logos, prints, stripes and patterns
                - trousers or shorts
                - shoes
                - bags and accessories
                - facial appearance, if clearly visible
                - other distinctive characteristics

                Ignore:
                - background
                - position in the image
                - camera angle
                - lighting differences
                - scale differences

                If a characteristic is hidden, blurry, cropped or uncertain,
                treat it as UNKNOWN.

                A SAME decision should be supported by multiple compatible
                characteristics beyond color similarity.

                A DIFFERENT decision should require clear contradictory evidence.

                Give one short reason.

                Then output exactly:

                REASON: <one short sentence>
                FINAL: SAME

                or

                REASON: <one short sentence>
                FINAL: DIFFERENT
                """

        content = []

        for path in temp_paths:
            content.append({"type": "image", "image": path})

        content.append({"type": "text","text": prompt})

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = qwen_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

        inputs = inputs.to(qwen_model.device)

        with torch.inference_mode():

            generated_ids = qwen_model.generate(**inputs, max_new_tokens=120, do_sample=False)

        generated_ids_trimmed = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]

        response = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        print("\nQwen-Antwort:")
        print(response)

        response_upper = response.upper()

        if "FINAL: SAME" in response_upper:
            return "SAME"

        if "FINAL: DIFFERENT" in response_upper:
            return "DIFFERENT"

        return None

    finally:

        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass
    


# -----------------------    
# Relinking
#------------------------

# Zuordnung und Verwaltung der Track-IDs
track_to_target = {}                 # ByteTrack-ID -> finale Target-ID
active_track_ids_last_frame = set()  # Menge der Track-IDs, die im letzten Frame aktiv waren
lost_tracks = {}                     # Informationen zu aktuell verlorenen Tracks
last_person_boxes = {}               #letzte bekannte Personenbox pro Track
track_first_seen = {}                # erstes Auftreten einer Track-ID
track_last_seen = {}                 # letztes Auftreten einer Track-ID
used_target_ids = set()              # bereits vergebene Target-IDs

# Re-Linking-Zustand
pending_track_frames = {}             # Anzahl beobachteter Frames vor der Zuordnung (für bessere Histogramme)
matched_lost_tracks = set()           # bereits zugeordnete verlorene Tracks
relinked_track_ids = set()            # erfolgreich re-gelinkte neue Track-IDs


#Zeitliche Grenzen
MIN_LOST_FRAMES_FOR_RELINKING = 2     # Mindestanzahl an Frames, die ein Track verschwunden sein muss, bevor er für Relinking in Frage kommt
MAX_LOST_FRAMES_FOR_RELINKING = 200   # Nach dieser Anzahl an Frames wird ein verlorener Track nicht mehr berücksichtigt.
MAX_TRACK_MEMORY_FRAMES = 30          # Speicherdauer nicht mehr benötigter Track-Daten
MIN_SHORT_TRACK_FRAMES = 5            # Mindestlänge für kurze Tracks

#Histogram Modell
person_histogram_models = {}   # geglättetes Histogramm
HISTOGRAM_ALPHA = 0.15         # Gewicht des aktuellen Histogramms beim Aktualisieren des geglätteten Modells.
MIN_SIMILARITY_MARGIN = 0.08   # Mindestabstand zwischen bestem und zweitbestem Histogramm-Kandidaten.


#Positionsüberprüfung
MAX_POSITION_DISTANCE_FACTOR = 1.5 # Maximal erlaubte Positionsänderung relativ zur Größe der alten Personenbox.

# Trackingdaten für die spätere Videoausgabe
tracking_results = []           # fertigen Einträge mit finalen Target-IDs.
pending_tracking_results = {}   # hält die ersten Frames einer neuen ByteTrack-ID zurück, solange noch nicht feststeht, ob und wie Qwen sie re-linkt.



def finalize_pending_tracking_results(track_id, target_id):
    """Übernimmt ausstehende Trackingdaten rückwirkend mit der finalen Target-ID."""

    pending_entries = pending_tracking_results.pop(track_id, [])

    for entry in pending_entries:
        tracking_results.append({**entry, "target_id": int(target_id)})
        

# Histogramm und Kandidatenauswahl
# -------------------------------------------------

def get_new_target_id(track_id):
    """Gibt eine noch nicht verwendete Target-ID zurück."""
    
    if track_id not in used_target_ids:
        return track_id

    new_target_id = max(used_target_ids, default=0) + 1

    while new_target_id in used_target_ids:
        new_target_id += 1

    return new_target_id



def get_relinking_candidates(frame_idx):
    """Liefert verlorene Tracks innerhalb des zulässigen Zeitfensters."""
    
    candidates = {}
    for lost_track_id, lost_data in lost_tracks.items():
        frame_since_lost = (frame_idx - lost_data["last_seen"])
        
        if MIN_LOST_FRAMES_FOR_RELINKING <= frame_since_lost <= MAX_LOST_FRAMES_FOR_RELINKING:
            candidates[lost_track_id] = lost_data
            
    return candidates



def calculate_histogram(frame, person_box):
    """Berechnet ein Histogramm aus dem Oberkörperbereich."""
    
    x1, y1, x2, y2 = person_box
    frame_height, frame_width = frame.shape[:2]
    
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)
    
    person_width = x2 - x1
    person_height = y2 - y1
    
    if person_width <= 0 or person_height <= 0:
        return None
    
    roi_y1 = y1 + int(person_height * 0.20) # nur der obere Teil der Personenbox wird für das Histogramm verwendet
    roi_y2 = y1 + int(person_height * 0.65) 
    
    roi_x1 = x1 + int(person_width * 0.20)
    roi_x2 = x2 - int(person_width * 0.20)
    
    person_roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if person_roi.size == 0:
        return None
    
    hsv_roi= cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV) 
    histogram = cv2.calcHist([hsv_roi], [0, 1], None, [32, 32], [0, 180, 0, 256])
    
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
    
    return histogram



def compare_person_histograms(histogram_a, histogram_b):
    """Vergleicht zwei Histogramme anhand ihrer Korrelation."""
    
    if histogram_a is None or histogram_b is None:
        return None

    similarity = cv2.compareHist(
        histogram_a,
        histogram_b,
        cv2.HISTCMP_CORREL
    )

    return float(similarity)



def update_histogram_models(track_id, current_histogram):
    """Aktualisiert das geglättete Histogramm eines Tracks."""
    
    if current_histogram is None:
        return
    
    if track_id not in person_histogram_models:
        person_histogram_models[track_id] = current_histogram.copy()
        return
    
    old_histogram = person_histogram_models.get(track_id)
    
    updated_histogram = ((1.0 - HISTOGRAM_ALPHA) * old_histogram + HISTOGRAM_ALPHA * current_histogram)
    
    cv2.normalize(updated_histogram, updated_histogram, 0, 1, cv2.NORM_MINMAX)
    
    person_histogram_models[track_id]= updated_histogram



def calculate_position_distance(box_a, box_b):
    """Berechnet die Distanz zwischen den Mittelpunkten zweier Boxen."""
    
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
    center_ax = (ax1 + ax2) / 2
    center_ay = (ay1 + ay2) / 2
    
    center_bx = (bx1 + bx2) / 2
    center_by = (by1 + by2) / 2
    
    return np.hypot(center_ax - center_bx, center_ay - center_by)



def position_is_plausible (old_person_box, new_person_box):
    """Prüft, ob die Positionsänderung für ein Re-Linking plausibel ist."""
    
    old_x1, old_y1, old_x2, old_y2 = old_person_box
    
    old_width = old_x2 - old_x1
    old_height = old_y2 - old_y1
    
    reference_size = max(old_width, old_height, 1)
    
    distance = calculate_position_distance(old_person_box, new_person_box)
    
    max_distance = MAX_POSITION_DISTANCE_FACTOR * reference_size
    
    return distance <= max_distance

    

def find_matching_target(track_id, current_histogram, current_person_box, frame_idx):
    """Sucht geeignete verlorene Tracks für eine VLM-Verifikation."""
    
    candidates = get_relinking_candidates(frame_idx)
    candidates_scores = []
    
    first_new_track_frame = track_first_seen.get(track_id, frame_idx)
    
    for lost_track_id, lost_data in candidates.items():
        
        # Bereits verwendete verlorene Tracks überspringen
        if lost_track_id in matched_lost_tracks:
            continue
        
        
        # Neue Tracks werden zunächst mehrere Frames beobachtet und anschließend rückwirkend
        # einer Target-ID zugeordnet. Ein verlorener Track kommt daher nur infrage, wenn er 
        # bereits vor dem ersten Auftreten des neuen Tracks verschwunden war.
        if first_new_track_frame <= lost_data["last_seen"]:
            continue

        
        lost_histogram = lost_data["histogram"]
        if lost_histogram is None:
            continue
        
        # Bei Tracks, die nicht am Bildrand verschwunden sind, muss die neue Position zusätzlich plausibel sein
        if not lost_data["left_from_border"]:
            if not position_is_plausible(lost_data["person_box"], current_person_box):
                #print(f"Kandidat BT {lost_track_id} wurde wegen unplausibler Position ausgeschlossen.")
                continue
            
        
            
        similarity = compare_person_histograms(current_histogram, lost_histogram)
        
        if similarity is None:
            continue
        
        candidates_scores.append((similarity, lost_track_id, lost_data["target_id"]))
        
    if not candidates_scores:
        return None
    
    # Kandidaten nach Histogramm-Ähnlichkeit sortieren
    candidates_scores.sort(key= lambda candidate: candidate[0], reverse=True)
    
    print("Ranking der Re-Linking-Kandidaten:")

    for similarity, lost_track_id, target_id in candidates_scores:
        print(
            f"  BT {lost_track_id} -> Target {target_id}: "
            f"{similarity:.3f}"
        )
    
    best_similarity, best_lost_track_id, best_target_id = candidates_scores[0]
    
    # Mindestähnlichkeit prüfen
    if best_similarity < MIN_HISTOGRAM_SIMILARITY:
        print (
            f"Kein Re-Linking: Beste Ähnlichkeit "
            f"{best_similarity:.3f} liegt unter "
            f"{MIN_HISTOGRAM_SIMILARITY:.3f}."
        )
        return None
    
    matches = [
        {"lost_track_id": best_lost_track_id,
        "target_id": best_target_id,
        "similarity": best_similarity}]

    # Wenn der zweitbeste Kandidat ähnlich gut ist, soll Qwen auch diesen Kandidaten überprüfen.
    if len(candidates_scores) > 1:
        second_similarity, second_lost_track_id, second_target_id = candidates_scores[1]

        similarity_margin = best_similarity - second_similarity

        if (second_similarity >= MIN_HISTOGRAM_SIMILARITY and similarity_margin < MIN_SIMILARITY_MARGIN):
            print(
                f"Zwei ähnliche Kandidaten gefunden: "
                f"{best_similarity:.3f} und {second_similarity:.3f}"
            )

            matches.append({
                "lost_track_id": second_lost_track_id,
                "target_id": second_target_id,
                "similarity": second_similarity
            })

    return matches        




def assign_target_id(track_id, person_box, current_histogram, frame_idx):
    """
    Bestimmt die finale Target-ID einer ByteTrack-ID.

    Bereits bekannte Track-IDs behalten ihre Zuordnung. Neue Track-IDs
    werden zunächst über mehrere Frames beobachtet. Nach Ablauf der
    Pending-Phase wird einmalig versucht, sie mit einem verlorenen Track
    zu verknüpfen.
    """

    # Die ByteTrack-ID wurde bereits endgültig zugeordnet.
    if track_id in track_to_target:
        return track_to_target[track_id]

    # Anzahl der bisher sichtbaren Frames dieser neuen ID erhöhen.
    pending_track_frames[track_id] = (pending_track_frames.get(track_id, 0) + 1)

    pending_frames = pending_track_frames[track_id]

    # Während der Pending-Phase noch keine endgültige Zuordnung speichern
    if pending_frames < MIN_FRAMES_BEFORE_RELINKING:
        return track_id

    # Nach Ablauf der Pending-Phase einmalig nach einem passenden verlorenen Track suchen (bzw. zwei bei ähnlich guten Histogramm-Kandidaten)
    matches = find_matching_target(track_id, current_histogram, person_box, frame_idx)

    relink_successful = False

    if matches is not None:

        for match in matches:
            target_id = match["target_id"]
            lost_track_id = match["lost_track_id"]

            old_reference_frames = lost_tracks[lost_track_id].get("reference_frames",[])

            new_reference_frames = reference_frames.get(track_id,[])

            vision_result = compare_reference_frames_with_qwen(old_reference_frames, new_reference_frames)

            print(
                f"Qwen-Vergleich BT {lost_track_id} -> "
                f"BT {track_id}: {vision_result}"
            )

            if vision_result != "SAME":
                continue

            track_to_target[track_id] = target_id
            used_target_ids.add(target_id)

            track_to_target.pop(lost_track_id, None)
            lost_tracks.pop(lost_track_id, None)
            matched_lost_tracks.add(lost_track_id)
            relinked_track_ids.add(track_id)

            print(
                f"Re-Linking erfolgreich: "
                f"BT {track_id} -> Target {target_id} "
                f"(vorherige BT-ID {lost_track_id}, "
                f"Histogramm {match['similarity']:.3f}, "
                f"Qwen SAME)"
            )

            relink_successful = True
            break
    
        
        if not relink_successful:
            new_target_id = get_new_target_id(track_id)

            track_to_target[track_id] = new_target_id
            used_target_ids.add(new_target_id)

            print(
                f"Keine Qwen-Zuordnung bestätigt: "
                f"BT {track_id} -> Target {new_target_id}"
            )
    else:
        # Kein geeigneter Histogramm-Kandidat vorhanden
        new_target_id = get_new_target_id(track_id)

        track_to_target[track_id] = new_target_id
        used_target_ids.add(new_target_id)

        print(
            f"Neue Person bestätigt: "
            f"BT {track_id} -> Target {new_target_id}"
        )



    # Die Wartephase ist abgeschlossen
    pending_track_frames.pop(track_id, None)

    return track_to_target[track_id]


# -------------------------------------------------
# Kalman-Filter zur Stabilisierung der Gesichtsboxen
# -------------------------------------------------

# Kalman-Zustand pro Track-ID
face_kalman_filters = {}
face_missing_frames = {}
last_face_sizes = {}
face_detection_counts = {}

# Kalman-Parameter
MAX_MISSING_FACE_FRAMES = 4
MIN_FACE_DETECTIONS_FOR_KALMAN = 4


def create_face_kalman_filter(face_box):
    """Initialisiert einen Kalman-Filter für den Mittelpunkt einer Gesichtsbox."""
    
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
    """Aktualisiert den Kalman-Filter anhand einer erkannten Gesichtsbox."""
    
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
    """Sagt die nächste Gesichtsbox anhand des Kalman-Filters voraus."""
    
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


# -------------------------------------------------
# Auswahl des passenden Gesichts
# -------------------------------------------------
    

def calculate_face_distance(face, x1, y1, last_center_x, last_center_y):
    """Berechnet die Distanz eines Gesichts zur zuletzt bekannten Position."""

    local_x1, local_y1, local_x2, local_y2 = face.bbox.astype(int)

    center_x = x1 + (local_x1 + local_x2) / 2
    center_y = y1 + (local_y1 + local_y2) / 2

    return np.hypot(
        center_x - last_center_x,
        center_y - last_center_y
    )
    
def calculate_initial_face_score(face, roi_x1, roi_y1, person_box):
    """Bewertet ein Gesicht anhand von Konfidenz und Position innerhalb der Personenbox."""
    
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

# -------------------------------------------------
# Gesichtserkennung innerhalb der Personenbox
# -------------------------------------------------

last_face_boxes = {}

def detect_face(frame, person_box, face_detector, track_id):
    """Erkennt das passende Gesicht innerhalb einer Personenbox."""
    
    x1, y1, x2, y2 = person_box

    frame_height, frame_width = frame.shape[:2]

    # Personen-ROI am Bildrand leicht vergrößern
    if touches_frame_border(person_box, frame):
        person_width = x2 - x1
        person_height = y2 - y1

        x1 -= int(person_width * 0.20)
        x2 += int(person_width * 0.20)
        y1 -= int(person_height * 0.20)
        y2 += int(person_height * 0.05)
        
    # ROI auf die Bildgrenzen beschränken
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
    
    # Bei nur einem erkannten Gesicht ist keine weitere Auswahl nötig
    if len(faces) == 1:
        best_face = faces[0]
    
    # Ohne vorherige Gesichtsposition anhand von Konfidenz und erwarteter Kopfposition auswählen
    elif track_id not in last_face_boxes:
        best_face = max(faces, key=lambda face: calculate_initial_face_score(face, x1, y1, person_box))


    # Bei vorhandener Historie das Gesicht wählen, das der zuletzt bekannten Gesichtsposition am nächsten liegt
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
        
    # SCRFD-Koordinaten aus der ROI auf den gesamten Frame übertragen
    face_x1, face_y1, face_x2, face_y2 = best_face.bbox.astype(int)

    face_x1 += x1
    face_y1 += y1
    face_x2 += x1
    face_y2 += y1

    # Gesichtsbox auf die Bildgrenzen beschränken
    face_x1 = max(0, face_x1)
    face_y1 = max(0, face_y1)
    face_x2 = min(frame_width, face_x2)
    face_y2 = min(frame_height, face_y2)

    face_box = (face_x1, face_y1, face_x2, face_y2)
    last_face_boxes[track_id] = face_box

    return face_box



# -------------------------------------------------
# Gesichtsanonymisierung
# -------------------------------------------------

def anonymize_face(frame, face_box):
    """Anonymisiert eine Gesichtsregion durch größenabhängigen Gaussian Blur."""

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

# -------------------------------------------------
# Frame-Verarbeitung
# -------------------------------------------------

def process_frame(frame, model, face_detector, frame_idx):
    
    global active_track_ids_last_frame
    
    original_frame = frame.copy()
    
    active_track_ids_current_frame = set() 

    # Persondetektion und Tracking
    results = model.track(frame, persist=True, tracker="bytetrack_custom.yaml", verbose=False)
    
    boxes = results[0].boxes
    
    for box in boxes:
        
        #Track-Informationen auslesen
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int) # Koordinaten der Box
        cls = int(box.cls[0]) # Klasse als Integer
        conf = float(box.conf[0])# Konfidenz als Float
        
        if cls != 0: # Klasse 0 entspricht "person" im COCO-Datensatz
            continue    
        
        if box.id is None: #Objekte ohne Track-ID überspringen
            continue
        
        track_id = int(box.id[0])
        
        if track_id not in track_first_seen:
            track_first_seen[track_id] = frame_idx
        
        active_track_ids_current_frame.add(track_id)  # Track-ID als aktiv markieren
        
        # Falls ByteTrack dieselbe ID wieder aufnimmt, wird sie nicht länger als verloren geführt
        if track_id in lost_tracks:
            lost_tracks.pop(track_id, None)
            print(f"Track-ID {track_id} wurde wiedergefunden und aus den verlorenen Tracks entfernt.")
        
        # Zeitpunkt speichern, zu dem der Track zuletzt sichtbar war
        track_last_seen[track_id] = frame_idx
        
        person_box = (x1, y1, x2, y2)
        
        last_person_boxes[track_id] = person_box  # Speichern der zuletzt bekannten Personenbox 
        
        # Referenzframes für die spätere VLM-Verifikation aktualisieren
        update_reference_frames(track_id, original_frame, person_box, conf, frame_idx)
        
        
        # -------------------------------------------------
        # Histogramm berechnen bzw. aktualisieren
        # -------------------------------------------------
        
        current_histogram = None
        
        if track_id in track_to_target:
            # Bereits bestätigte Tracks aktualisieren ihr Histogrammmodell
            current_histogram = calculate_histogram(original_frame, person_box)
            update_histogram_models(track_id, current_histogram)
            
        else:
            # Bei neuen Tracks wird das Histogramm erst am Ende der Pending-Phase für den Re-Linking-Versuch benötigt
            next_pending_frame = (pending_track_frames.get(track_id,0 ) +1)
            if next_pending_frame >= MIN_FRAMES_BEFORE_RELINKING:
                current_histogram = calculate_histogram(original_frame, person_box)
                
        # -------------------------------------------------
        # Target-ID bestimmen
        # -------------------------------------------------
        
        target_id = assign_target_id(track_id, person_box, current_histogram, frame_idx)
        
        tracking_entry = {
            "frame": int(frame_idx),
            "track_id": int(track_id),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)}
        
        if track_id in track_to_target:
        # Target-ID ist jetzt endgültig bekannt

            if track_id in pending_tracking_results:
                finalize_pending_tracking_results(track_id, target_id)

            tracking_results.append({**tracking_entry, "target_id": int(target_id)})

        else:
            # Target-ID ist während der Pending-Phase noch nicht endgültig
            pending_tracking_results.setdefault(track_id, []).append(tracking_entry)
        
        
        # Falls für einen neuen Track erstmals ein Histogramm berechnet wurde, wird es als Ausgangsmodell gespeichert
        if (track_id in track_to_target and current_histogram is not None and track_id not in person_histogram_models):
            person_histogram_models[track_id] = current_histogram.copy()
        
        
        # --------------------------------------
        # Gesichtsdetektion und Anonymisierung
        # --------------------------------------
        face_box = detect_face(original_frame, person_box, face_detector, track_id)

        if face_box is not None:
                
            # Anzahl erfolgreicher Gesichtsdetektionen für diesen Track erhöhen
            face_detection_counts[track_id] = (face_detection_counts.get(track_id, 0) + 1)
            
            # Aktuelle Größe der Gesichtsbox bestimmen
            x1_face, y1_face, x2_face, y2_face = face_box
            face_size = (x2_face - x1_face, y2_face - y1_face)

            # Beim ersten erkannten Gesicht Kalman-Filter initialisieren, anschließend mit jeder neuen Detektion aktualisieren
            if track_id not in face_kalman_filters:
                face_kalman_filters[track_id] = create_face_kalman_filter(face_box)
            else:
                face_size = update_face_kalman_filter(face_kalman_filters[track_id],face_box)
                
                
            # Aktuelle Gesichtsgröße speichern und Zähler fehlender Detektionen nach erfolgreicher Erkennung zurücksetzen
            last_face_sizes[track_id] = face_size
            face_missing_frames[track_id] = 0

            anonymize_face(frame, face_box)

        else:

            # Anzahl aufeinanderfolgender Frames ohne Gesichtsdetektion sowie bisherige erfolgreiche Detektionen abrufen
            missing_frames = face_missing_frames.get(track_id, 0)
            detection_count = face_detection_counts.get(track_id, 0)


            # Kalman-Vorhersage nur verwenden, wenn zuvor ausreichend Gesichtsdetektionen vorlagen und die maximale Anzahl fehlender Frames noch nicht erreicht wurde
            if (
                track_id in face_kalman_filters
                and track_id in last_face_sizes
                and detection_count >= MIN_FACE_DETECTIONS_FOR_KALMAN
                and missing_frames < MAX_MISSING_FACE_FRAMES
            ):
                predicted_face_box = predict_face_box(face_kalman_filters[track_id], last_face_sizes[track_id], frame)

                # Vorhergesagte Gesichtsposition ebenfalls anonymisieren
                if predicted_face_box is not None:
                    anonymize_face(frame, predicted_face_box)

                face_missing_frames[track_id] = missing_frames + 1
         
        
        
    # -------------------------------------------------
    # Verschwundene Tracks speichern
    # -------------------------------------------------
    disappeared_track_ids = (active_track_ids_last_frame- active_track_ids_current_frame)

    for disappeared_track_id in disappeared_track_ids:
        if disappeared_track_id not in last_person_boxes:
            continue
        
        # Falls der Track vor Abschluss der Pending-Phase verschwindet,
        # erhält er ohne Re-Linking eine eigene finale Target-ID.
        if disappeared_track_id not in track_to_target:
            
            track_length = ( track_last_seen[disappeared_track_id] - track_first_seen[disappeared_track_id] + 1 )
            
            if track_length < MIN_SHORT_TRACK_FRAMES:
                pending_tracking_results.pop(disappeared_track_id, None)
                pending_track_frames.pop(disappeared_track_id, None)

                print(
                    f"Kurzer Track verworfen: "
                    f"BT {disappeared_track_id} "
                    f"({track_length} Frames)"
                )

                continue
            
            
            new_target_id = get_new_target_id(disappeared_track_id)

            track_to_target[disappeared_track_id] = new_target_id
            used_target_ids.add(new_target_id)
            
            # Bisherige Pending-Frames mit der finalen ID übernehmen
            finalize_pending_tracking_results(disappeared_track_id, new_target_id)

            print(
                f"Kurzer Track abgeschlossen: "
                f"BT {disappeared_track_id} -> Target {new_target_id}")

        lost_tracks[disappeared_track_id] = {
            "target_id": track_to_target[disappeared_track_id],
            "last_seen": frame_idx - 1,
            "person_box": last_person_boxes[disappeared_track_id],
            "histogram": person_histogram_models.get(disappeared_track_id),
            "left_from_border": touches_frame_border(last_person_boxes[disappeared_track_id], frame),
            "reference_frames": reference_frames.get(disappeared_track_id, [])
        }
        
        pending_track_frames.pop(disappeared_track_id, None)


    active_track_ids_last_frame = active_track_ids_current_frame
    
    
    
    # Überprüfung auf abgelaufene verlorene Tracks und Entfernen aus dem Speicher
    expired_lost_track_ids = [
        lost_track_id 
        for lost_track_id, lost_data in lost_tracks.items()
        if (frame_idx - lost_data["last_seen"] > MAX_LOST_FRAMES_FOR_RELINKING)
    ]

    for expired_track_id in expired_lost_track_ids:
        expired_target_id = lost_tracks[expired_track_id]["target_id"]

        lost_tracks.pop(expired_track_id, None)

        print(
            f"Verlorene Track-ID {expired_track_id} "
            f"mit Target-ID {expired_target_id} wurde "
            f"aus dem Re-Linking-Speicher entfernt."
        )
        
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
        person_histogram_models.pop(track_id, None)
        pending_track_frames.pop(track_id, None)
        track_first_seen.pop(track_id, None)
        reference_frames.pop(track_id, None)

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

    if SAVE_VIDEO:
        video_writer.write(processed_frame)

    
    
# ----------------------------------------
# noch offene Pending-Tracks finalisieren
# ----------------------------------------
       
for track_id in list(pending_tracking_results.keys()):
    new_target_id = get_new_target_id(track_id)

    track_to_target[track_id] = new_target_id
    used_target_ids.add(new_target_id)

    finalize_pending_tracking_results(track_id, new_target_id)
    


# ----------------------
# JSON Datei erstellen
# ----------------------
tracking_results.sort(key=lambda entry: (entry["frame"], entry["target_id"]))

os.makedirs("tracking_results", exist_ok=True)

tracking_output_path = (f"tracking_results/{video_name}_qwen_tracks.json")

with open(tracking_output_path, "w") as f:
    json.dump(tracking_results, f, indent=2)

print(f"Trackingdaten gespeichert: {tracking_output_path}")


cap.release()

if SAVE_VIDEO:
    video_writer.release()
    

if processed_frames > 0:

    average_processing_time = (total_processing_time / processed_frames)

    processing_fps = (1.0 / average_processing_time)

    print(f"\nVerarbeitete Frames: {processed_frames}")

    print(f"Durchschnittliche Zeit pro Frame: {average_processing_time:.4f} s")
    
    print(f"Verarbeitungsgeschwindigkeit: {processing_fps:.2f} FPS")

# Pipeline mit Relinking und lokalen VLM

import time
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import torchvision
from insightface.app import FaceAnalysis
import base64

from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
import cv2
import os
import tempfile

import gc
import mlx.core as mx


QWEN_MODEL_NAME = "mlx-community/Qwen3-VL-4B-Instruct-4bit"

qwen_model, qwen_processor = load(QWEN_MODEL_NAME)
qwen_config = load_config(QWEN_MODEL_NAME)


def compare_reference_frames_with_qwen(old_frames, new_frames):

    if len(old_frames) < 1 or len(new_frames) < 1:
        return None

    # Für den aktuellen Aufbau maximal zwei alte und zwei neue Bilder
    selected_old = old_frames[:2]
    selected_new = new_frames[:2]

    # MLX-VLM arbeitet hier am einfachsten mit Bilddateien.
    temp_paths = []

    try:
        # Alte Referenzen speichern
        for i, reference in enumerate(selected_old):
            path = os.path.join(
                tempfile.gettempdir(),
                f"qwen_old_{i}.jpg"
            )

            cv2.imwrite(path, reference["crop"])
            temp_paths.append(path)

        # Neue Referenzen speichern
        for i, reference in enumerate(selected_new):
            path = os.path.join(
                tempfile.gettempdir(),
                f"qwen_new_{i}.jpg"
            )

            cv2.imwrite(path, reference["crop"])
            temp_paths.append(path)

        num_old = len(selected_old)
        num_new = len(selected_new)

        prompt = f"""
You are performing a person re-identification verification.

The first {num_old} image(s) show the SAME physical person from a previously lost track.
The following {num_new} image(s) show a newly detected track.

Use ALL old reference images together and compare them with ALL new reference images.

IMPORTANT CONTEXT:

The old and new tracks have already been selected as possible matches because
a previous color-histogram comparison found relatively high color similarity.

Therefore, similar dominant colors and similar clothing colors are already
expected and provide only weak additional evidence for SAME.

Your task is to go BEYOND this known color similarity and search for additional
person-specific visual evidence that can confirm or reject the proposed match.

Compare all clearly visible person-specific characteristics, including:
- facial appearance, if clearly visible
- glasses
- hair and hairstyle
- body build and body proportions
- exact clothing design, shape and cut
- logos, prints, stripes and patterns
- trousers or shorts
- shoes
- bags and accessories
- other distinctive visible characteristics

Only use a characteristic when it is clearly visible and reliably comparable.
If it is hidden, blurry, cropped, too small or uncertain, treat it as UNKNOWN.

A small detail that appears in only one image must not automatically be treated
as a contradiction, because it may simply not be visible in the other image.

Do NOT classify SAME mainly because clothing colors match.

A SAME decision should be supported by multiple compatible person-specific
characteristics beyond the known color similarity.

Do NOT classify DIFFERENT because of one small or uncertain detail.

A DIFFERENT decision should normally require at least TWO independent,
clearly visible and reliable person-specific contradictions.

Alternatively, DIFFERENT may be chosen if ONE major characteristic provides
a very strong and unambiguous contradiction.

Small or changeable details such as small logos, sock details, shoe details,
glasses, small accessories or minor hairstyle differences must NOT alone
determine the final identity decision.

Differences caused only by lighting, viewpoint, pose, scale or partial
occlusion are NOT identity differences.

Completely IGNORE:
- background and surroundings
- scene content
- camera angle itself
- position within the image
- distance from the camera

Base the final decision on the overall person-specific evidence.

Give ONE short sentence explaining the most decisive reliable evidence.

Then output exactly:

REASON: <one short sentence>
FINAL: SAME

or

REASON: <one short sentence>
FINAL: DIFFERENT
"""

        formatted_prompt = apply_chat_template(
            qwen_processor,
            qwen_config,
            prompt,
            num_images=len(temp_paths)
        )

        result = generate(
            qwen_model,
            qwen_processor,
            formatted_prompt,
            image=temp_paths,
            max_tokens=120,
            verbose=False
        )

        response = result.text.strip()

        del result
        gc.collect()

        print("Qwen-Antwort:")
        print(response)

        # Nur die finale Entscheidung für die Pipeline zurückgeben
        response_upper = response.upper()

        if "FINAL: SAME" in response_upper:
            return "SAME"

        if "FINAL: DIFFERENT" in response_upper:
            return "DIFFERENT"

        return None

    finally:
        # Temporäre Bilder wieder löschen
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


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
video_path = "Testvideos/ähnliche_farben.mp4"

cap = cv2.VideoCapture(video_path)

# Dateiname ohne Ordner und Endung
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Videoeigenschaften abrufen
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"fps: {fps}")

# VideoWriter-Objekt erstellen, um das Ergebnisvideo zu speichern
save_video = False 
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = None
OUTPUT_FPS = 45.0

if save_video:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        f"Ausgabevideos/{video_name}_erweiterung.mp4",
        fourcc,
        OUTPUT_FPS,
        (frame_width, frame_height)
    )

# Selektives Tracking
show_all_tracks = False # Wenn True, werden alle Tracks angezeigt, False: nur ausgewählten

selected_track_ids = set() # Set zum Speichern der ausgewählten Track-IDs

current_tracks = [] # Welche IDs sind gerade wo sichtbar? Koordinaten werden dann bei Mausklick verglichen. Aufbau: (track_id, x1, y1, x2, y2)


# Mausklick-Funktion
def mouse_click(event, x, y, flags, param):

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Prüfen, ob der Klick innerhalb einer aktuellen Personenbox liegt
    for track_id, target_id, x1, y1, x2, y2 in current_tracks:

        if x1 <= x <= x2 and y1 <= y <= y2:

            if target_id in selected_track_ids:
                selected_track_ids.remove(target_id)
                print(f"Target-ID {target_id} wurde abgewählt.")

            else:
                selected_track_ids.add(target_id)
                print(f"Target-ID {target_id} wurde ausgewählt.")

            break

window_name = "Processed Frame"

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_click)



# Farben für Track-IDs definieren

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
    return ID_COLORS[track_id % len(ID_COLORS)]

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
    
 
#----------------------------
# Relinking mit Sprachmodell
#----------------------------    

reference_frames = {}

def calculate_reference_score(person_box, confidence, frame):
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
    x1, y1, x2, y2 = person_box

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return

    score = calculate_reference_score(
        person_box,
        confidence,
        frame
    )

    candidate = {
        "frame_idx": frame_idx,
        "crop": crop.copy(),
        "score": score
    }

    if track_id not in reference_frames:
        reference_frames[track_id] = [candidate]
        return

    stored = reference_frames[track_id]

    # Erstes Bild vorhanden, zweites aber noch nicht:
    # nur aufnehmen, wenn genügend zeitlicher Abstand besteht
    if len(stored) == 1:
        if abs(frame_idx - stored[0]["frame_idx"]) >= 8:
            stored.append(candidate)

        elif score > stored[0]["score"]:
            stored[0] = candidate

        return

    # Von den bereits gespeicherten Referenzen die schlechtere suchen
    worst_index = min(
        range(len(stored)),
        key=lambda i: stored[i]["score"]
    )

    other_index = 1 - worst_index

    # Ersatz nur, wenn der neue Frame besser ist und
    # genügend Abstand zur anderen Referenz besitzt
    if (
        score > stored[worst_index]["score"]
        and abs(frame_idx - stored[other_index]["frame_idx"]) >= 8
    ):
        stored[worst_index] = candidate
        

def save_reference_test(
    old_frames,
    new_frames,
    old_track_id,
    new_track_id
):
    os.makedirs("reference_tests", exist_ok=True)

    for i, reference in enumerate(old_frames):
        cv2.imwrite(
            f"reference_tests/old_{old_track_id}_{i}.jpg",
            reference["crop"]
        )

    for i, reference in enumerate(new_frames):
        cv2.imwrite(
            f"reference_tests/new_{new_track_id}_{i}.jpg",
            reference["crop"]
        )
 
 
def crop_to_base64(crop):
    success, buffer = cv2.imencode(".jpg", crop)

    if not success:
        return None

    return base64.b64encode(buffer).decode("utf-8")

def crop_to_bytes(crop):
    success, buffer = cv2.imencode(".jpg", crop)

    if not success:
        return None

    return buffer.tobytes()


    
    
# -----------------------    
# Abschnitt für Relinking
#------------------------

#Variablen
track_to_target = {}  # Mapping von Track-ID zu Target-ID, z.B. {1: 2} bedeutet, dass Track 1 zu Target 2 gehört
active_track_ids_last_frame = set()  # Menge der Track-IDs, die im letzten Frame aktiv waren
lost_tracks = {}  # Mapping von verlorenen Track-IDs zu den Frames, in denen sie zuletzt gesehen wurden. Speichert zu Track ID: Target-ID, zuletzt gesehenes Frame, zuletzt bekannte Personenbox (x1, y1, x2, y2)
last_person_boxes = {}  # Mapping von Track-ID zu zuletzt bekannter Personenbox (x1, y1, x2, y2)
MIN_LOST_FRAMES_FOR_RELINKING = 2  # Mindestanzahl an Frames, die ein Track verschwunden sein muss, bevor er für Relinking in Frage kommt
MAX_LOST_FRAMES_FOR_RELINKING = 200  # Maximale Anzahl an Frames, die ein Track verschwunden sein darf, bevor er für Relinking in Frage kommt

person_histogram_models = {}  #geglättetes Histogramm
HISTOGRAM_ALPHA = 0.15

MIN_HISTOGRAM_SIMILARITY= 0.70
MIN_SCORE_MARGIN = 0.10
MIN_SIMILARITY_MARGIN = 0.08
MAX_POSITION_DISTANCE_FACTOR = 1.5 

matched_lost_tracks = set()
relinked_track_ids = set()


pending_track_frames ={}                 # paar frames abwarten bis zum Relinking, damit Histogramm aussagekräftig ist
MIN_FRAMES_BEFORE_RELINKING = 15
PENDING_HISTOGRAM_ALPHA = 0.30


# Funktion, um Kandidaten für Relinking zu erhalten
def get_relinking_candidates(frame_idx):
    candidates = {}
    for lost_track_id, lost_data in lost_tracks.items():
        frame_since_lost = (frame_idx - lost_data["last_seen"])
        
        if MIN_LOST_FRAMES_FOR_RELINKING <= frame_since_lost <= MAX_LOST_FRAMES_FOR_RELINKING:
            candidates[lost_track_id] = lost_data
            
    return candidates


#Histogramm der Personenbox berechnen
def calculate_histogram(frame, person_box):
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
    if histogram_a is None or histogram_b is None:
        return None

    similarity = cv2.compareHist(
        histogram_a,
        histogram_b,
        cv2.HISTCMP_CORREL
    )

    return float(similarity)


def update_histogram_models(track_id, current_histogram):
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
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    
    center_ax = (ax1 + ax2) / 2
    center_ay = (ay1 + ay2) / 2
    
    center_bx = (bx1 + bx2) / 2
    center_by = (by1 + by2) / 2
    
    return np.hypot(center_ax - center_bx, center_ay - center_by)

def position_is_plausible (old_person_box, new_person_box):
    old_x1, old_y1, old_x2, old_y2 = old_person_box
    
    old_width = old_x2 - old_x1
    old_height = old_y2 - old_y1
    
    reference_size = max(old_width, old_height, 1)
    
    distance = calculate_position_distance(old_person_box, new_person_box)
    
    max_distance = MAX_POSITION_DISTANCE_FACTOR * reference_size
    
    return distance <= max_distance

def find_matching_target(current_histogramm, current_person_box, frame_idx):
    candidates = get_relinking_candidates(frame_idx)
    candidates_scores = []
    
    for lost_track_id, lost_data in candidates.items():
        
        if lost_track_id in matched_lost_tracks:
            continue
        
        lost_histogram = lost_data["histogram"]
        if lost_histogram is None:
            continue
        
        if not lost_data["left_from_border"]:
            if not position_is_plausible(lost_data["person_box"], current_person_box):
                #print(f"Kandidat BT {lost_track_id} wurde wegen unplausibler Position ausgeschlossen.")
                continue
            
        
            
        similarity = compare_person_histograms(current_histogramm, lost_histogram)
        
        if similarity is None:
            continue
        
        #print(
            #f"Vergleich mit verlorenem BT {lost_track_id} "
            #f"(Target {lost_data['target_id']}): "
            #f"Histogramm-Ähnlichkeit = {similarity:.3f}, "
            #f"Randverlust = {lost_data['left_from_border']}" )
        
        candidates_scores.append((similarity, lost_track_id, lost_data["target_id"]))
        
    if not candidates_scores:
        return None
    
    candidates_scores.sort(key= lambda candidate: candidate[0], reverse=True)
    
    print("Ranking der Re-Linking-Kandidaten:")

    for similarity, lost_track_id, target_id in candidates_scores:
        print(
            f"  BT {lost_track_id} -> Target {target_id}: "
            f"{similarity:.3f}"
        )
    
    best_similarity, best_lost_track_id, best_target_id = candidates_scores[0]
    
    if best_similarity < MIN_HISTOGRAM_SIMILARITY:
        print (
            f"Kein Re-Linking: Beste Ähnlichkeit "
            f"{best_similarity:.3f} liegt unter "
            f"{MIN_HISTOGRAM_SIMILARITY:.3f}."
        )
        return None
    
    if len(candidates_scores) > 1:
        second_best_similarity = candidates_scores[1][0]
        
        similarity_margin = best_similarity - second_best_similarity
        
        if similarity_margin < MIN_SIMILARITY_MARGIN:
            #print(
                #f"Kein Re-Linking: Unterschied zwischen bestem "
                #f"und zweitbestem Kandidaten beträgt nur "
                #"{similarity_margin:.3f}."
            #)
            return None
        
    return {"lost_track_id": best_lost_track_id, "target_id": best_target_id, "similarity": best_similarity}



def assign_target_id(track_id, person_box, current_histogram, frame_idx):
    """
    Gibt die Target-ID einer ByteTrack-ID zurück.

    Bekannte ByteTrack-IDs behalten ihre vorhandene Zuordnung.
    Neue IDs werden zunächst einige Frames beobachtet.
    Anschließend wird ein Re-Linking-Versuch durchgeführt:

    1. Histogramm und Position bestimmen einen plausiblen alten Kandidaten.
    2. Gemini vergleicht die Referenzbilder der alten und neuen Track-ID.
    3. Nur wenn Gemini SAME zurückgibt, wird tatsächlich relinkt.
    """

    # Die ByteTrack-ID wurde bereits endgültig zugeordnet.
    if track_id in track_to_target:
        return track_to_target[track_id]

    # Anzahl der bisher sichtbaren Frames dieser neuen ID erhöhen.
    pending_track_frames[track_id] = (
        pending_track_frames.get(track_id, 0) + 1
    )

    pending_frames = pending_track_frames[track_id]

    # Während der Wartephase noch keine endgültige Zuordnung.
    if pending_frames < MIN_FRAMES_BEFORE_RELINKING:
        return track_id

    # Nach Ablauf der Wartephase einen Kandidaten über
    # Histogramm und Position bestimmen.
    match = find_matching_target(
        current_histogram,
        person_box,
        frame_idx
    )

    if match is not None:

        target_id = match["target_id"]
        lost_track_id = match["lost_track_id"]

        # Referenzbilder der alten Track-ID
        old_reference_frames = lost_tracks[
            lost_track_id
        ].get("reference_frames", [])

        # Referenzbilder der neuen Track-ID
        new_reference_frames = reference_frames.get(
            track_id,
            []
        )

        # Zum Testen weiterhin speichern
        save_reference_test(
            old_reference_frames,
            new_reference_frames,
            lost_track_id,
            track_id
        )

        # Zusätzliche visuelle Verifikation durch Gemini
        vision_result = compare_reference_frames_with_qwen(
            old_reference_frames,
            new_reference_frames
        )

        print(
            f"Qwen-Vergleich BT {lost_track_id} -> "
            f"BT {track_id}: {vision_result}"
        )

        # ---------------------------------
        # Gemini bestätigt den Kandidaten
        # ---------------------------------
        if vision_result == "SAME":

            track_to_target[track_id] = target_id

            # Alter Track wurde erfolgreich wiedergefunden
            lost_tracks.pop(lost_track_id, None)

            # Kandidat darf nicht erneut verwendet werden
            matched_lost_tracks.add(lost_track_id)

            relinked_track_ids.add(track_id)

            print(
                f"Re-Linking erfolgreich: "
                f"BT {track_id} -> Target {target_id} "
                f"(vorherige BT-ID {lost_track_id}, "
                f"Ähnlichkeit {match['similarity']:.3f})"
            )

        # ---------------------------------
        # Gemini lehnt den Kandidaten ab
        # ---------------------------------
        else:

            # Neue ByteTrack-ID wird als eigene Target-ID bestätigt
            track_to_target[track_id] = track_id

            print(
                f"Re-Linking durch Gemini abgelehnt: "
                f"BT {track_id} -> Target {track_id} "
                f"(Kandidat war BT {lost_track_id}, "
                f"Histogramm-Ähnlichkeit "
                f"{match['similarity']:.3f})"
            )

            # WICHTIG:
            # lost_track_id NICHT aus lost_tracks entfernen.
            # Die tatsächlich zugehörige Person könnte später
            # noch einmal erscheinen.

    else:

        # Kein geeigneter alter Track gefunden.
        # Die ByteTrack-ID wird als neue Target-ID bestätigt.
        track_to_target[track_id] = track_id

        print(
            f"Neue Person bestätigt: "
            f"BT {track_id} -> Target {track_id}"
        )

    # Wartephase für diese ByteTrack-ID ist abgeschlossen.
    pending_track_frames.pop(track_id, None)

    return track_to_target[track_id]





#--------------
# Kalman-Filter
#--------------

# Variablen für Kalman-Filter, wenn Gesicht nicht erkannt wird

face_kalman_filters = {}
face_missing_frames = {}
last_face_sizes = {}

max_missing_frames = 4

face_detection_counts = {}
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
    
    global active_track_ids_last_frame
    
    original_frame = frame.copy()
    
    current_tracks.clear()  # Liste der aktuellen Tracks für diesen Frame zurücksetzen
    
    active_track_ids_current_frame = set()  
    
    results = model.track(frame, persist=True, tracker="bytetrack_custom.yaml", verbose=False)
    
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
        
        active_track_ids_current_frame.add(track_id)  # Track-ID als aktiv markieren
        
        if track_id in lost_tracks:
            lost_tracks.pop(track_id, None)
            print(f"Track-ID {track_id} wurde wiedergefunden und aus den verlorenen Tracks entfernt.")
        
        # Zeitpunkt speichern, zu dem der Track zuletzt sichtbar war
        track_last_seen[track_id] = frame_idx
        
        person_box = (x1, y1, x2, y2)
        
        last_person_boxes[track_id] = person_box  # Speichern der zuletzt bekannten Personenbox 
        
        update_reference_frames(track_id, original_frame, person_box, conf, frame_idx)
        
        current_histogram = None
        
        if track_id in track_to_target:
            current_histogram = calculate_histogram(original_frame, person_box)
            update_histogram_models(track_id, current_histogram)
            
        else:
            next_pending_frame = (pending_track_frames.get(track_id,0 ) +1)
            if next_pending_frame >= MIN_FRAMES_BEFORE_RELINKING:
                current_histogram = calculate_histogram(original_frame, person_box)
        
        target_id = assign_target_id(track_id, person_box, current_histogram, frame_idx)
        
        if (track_id in track_to_target and current_histogram is not None and track_id not in person_histogram_models):
            person_histogram_models[track_id] = current_histogram.copy()
        
        # Track und Personenbox für Mausklick speichern
        current_tracks.append((track_id, target_id, x1, y1, x2, y2))  # Track-ID zur Liste der aktuellen Tracks hinzufügen
        
        # Gesichtsdetektion und Anonymisierung
        face_box = detect_face(original_frame, person_box, face_detector, track_id)

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
                and missing_frames < max_missing_frames
            ):
                predicted_face_box = predict_face_box(face_kalman_filters[track_id], last_face_sizes[track_id], frame)

                if predicted_face_box is not None:
                    anonymize_face(frame, predicted_face_box)

                    # Nur zum Testen
                    cv2.rectangle(
                        frame,
                        (predicted_face_box[0], predicted_face_box[1]),
                        (predicted_face_box[2], predicted_face_box[3]),
                        (0, 165, 255),
                        2
                    )

                face_missing_frames[track_id] = missing_frames + 1
                
                
        if track_id not in track_to_target:
            continue   
            
        # Selektive Darstellung der Personenboxen
        
        if (not show_all_tracks) and (target_id not in selected_track_ids):
            continue # Wenn nur ausgewählte Tracks angezeigt werden sollen und die aktuelle Track-ID nicht ausgewählt ist, überspringen
        
            
        if track_id in relinked_track_ids:
            label = f"BT {track_id} | T {target_id} | R"
        
        else:
            label = f"BT {track_id} | T {target_id}"
        
        color = get_color_for_id(target_id)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5) # Farbige Box um Personen zeichnen , die 5 steht für die Dicke der Box
            
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        
        
    # Überprüfung auf verschwundene Tracks und Speicherung der letzten bekannten Positionen
    disappeared_track_ids = (active_track_ids_last_frame- active_track_ids_current_frame)

    for disappeared_track_id in disappeared_track_ids:
        if disappeared_track_id not in last_person_boxes:
            continue

        lost_tracks[disappeared_track_id] = {
            "target_id": track_to_target.get(disappeared_track_id,disappeared_track_id),
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

    cv2.imshow(
        window_name,
        processed_frame
    )

    key = cv2.waitKey(1) & 0xFF

    # ESC zum Beenden
    if key == 27:
        break

    # Zwischen allen und ausgewählten Tracks wechseln
    if key == ord("s"):
        show_all_tracks = not show_all_tracks

        if show_all_tracks:
            print("Alle Tracks werden angezeigt.")
        else:
            print("Nur ausgewählte Tracks werden angezeigt.")

    # Auswahl löschen
    if key == ord("c"):
        selected_track_ids.clear()
        print("Alle ausgewählten Track-IDs wurden gelöscht.")


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
    
    
gc.collect()

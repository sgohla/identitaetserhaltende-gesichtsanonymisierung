# MOT EVAL mit Histogramm + VLM Relinking

import time
import os
import cv2
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import torch
import tempfile
from transformers import (Qwen2_5_VLForConditionalGeneration, AutoProcessor)
from qwen_vl_utils import process_vision_info


SEQUENCES = [
    "MOT17-02-FRCNN",
    "MOT17-04-FRCNN",
    "MOT17-09-FRCNN",
]

# YOLO-Modell laden
model = None


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
    
    
# -----------------------
# MOT-Evaluation
# -----------------------

mot_results_bytetrack = []   # Baseline: rohe ByteTrack-IDs
mot_results_target = []      # Erweiterung: finale Target-IDs

# Ergebnisse neuer Tracks werden bis zur endgültigen Target-ID gepuffert
pending_mot_results = {}

def finalize_pending_mot_results(track_id, target_id):
    """
    Schreibt alle bisher gepufferten Frames einer neuen ByteTrack-ID
    rückwirkend mit der endgültigen Target-ID in die MOT-Ergebnisse.
    """

    if track_id not in pending_mot_results:
        return

    for entry in pending_mot_results[track_id]:
        mot_results_target.append({
            "frame": entry["frame"],
            "id": target_id,
            "x": entry["x"],
            "y": entry["y"],
            "w": entry["w"],
            "h": entry["h"],
            "conf": entry["conf"]
        })

    del pending_mot_results[track_id]  
    
def target_id_conflicts_with_pending_frames(track_id, target_id):
    """
    Prüft, ob die gewünschte Target-ID in einem der gepufferten
    Frames des neuen Tracks bereits von einem anderen Track verwendet wird.
    """

    pending_entries = pending_mot_results.get(track_id, [])

    if not pending_entries:
        return False

    pending_frame_ids = {
        entry["frame"]
        for entry in pending_entries
    }

    for result in mot_results_target:
        if (
            result["id"] == target_id
            and result["frame"] in pending_frame_ids
        ):
            return True

    return False  
    
def save_mot_results(results, output_path):
    """
    Speichert Tracking-Ergebnisse im MOTChallenge-Format:
    frame, id, x, y, width, height, confidence, -1, -1, -1
    """

    results = sorted(
        results,
        key=lambda r: (r["frame"], r["id"])
    )

    with open(output_path, "w") as f:
        for r in results:
            f.write(
                f'{r["frame"]},'
                f'{r["id"]},'
                f'{r["x"]:.2f},'
                f'{r["y"]:.2f},'
                f'{r["w"]:.2f},'
                f'{r["h"]:.2f},'
                f'{r["conf"]:.6f},'
                f'-1,-1,-1\n'
            )
    
    
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


def compare_reference_frames_with_qwen(
    old_frames,
    new_frames
):

    if len(old_frames) < 1 or len(new_frames) < 1:
        return None

    selected_old = old_frames[:2]
    selected_new = new_frames[:2]

    temp_paths = []

    try:

        # Alte Referenzen
        for reference in selected_old:

            temp_file = tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            )

            temp_file.close()

            cv2.imwrite(
                temp_file.name,
                reference["crop"]
            )

            temp_paths.append(temp_file.name)

        # Neue Referenzen
        for reference in selected_new:

            temp_file = tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            )

            temp_file.close()

            cv2.imwrite(
                temp_file.name,
                reference["crop"]
            )

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
            content.append({
                "type": "image",
                "image": path
            })

        content.append({
            "type": "text",
            "text": prompt
        })

        messages = [
            {
                "role": "user",
                "content": content
            }
        ]

        text = qwen_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(
            messages
        )

        inputs = qwen_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to(qwen_model.device)

        with torch.inference_mode():

            generated_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=False
            )

        generated_ids_trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids
            in zip(inputs.input_ids, generated_ids)
        ]

        response = qwen_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

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

MIN_HISTOGRAM_SIMILARITY= 0.65
MIN_SCORE_MARGIN = 0.10
MIN_SIMILARITY_MARGIN = 0.08
MAX_POSITION_DISTANCE_FACTOR = 1.5 

matched_lost_tracks = set()
relinked_track_ids = set()


pending_track_frames ={}                 # paar frames abwarten bis zum Relinking, damit Histogramm aussagekräftig ist
MIN_FRAMES_BEFORE_RELINKING = 10
PENDING_HISTOGRAM_ALPHA = 0.30


track_first_seen= {}


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

def find_matching_target(track_id, current_histogramm, current_person_box, frame_idx, active_target_ids_current_frame):
    candidates = get_relinking_candidates(frame_idx)
    candidates_scores = []
    
    first_new_track_frame = track_first_seen.get(track_id, frame_idx)
    
    for lost_track_id, lost_data in candidates.items():
        
        if lost_track_id in matched_lost_tracks:
            continue
        
        candidate_target_id = lost_data["target_id"]
        
        # Target-ID wurde im aktuellen Frame bereits
        # einer anderen Person zugeordnet.
        if candidate_target_id in active_target_ids_current_frame:
            continue

        # Eine Target-ID darf nicht bereits von einem aktuell
        # sichtbaren anderen ByteTrack verwendet werden.
        target_already_active = any(
            active_bt_id != track_id
            and active_bt_id in track_to_target
            and track_to_target[active_bt_id] == candidate_target_id
            for active_bt_id in active_track_ids_last_frame
        )

        if target_already_active:
            continue
        
        
        # Alter und neuer Track dürfen sich zeitlich nicht überschneiden.
        # Wenn der neue Track bereits sichtbar war, bevor der alte
        # Track verschwunden ist, können beide nicht dieselbe Person sein.
        if first_new_track_frame <= lost_data["last_seen"]:
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



def assign_target_id(track_id, person_box, current_histogram, frame_idx, active_target_ids_current_frame):
    """
    Gibt die Target-ID einer ByteTrack-ID zurück.

    Bekannte ByteTrack-IDs behalten ihre vorhandene Zuordnung.
    Neue IDs werden zunächst zwei Frames lang beobachtet.
    Im dritten Frame wird einmalig ein Re-Linking-Versuch
    mit dem dann aktuellen Histogramm durchgeführt.
    """

    # Die ByteTrack-ID wurde bereits endgültig zugeordnet.
    if track_id in track_to_target:
        return track_to_target[track_id]

    # Anzahl der bisher sichtbaren Frames dieser neuen ID erhöhen.
    pending_track_frames[track_id] = (pending_track_frames.get(track_id, 0) + 1)

    pending_frames = pending_track_frames[track_id]

    # In Frame 1 und 2 noch keine endgültige Entscheidung.
    # Vorläufig wird Target = ByteTrack zurückgegeben,
    # aber nicht in track_to_target gespeichert.
    if pending_frames < MIN_FRAMES_BEFORE_RELINKING:
        return track_id

    # Im dritten Frame wurde außerhalb dieser Funktion
    # erstmals ein Histogramm berechnet.
    match = find_matching_target(track_id, current_histogram, person_box, frame_idx, active_target_ids_current_frame)

    if match is not None:

        target_id = match["target_id"]
        lost_track_id = match["lost_track_id"]

        old_reference_frames = lost_tracks[
            lost_track_id
        ].get(
            "reference_frames",
            []
        )

        new_reference_frames = reference_frames.get(
            track_id,
            []
        )

        vision_result = compare_reference_frames_with_qwen(
            old_reference_frames,
            new_reference_frames
        )

        print(
            f"Qwen-Vergleich BT {lost_track_id} -> "
            f"BT {track_id}: {vision_result}"
        )

        # Qwen bestätigt den Histogramm-Kandidaten
        if vision_result == "SAME":

            if target_id_conflicts_with_pending_frames(
                track_id,
                target_id
            ):
                track_to_target[track_id] = track_id

                print(
                    f"Re-Linking verworfen: Target {target_id} "
                    f"war während der Pending-Frames bereits aktiv."
                )

            else:
                track_to_target[track_id] = target_id

                lost_tracks.pop(
                    lost_track_id,
                    None
                )

                matched_lost_tracks.add(
                    lost_track_id
                )

                relinked_track_ids.add(
                    track_id
                )

                print(
                    f"Re-Linking erfolgreich: "
                    f"BT {track_id} -> Target {target_id} "
                    f"(vorherige BT-ID {lost_track_id}, "
                    f"Histogramm {match['similarity']:.3f}, "
                    f"Qwen SAME)"
                )
        # Qwen lehnt Kandidaten ab
        else:

            track_to_target[track_id] = track_id

            print(
                f"Re-Linking durch Qwen abgelehnt: "
                f"BT {track_id} -> Target {track_id} "
                f"(Kandidat BT {lost_track_id}, "
                f"Histogramm {match['similarity']:.3f}, "
                f"Qwen {vision_result})"
            )
    else:
        # Kein geeigneter Histogramm-Kandidat vorhanden
        track_to_target[track_id] = track_id

        print(
            f"Neue Person bestätigt: "
            f"BT {track_id} -> Target {track_id}"
        )


    # Die Wartephase ist abgeschlossen
    pending_track_frames.pop(track_id, None)

    return track_to_target[track_id]
    

    

track_last_seen = {}
MAX_TRACK_MEMORY_FRAMES = 30


def process_frame(frame, model, frame_idx):
    
    global active_track_ids_last_frame
    
    original_frame = frame.copy()
    
    
    active_track_ids_current_frame = set() 
    active_target_ids_current_frame = set() 
    
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
        
        if track_id not in track_first_seen:
            track_first_seen[track_id] = frame_idx
        
        active_track_ids_current_frame.add(track_id)  # Track-ID als aktiv markieren
        
        if track_id in lost_tracks:
            lost_tracks.pop(track_id, None)
            print(f"Track-ID {track_id} wurde wiedergefunden und aus den verlorenen Tracks entfernt.")
        
        # Zeitpunkt speichern, zu dem der Track zuletzt sichtbar war
        track_last_seen[track_id] = frame_idx
        
        person_box = (x1, y1, x2, y2)
        
        update_reference_frames(track_id, original_frame, person_box, conf, frame_idx)
        

        # MOT-Eintrag für diesen Frame vorbereiten
        mot_entry = {
            "frame": frame_idx,
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "conf": conf
        }

        # ByteTrack-Baseline:
        # Hier wird immer die originale ByteTrack-ID verwendet.
        mot_results_bytetrack.append({
            **mot_entry,
            "id": track_id
        })

        
        last_person_boxes[track_id] = person_box  # Speichern der zuletzt bekannten Personenbox 
        
        current_histogram = None
        
        if track_id in track_to_target:
            current_histogram = calculate_histogram(original_frame, person_box)
            update_histogram_models(track_id, current_histogram)
            
        else:
            next_pending_frame = (pending_track_frames.get(track_id,0 ) +1)
            if next_pending_frame >= MIN_FRAMES_BEFORE_RELINKING:
                current_histogram = calculate_histogram(original_frame, person_box)
        
        target_id = assign_target_id(track_id, person_box, current_histogram, frame_idx, active_target_ids_current_frame)
        
        if track_id in track_to_target:
            active_target_ids_current_frame.add(target_id)
        

        # -------------------------------------------------
        # MOT-Ausgabe für die Re-Linking-Variante
        # -------------------------------------------------

        if track_id in track_to_target:
            # Die Target-ID ist endgültig bekannt.

            if track_id in pending_mot_results:
                # Die vorherigen Pending-Frames rückwirkend
                # mit der jetzt bekannten Target-ID speichern.
                finalize_pending_mot_results(track_id, target_id)

            # Den aktuellen Frame direkt speichern.
            mot_results_target.append({
                **mot_entry,
                "id": target_id
            })

        else:
            # Die ersten Frames eines neuen Tracks:
            # Target-ID ist noch nicht endgültig bekannt.
            pending_mot_results.setdefault(track_id, []).append(mot_entry)
        
        if (track_id in track_to_target and current_histogram is not None and track_id not in person_histogram_models):
            person_histogram_models[track_id] = current_histogram.copy()
        
        
        
        
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
        person_histogram_models.pop(track_id, None)
        pending_track_frames.pop(track_id, None)

    return frame

def run_sequence(sequence_name):
    global track_to_target
    global active_track_ids_last_frame
    global lost_tracks
    global last_person_boxes
    global person_histogram_models
    global matched_lost_tracks
    global relinked_track_ids
    global pending_track_frames
    global track_first_seen
    global reference_frames
    global track_last_seen
    global mot_results_bytetrack
    global mot_results_target
    global pending_mot_results
    global model
    
    track_to_target = {}
    active_track_ids_last_frame = set()
    lost_tracks = {}
    last_person_boxes = {}

    person_histogram_models = {}
    matched_lost_tracks = set()
    relinked_track_ids = set()

    pending_track_frames = {}
    track_first_seen = {}

    reference_frames = {}
    track_last_seen = {}

    mot_results_bytetrack = []
    mot_results_target = []
    pending_mot_results = {}

    # ByteTrack-Zustand ebenfalls neu starten
    model = YOLO("yolov8m.pt")
    
    sequence_path = Path("MOT17/val") / sequence_name
    image_dir = sequence_path / "img1"

    image_paths = sorted(image_dir.glob("*.jpg"))

    video_name = sequence_name.replace("-FRCNN", "")

    print("\n" + "=" * 60)
    print(f"Starte Sequenz: {sequence_name}")
    print(f"Anzahl Frames: {len(image_paths)}")
    print("=" * 60)
    


    processed_frames = 0
    total_processing_time = 0.0

    frame_idx = 0


    for image_path in image_paths:

        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"Frame konnte nicht gelesen werden: {image_path}")
            continue

        frame_idx += 1

        # Zeitmessung für die Verarbeitung eines Frames starten
        start_time = time.perf_counter()

        process_frame(frame, model, frame_idx)    

        # Verarbeitungszeit dieses Frames berechnen
        processing_time = time.perf_counter() - start_time

        total_processing_time += processing_time
        processed_frames += 1
    
            

    # -------------------------------------------------
    # Noch offene Pending-Tracks für die MOT-Evaluation
    # abschließen
    # -------------------------------------------------

    for track_id in list(pending_mot_results.keys()):
        finalize_pending_mot_results(track_id, track_id)


        
    if processed_frames > 0:

        average_processing_time = (total_processing_time / processed_frames)

        processing_fps = (1.0 / average_processing_time)

        print(f"\nVerarbeitete Frames: {processed_frames}")

        print("Durchschnittliche Zeit pro Frame: "f"{average_processing_time:.4f} s")

        print("Verarbeitungsgeschwindigkeit: "f"{processing_fps:.2f} FPS")
        
    os.makedirs(
        "mot_results",
        exist_ok=True
    )

    hist_tag = f"{int(MIN_HISTOGRAM_SIMILARITY * 100):03d}"
    pending_tag = MIN_FRAMES_BEFORE_RELINKING

    config_tag = f"h{hist_tag}_p{pending_tag}"

    save_mot_results(
        mot_results_bytetrack,
        f"mot_results/{video_name}_bytetrack_{config_tag}.txt"
    )

    save_mot_results(
        mot_results_target,
        f"mot_results/{video_name}_qwen_{config_tag}.txt"
    )
    

    print("\nMOT-Evaluationsergebnisse gespeichert:")
    print(
        f"  ByteTrack: mot_results/{video_name}_bytetrack_{config_tag}.txt"
    )
    print(
        f"  Re-Linking: mot_results/{video_name}_qwen_{config_tag}.txt"
    )


    print(f"Erfolgreiche Re-Linkings: {len(relinked_track_ids)}")
    print(f"ByteTrack-Einträge: {len(mot_results_bytetrack)}")
    print(f"Target-Einträge: {len(mot_results_target)}")
    
    
    
for sequence_name in SEQUENCES:
    run_sequence(sequence_name)
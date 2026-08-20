# Basispipeline ByteTrack und SCRFD auf Person ROI

# Stand: 27.7. 
import time
import os
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import torchvision
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
video_path = "Testvideos/durcheinander.mp4"

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
OUTPUT_FPS = 50.0

if save_video:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_writer = cv2.VideoWriter(
        f"Ausgabevideos/{video_name}_bytetrackvergleich.mp4",
        fourcc,
        OUTPUT_FPS,
        (frame_width, frame_height)
    )


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



def detect_face(frame, person_box, face_detector):

    x1, y1, x2, y2 = person_box

    frame_height, frame_width = frame.shape[:2]

    # Personenbox auf die Bildgrenzen beschränken
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    # Personenbereich ausschneiden
    person_roi = frame[y1:y2, x1:x2]

    # Leeren oder ungültigen Bereich überspringen
    if person_roi.size == 0:
        return None

    # Gesichtserkennung mit SCRFD
    faces = face_detector.get(person_roi)

    # Kein Gesicht erkannt
    if not faces:
        return None

    # Gesicht mit der höchsten Detektionskonfidenz auswählen
    best_face = max(faces, key=lambda face: face.det_score)

    # Gesichtsbox liegt zunächst relativ zur Personenbox vor
    face_x1, face_y1, face_x2, face_y2 = (best_face.bbox.astype(int))

    # Koordinaten auf den gesamten Frame zurückrechnen
    face_x1 += x1
    face_y1 += y1
    face_x2 += x1
    face_y2 += y1

    # Gesichtsbox ebenfalls auf Bildgrenzen beschränken
    face_x1 = max(0, face_x1)
    face_y1 = max(0, face_y1)
    face_x2 = min(frame_width, face_x2)
    face_y2 = min(frame_height, face_y2)

    return (face_x1,face_y1,face_x2,face_y2)


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
    

def process_frame(frame, model, face_detector):
    
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
        
        person_box = (x1, y1, x2, y2)
        
        face_box = detect_face(frame, person_box, face_detector)
        
        if face_box is not None:
            anonymize_face(frame, face_box)
            
        color = get_color_for_id(track_id)
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5) # Farbige Box um Personen zeichnen , die 5 steht für die Dicke der Box
            
        cv2.putText(frame, f"ID {track_id} | {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    return frame



processed_frames = 0
total_processing_time = 0.0


while True:

    ret, frame = cap.read()

    if not ret:
        break

    start_time = time.perf_counter()

    processed_frame = process_frame(frame, model, face_detector)

    processing_time = time.perf_counter() - start_time

    total_processing_time += processing_time
    processed_frames += 1

    if save_video:
        video_writer.write(processed_frame)

    cv2.imshow("Processed Frame", processed_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()

if save_video:
    video_writer.release()
    
cv2.destroyAllWindows()


if processed_frames > 0:

    average_processing_time = total_processing_time / processed_frames

    processing_fps = 1 / average_processing_time

    print(f"\nVerarbeitete Frames: {processed_frames}")
    print(f"Durchschnittliche Zeit pro Frame: {average_processing_time:.4f} s")
    print(f"Verarbeitungsgeschwindigkeit: {processing_fps:.2f} FPS")
       
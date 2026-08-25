import json
import os
import cv2


# -------------------------------------------------
# Konfiguration
# -------------------------------------------------

# Bereits anonymisiertes Video und zugehörige Trackingdaten
video_path = "Ausgabevideos/durcheinander_histogram_anonymized.mp4"
tracking_path = "tracking_results/durcheinander_histogram_tracks.json"

# Optionales Speichern der selektiven Darstellung
SAVE_VIDEO = False
output_path = "Ausgabevideos/durcheinander_histogram_selected.mp4"

# Darstellungszustand
show_all_targets = True
selected_target_ids = set()

# Im aktuellen Frame sichtbare Targets und ihre Personenboxen
# Aufbau: (target_id, x1, y1, x2, y2)
current_tracks = []


# -------------------------------------------------
# Trackingdaten laden
# -------------------------------------------------

with open(tracking_path, "r") as f:
    tracking_results = json.load(f)


# Trackingdaten nach Frames gruppieren, damit direkt auf die
# Einträge des jeweils aktuellen Frames zugegriffen werden kann
tracks_by_frame = {}

for entry in tracking_results:
    frame_idx = entry["frame"]
    tracks_by_frame.setdefault(frame_idx, []).append(entry)


# -------------------------------------------------
# Darstellung
# -------------------------------------------------

# Farben zur Unterscheidung verschiedener Target-IDs
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


def get_color_for_id(target_id):
    """Weist jeder Target-ID eine wiederkehrende Farbe zu."""
    return ID_COLORS[target_id % len(ID_COLORS)]


# -------------------------------------------------
# Selektive Auswahl
# -------------------------------------------------

def mouse_click(event, x, y, flags, param):
    """Wählt das angeklickte Target aus oder hebt dessen Auswahl auf."""

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Prüfen, ob der Mausklick innerhalb einer aktuellen Personenbox liegt
    for target_id, x1, y1, x2, y2 in current_tracks:

        if x1 <= x <= x2 and y1 <= y <= y2:

            if target_id in selected_target_ids:
                selected_target_ids.remove(target_id)
                print(f"Target-ID {target_id} wurde abgewählt.")
            else:
                selected_target_ids.add(target_id)
                print(f"Target-ID {target_id} wurde ausgewählt.")

            break


# -------------------------------------------------
# Interaktiver Viewer
# -------------------------------------------------

def run_viewer():
    """Zeigt das Video an und ermöglicht die interaktive Target-Auswahl."""

    global show_all_targets

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    paused = False
    current_frame = None

    window_name = "Tracking Viewer"

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)

    while True:

        # Während einer Pause wird der aktuelle Frame beibehalten
        if not paused:
            ret, frame = cap.read()

            if not ret:
                break

            frame_idx += 1
            current_frame = frame.copy()

        # Auf einer Kopie zeichnen, damit beim erneuten Anzeigen desselben
        # Frames keine alten Bounding Boxes erhalten bleiben
        display_frame = current_frame.copy()

        # Trackingdaten des aktuellen Frames abrufen
        frame_tracks = tracks_by_frame.get(frame_idx, [])

        # Personenboxen des aktuellen Frames für die Mausklick-Auswahl speichern
        current_tracks.clear()

        for entry in frame_tracks:
            target_id = entry["target_id"]

            x1 = entry["x1"]
            y1 = entry["y1"]
            x2 = entry["x2"]
            y2 = entry["y2"]

            current_tracks.append((target_id, x1, y1, x2, y2))

            # Im selektiven Modus nur ausgewählte Targets darstellen
            if not show_all_targets and target_id not in selected_target_ids:
                continue

            color = get_color_for_id(target_id)

            # Personenbox und finale Target-ID einzeichnen
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 5)
            cv2.putText(display_frame, f"Target {target_id}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(30) & 0xFF

        # ESC: Viewer beenden
        if key == 27:
            break

        # Leertaste: Wiedergabe pausieren oder fortsetzen
        elif key == ord(" "):
            paused = not paused
            print(f"Pause: {paused}")

        # S: zwischen allen und nur ausgewählten Targets wechseln
        elif key == ord("s"):
            show_all_targets = not show_all_targets
            print(f"Alle Targets anzeigen: {show_all_targets}")

        # C: bisherige Target-Auswahl zurücksetzen
        elif key == ord("c"):
            selected_target_ids.clear()
            print("Auswahl wurde zurückgesetzt.")

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------
# Selektives Video speichern
# -------------------------------------------------

def save_selected_video():
    """Erzeugt ein neues Video mit den zuvor ausgewählten Targets."""

    if not selected_target_ids:
        print("Keine Targets ausgewählt. Es wurde kein Video gespeichert.")
        return

    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        # Trackingdaten des aktuellen Frames abrufen
        frame_tracks = tracks_by_frame.get(frame_idx, [])

        for entry in frame_tracks:
            target_id = entry["target_id"]

            # Im Ausgabevideo ausschließlich ausgewählte Targets darstellen
            if target_id not in selected_target_ids:
                continue

            x1 = entry["x1"]
            y1 = entry["y1"]
            x2 = entry["x2"]
            y2 = entry["y2"]

            color = get_color_for_id(target_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
            cv2.putText(frame, f"Target {target_id}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        video_writer.write(frame)

    cap.release()
    video_writer.release()

    print(f"Selektives Video gespeichert: {output_path}")


# -------------------------------------------------
# Programm starten
# -------------------------------------------------

run_viewer()

if SAVE_VIDEO:
    save_selected_video()
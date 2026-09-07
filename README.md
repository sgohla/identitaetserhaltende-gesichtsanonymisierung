# Identitätserhaltende Anonymisierung von Gesichtern in Videoaufnahmen

Dieses Repository enthält den im Rahmen der Bachelorarbeit **„Identitätserhaltende Anonymisierung von Gesichtern in Videoaufnahmen“** entwickelten Code.

Die entwickelte Pipeline kombiniert Personendetektion und -tracking mit Gesichtserkennung und -anonymisierung. Zusätzlich wurden Verfahren zum Re-Linking unterbrochener Tracks sowie zur Stabilisierung der Gesichtsanonymisierung umgesetzt.

## Verwendete Verfahren

Für die grundlegende Verarbeitung werden folgende Modelle und Verfahren verwendet:

- YOLOv8m zur Personendetektion
- ByteTrack zur Personenverfolgung
- SCRFD zur Gesichtserkennung
- Gaußsche Weichzeichnung zur Gesichtsanonymisierung
- Kalman-Filter zur Überbrückung kurzzeitiger Ausfälle der Gesichtserkennung

Für das Re-Linking unterbrochener Tracks stehen zwei Varianten zur Verfügung:

- Histogramm-basiertes Re-Linking
- VLM-basiertes Re-Linking mit Qwen2.5-VL-7B-Instruct

## Installation

Die benötigten Python-Abhängigkeiten können über die `requirements.txt` installiert werden.

```bash
pip install -r requirements.txt
```

Je nach verwendeter Variante können zusätzliche Abhängigkeiten für die eingesetzten Modelle erforderlich sein.

## Tracking und Re-Linking

Das Tracking kann entweder mit dem Histogramm-basierten oder dem VLM-basierten Re-Linking durchgeführt werden. Beide Varianten erzeugen eine JSON-Datei, in der die Ergebnisse des Trackings und die Zuordnung der Track-IDs zu den übergeordneten Target-IDs gespeichert werden.

### Histogramm-basiertes Re-Linking

Das Histogramm-basierte Re-Linking kann über folgendes Skript ausgeführt werden:

```bash
python <histogram_script>.py
```

### VLM-basiertes Re-Linking

Für das VLM-basierte Re-Linking wird Qwen2.5-VL-7B-Instruct verwendet.

```bash
python <qwen_script>.py
```

Für die Ausführung auf dem Mogon-Cluster steht zusätzlich ein SLURM-Skript zur Verfügung:

```bash
sbatch <qwen_script>.sbatch
```

Die für Qwen benötigten Modellgewichte sind nicht Bestandteil dieses Repositories und müssen separat bereitgestellt beziehungsweise geladen werden.

## Erstellung des anonymisierten Videos

Auf Grundlage der erzeugten Tracking-Daten kann anschließend das anonymisierte Video erstellt werden. Das entsprechende Skript befindet sich unter:

```text
eval/create_anonymized_video/
```

Bei der Erstellung des Videos wird die Gesichtserkennung innerhalb der durch YOLO und ByteTrack bestimmten Personenregionen durchgeführt. Zusätzlich werden die implementierten Verfahren zur Stabilisierung der Gesichtsanonymisierung angewendet. Dazu gehören die Auswahl eines passenden Gesichts bei mehreren erkannten Gesichtskandidaten sowie die Kalman-basierte Überbrückung kurzzeitiger Ausfälle der Gesichtserkennung.

Die erkannten Gesichter werden anschließend mittels einer an die Gesichtsgröße angepassten Gaußschen Weichzeichnung anonymisiert.

Das Skript kann über folgenden Befehl ausgeführt werden:

```bash
python <anonymization_script>.py
```

## Tracking Viewer

Der Tracking Viewer dient zur Visualisierung der Tracking-Ergebnisse zusammen mit dem erzeugten anonymisierten Video.

Für die Verwendung werden zwei Dateien benötigt:

- die beim Tracking erzeugte JSON-Datei
- das zugehörige anonymisierte Video

Der Viewer kann über folgendes Skript gestartet werden:

```bash
python <tracking_viewer>.py
```

Nach dem Start werden die JSON-Datei und das zugehörige anonymisierte Video ausgewählt. Die in der JSON-Datei gespeicherten Tracking- und Target-ID-Informationen werden anschließend gemeinsam mit dem Video dargestellt.

## Evaluation

Das Repository enthält außerdem Skripte, die für die Evaluation der entwickelten Verfahren verwendet wurden.

Die Evaluation des Personentrackings basiert auf ausgewählten Sequenzen des MOT17-Datensatzes. Dabei wurden ByteTrack sowie die entwickelten Re-Linking-Varianten miteinander verglichen.

Weitere Evaluationsskripte dienen der Untersuchung der ROI-basierten Gesichtserkennung und der Verfahren zur Stabilisierung der Gesichtsanonymisierung.

## Hinweise

Große Dateien wie Modellgewichte, verwendete Testvideos und Datensätze sind nicht Bestandteil des Repositories.

Für die Tracking-Evaluation wird der MOT17-Datensatz separat benötigt. Qwen2.5-VL-7B-Instruct und weitere verwendete Modellgewichte müssen ebenfalls separat bereitgestellt werden.

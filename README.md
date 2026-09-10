# Identitätserhaltende Anonymisierung von Personen in Videoaufnahmen

Dieses Repository enthält den im Rahmen der Bachelorarbeit **„Identitätserhaltende Anonymisierung von Personen in Videoaufnahmen“** entwickelten Code.

Die entwickelte Pipeline umfasst die Personendetektion und -verfolgung, Verfahren zum Re-Linking unterbrochener Tracks sowie die Gesichtsdetektion und -anonymisierung. Zusätzlich wurden Verfahren zur Stabilisierung der Gesichtsanonymisierung umgesetzt.

## Verwendete Verfahren

Für die grundlegende Verarbeitung werden folgende Modelle und Verfahren verwendet

- YOLOv8m zur Personendetektion
- ByteTrack zur Personenverfolgung
- SCRFD zur Gesichtsdetektion
- Gaußsche Weichzeichnung zur Gesichtsanonymisierung
- Kalman-Filter zur Überbrückung kurzzeitiger Ausfälle der Gesichtsdetektion

Für das Re-Linking unterbrochener Tracks stehen zwei Varianten zur Verfügung

- Histogramm-basiertes Re-Linking
- VLM-basiertes Re-Linking mit Qwen2.5-VL-7B-Instruct

## Installation

Die benötigten Python-Abhängigkeiten können über die `requirements.txt` installiert werden.

```bash
pip install -r requirements.txt
```

Die Pfade zu den jeweiligen Eingabevideos und Ergebnisdateien können in den entsprechenden Skripten angepasst werden.

## Tracking und Re-Linking

Für das Tracking und Re-Linking wird ein Eingabevideo verarbeitet. Dabei kann entweder das Histogramm-basierte oder das VLM-basierte Verfahren verwendet werden.

Beide Varianten erzeugen eine JSON-Datei mit den Tracking-Ergebnissen und den Zuordnungen der von ByteTrack vergebenen Track-IDs zu den übergeordneten Target-IDs.

### Histogramm-basiertes Re-Linking

Das Histogramm-basierte Re-Linking wird über folgendes Skript ausgeführt

```bash
python pipeline/pipeline_histogram_relinking.py
```
Neben den Tracking-Ergebnissen wird während der Verarbeitung auch das anonymisierte Ausgabevideo erzeugt. Die JSON-Datei und das zugehörige anonymisierte Video können anschließend gemeinsam im Tracking Viewer verwendet werden.

### VLM-basiertes Re-Linking

Für das VLM-basierte Re-Linking wird Qwen2.5-VL-7B-Instruct verwendet.

```bash
python pipeline/pipeline_qwen_relinking.py
```

Die für Qwen benötigten Modellgewichte sind nicht Bestandteil dieses Repositories und werden bei der Verwendung des Modells separat geladen.

Die VLM-basierte Variante wurde für die Ausführung auf Mogon umgesetzt. Die Erzeugung des anonymisierten Videos ist hierbei von der Erzeugung der Tracking-Ergebnisse getrennt. Ein zum selben Eingabevideo gehörendes anonymisiertes Video kann mit dem nachfolgend beschriebenen Skript erzeugt werden.

## Erstellung des anonymisierten Videos

Die Erstellung des anonymisierten Videos erfolgt unabhängig vom Re-Linking. Dafür wird dasselbe Eingabevideo mit folgendem Skript verarbeitet

```bash
python pipeline/create_anonymized_videos.py
```

Bei der Verarbeitung wird die Gesichtsdetektion innerhalb der durch YOLO und ByteTrack bestimmten Personenregionen durchgeführt. Zusätzlich werden die implementierten Verfahren zur Stabilisierung der Gesichtsanonymisierung angewendet. Dazu gehören die Auswahl eines passenden Gesichts bei mehreren erkannten Gesichtskandidaten sowie die Kalman-basierte Überbrückung kurzzeitiger Ausfälle der Gesichtsdetektion.

Die erkannten Gesichter werden anschließend mittels einer an die Gesichtsgröße angepassten Gaußschen Weichzeichnung anonymisiert.

## Tracking Viewer

Der Tracking Viewer dient zur gemeinsamen Visualisierung der Tracking-Ergebnisse und des anonymisierten Videos.

Dafür werden zwei zum selben Eingabevideo gehörende Dateien benötigt

- die beim Tracking und Re-Linking erzeugte JSON-Datei
- das zum selben Eingabevideo gehörende anonymisierte Video

Der Viewer wird über folgendes Skript gestartet

```bash
python pipeline/tracking_viewer.py
```

Vor dem Start werden im Skript die Pfade zur JSON-Datei und zum zugehörigen anonymisierten Video angegeben. Die gespeicherten Tracking- und Target-ID-Informationen werden anschließend gemeinsam mit dem anonymisierten Video dargestellt.

## Evaluation

Der Ordner `evaluation` enthält die Skripte, die für die Evaluation der entwickelten Verfahren verwendet wurden.

Die Evaluation des Personentrackings basiert auf ausgewählten Sequenzen des MOT17-Datensatzes. Dabei wurden ByteTrack sowie die entwickelten Re-Linking-Verfahren miteinander verglichen.

Weitere Evaluationsskripte dienen der Untersuchung der ROI-basierten Gesichtserkennung, der Auswahl bei mehreren Gesichtskandidaten sowie der Verfahren zur Stabilisierung der Gesichtsanonymisierung.

## Archive

Der Ordner `archive` enthält ältere und experimentelle Implementierungen, die während der Entwicklung der Pipeline entstanden sind. Diese Dateien sind nicht Bestandteil des finalen Workflows.

## Hinweise

Die verwendeten Testvideos, Datensätze und daraus erzeugten Ergebnisdateien sind nicht Bestandteil dieses Repositories.

Für die Anwendung der Pipeline kann ein eigenes Eingabevideo verwendet werden. Die JSON-Datei mit den Tracking-Ergebnissen und das zugehörige anonymisierte Video können anschließend gemeinsam im Tracking Viewer geöffnet werden.

# SmartDart
Ein computergestütztes Dart-Erkennungssystem mit automatischer Punktzählung für Dartspiele. Das Projekt kombiniert Computer Vision, maschinelles Lernen und eine grafische Benutzeroberfläche für ein vollständiges Darterlebnis. Dabei verwendet das System YOLO-basierte Objekterkennung zur automatischen Erkennung von Darts auf einer Dartscheibe und berechnet die erzielten Punkte. 

## Hauptfunktionen

### Automatische Dart-Erkennung
- YOLO-basierte Objekterkennung für Darts und Dartscheiben-Referenzpunkte
- Echtzeit-Kameraverarbeitung mit OpenCV
- Kamera-Kalibrierung für perspektivische Korrektur
- Intelligente Positionsfilterung zur Vermeidung von Doppelerkennungen

### Punktberechnung
- Automatische Dartscheiben-Kalibrierung basierend auf erkannten Referenzpunkten
- Präzise Punktberechnung für alle Dartscheiben-Bereiche (Single, Double, Triple, Bull)
- Template-Overlay zur visuellen Darstellung der Dartscheiben-Regionen
- Manuelle Punkteingabe als Fallback-Option

### Tournament Management
- Vollständige Turnier-Verwaltung mit mehreren Spielern
- 501-Dart-Spiel-Implementierung mit korrekten Finish-Regeln
- Automatische Rundenverwaltung und Spielerwechsel
- Statistiken und Durchschnittsberechnung pro Spieler
- Spielstand-Speicherung und -Wiederherstellung

### Benutzeroberfläche
- Moderne Tkinter-basierte GUI mit Echtzeit-Kamera-Feed
- Intuitive Bedienung mit Buttons für alle wichtigen Funktionen
- Status-Anzeigen für Kalibrierung und Spielfortschritt
- Manuelle Korrekturen und Undo-Funktionalität

## Technische Komponenten

### Machine Learning
- **YOLO-Modelle**: Verschiedene trainierte Modelle für Dart- und Dartscheiben-Erkennung
- **Active Learning**: Mehrstufiger Trainingsprozess mit aktiven Lernzyklen
- **Transfer Learning**: Feinabstimmung von vortrainierten YOLO-Modellen
- **Datenaugmentation**: Automatische Annotation-Generierung für Trainingsdaten

### Computer Vision
- **Kamera-Kalibrierung**: Perspektivische Korrektur durch Homographie-Transformation
- **Bildverarbeitung**: Echtzeit-Verarbeitung mit optimierter Performance
- **Template Matching**: Dartscheiben-Template für präzise Punktberechnung
- **Positionsstabilisierung**: Filterung und Stabilisierung erkannter Dart-Positionen

### Software-Architektur
- **Modularer Aufbau**: Getrennte Module für verschiedene Funktionalitäten
- **Threading**: Asynchrone Kameraverarbeitung für flüssige GUI
- **Event-System**: Reaktive Programmierung für Benutzerinteraktionen
- **Error Handling**: Robuste Fehlerbehandlung und Recovery-Mechanismen

## Projektstruktur

```
├── main.py                    # Kommandozeilen-Version für Entwicklung
├── gui_darts_tournament.py    # Hauptanwendung mit GUI
├── requirements.txt           # Python-Abhängigkeiten
├── models/                    # Trainierte YOLO-Modelle
├── utils/                     # Utility Module
│   ├── camera.py             # Kamera-Interface
│   ├── calibration.py        # Kamera-Kalibrierung
│   ├── predict.py            # YOLO-Vorhersagen
│   └── score_prediction.py   # Punktberechnung
├── training/                  # Machine Learning Pipeline
│   ├── train_model.py        # Modell-Training
│   ├── generate_annotations.py # Annotation-Generierung
│   ├── validate_annotations.py # Datenvalidierung
│   └── data/                 # Trainings- und Validierungsdaten
└── resources/                # Referenzbilder und Dokumentation
```

## Installation

### Voraussetzungen
- Python 3.11 oder höher
- Webcam oder kompatible Kamera

### Setup
```bash
pip install -r requirements.txt
```

## Verwendung

### GUI-Anwendung starten
```bash
python gui_darts_tournament.py
```

### Entwicklungs-/Debug-Version
```bash
python main.py
```

## Workflow

1. **Kamera starten**: Aktivierung der Kameraverbindung
2. **Kalibrierung**: Automatische oder manuelle Dartscheiben-Kalibrierung
3. **Spieler hinzufügen**: Konfiguration der Turnier-Teilnehmer
4. **Spiel starten**: Automatische Dart-Erkennung und Punktzählung
5. **Turnier verwalten**: Rundenverwaltung und Endergebnis

## Machine Learning Details

### Trainingsdaten
- **Classes**: 7 Klassen (Dart, 20, 3, 11, 6, 9, 15)
- **Active Learning**: Iterative Datensammlung und Modellverbesserung
- **Datenaugmentation**: Automatische Generierung zusätzlicher Trainingsdaten

### Modellarchitektur
- **YOLO8n / YOLO11n**: Basis-Modell für schnelle Inferenz
- **Transfer Learning**: Spezialisierung auf Dart-Erkennung
- **Multi-Stage Training**: Stufenweise Verbesserung durch Active Learning

### Performance-Optimierung
- **Batch Processing**: Effiziente Verarbeitung mehrerer Frames
- **Frame Skipping**: Reduzierte YOLO-Aufrufe für bessere Performance
- **Caching**: Zwischenspeicherung für konsistente Anzeige

## Technische Features

### Kamera-System
- **Multi-Source Support**: Webcam oder Bildordner als Eingabe
- **Adaptive Framerate**: Automatische Anpassung der Verarbeitungsgeschwindigkeit
- **Error Recovery**: Automatische Wiederherstellung bei Kamera-Problemen

### Kalibrierung
- **Automatische Erkennung**: Kalibrierung basierend auf YOLO-erkannten Referenzpunkten
- **Manuelle Korrektur**: Benutzergesteuerte Kalibrierungs-Optionen
- **Persistenz**: Speicherung der Kalibrierungsparameter

### Dart-Erkennung
- **Anti-Spam Filter**: Vermeidung von Mehrfacherkennungen
- **Positionstoleranz**: Intelligente Gruppierung ähnlicher Positionen
- **Cooldown-System**: Zeitbasierte Filterung für stabile Erkennung
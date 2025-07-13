import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import shutil
from pathlib import Path
import glob
from typing import List, Tuple, Optional

# Import der bestehenden Klassen
from utils.predict import Predictor
from utils.score_prediction import DartboardScorePredictor


class ImageProcessorGUI:
    """
    GUI für die Verarbeitung und Kategorisierung von Dart-Bildern.
    
    Workflow:
    1. YOLO-Erkennung auf allen Bildern
    2. Score-Prediction mit Dartboard-Kalibrierung
    3. GUI zur manuellen Kategorisierung in 9 Kategorien
    """
    
    def __init__(self):
        # Pfade
        self.input_folder = r"C:\Users\CARLO\OneDrive\Desktop\Darts\klassifikation_helligkeit\besondere_belichtung"
        self.output_base_folder = r"C:\Users\CARLO\OneDrive\Desktop\Darts\processed_images_helligkeit_yolo11n-al2-stg3.pt"
        
        # Verarbeitungsklassen
        self.predictor = Predictor("models/yolo11n-al2-stg3.pt")  # Anpassung des Modellpfads
        self.score_predictor = DartboardScorePredictor()
        
        # GUI-Variablen
        self.root = None
        self.canvas = None
        self.photo = None
        self.current_image_index = 0
        self.processed_images = []
        self.current_image_path = None
        
        # Kategorien-Setup
        self.categories = {
            # 1 Dart
            'q': '1Dart/gut',
            'w': '1Dart/medium', 
            'e': '1Dart/schlecht',
            # 2 Darts
            'a': '2Dart/gut',
            's': '2Dart/medium',
            'd': '2Dart/schlecht',
            # 3 Darts
            'y': '3Dart/gut',
            'x': '3Dart/medium',
            'c': '3Dart/schlecht'
        }
        
        self.setup_output_folders()
    
    def setup_output_folders(self):
        """Erstellt die Ausgabeordner-Struktur."""
        base_path = Path(self.output_base_folder)
        
        # Hauptordner für verarbeitete Bilder
        processed_path = base_path / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Kategorieordner
        for category in self.categories.values():
            category_path = base_path / "categorized" / category
            category_path.mkdir(parents=True, exist_ok=True)
    
    def process_all_images(self):
        """Verarbeitet alle Bilder mit YOLO und Score-Prediction."""
        if not os.path.exists(self.input_folder):
            messagebox.showerror("Fehler", f"Eingabeordner nicht gefunden: {self.input_folder}")
            return False
        
        # Bildformate
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext)))
            image_files.extend(glob.glob(os.path.join(self.input_folder, ext.upper())))
        
        # Duplikate entfernen und sortieren
        image_files = sorted(list(set(image_files)))
        
        print(f"Gefundene Dateierweiterungen im Ordner:")
        extensions_found = {}
        for file in image_files:
            ext = os.path.splitext(file)[1].lower()
            extensions_found[ext] = extensions_found.get(ext, 0) + 1
        for ext, count in extensions_found.items():
            print(f"  {ext}: {count} Dateien")
        
        if not image_files:
            messagebox.showerror("Fehler", f"Keine Bilder im Ordner gefunden: {self.input_folder}")
            return False
        
        print(f"Verarbeite {len(image_files)} Bilder...")
        print("Drücken Sie Ctrl+C, um die Verarbeitung zu stoppen und zur GUI zu wechseln.")
        print("Oder warten Sie, bis alle Bilder verarbeitet wurden.")
        
        processed_folder = os.path.join(self.output_base_folder, "processed")
        
        try:
            for i, image_path in enumerate(image_files):
                try:
                    print(f"Verarbeite Bild {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
                    
                    # Prüfen ob bereits verarbeitet
                    output_filename = f"processed_{os.path.basename(image_path)}"
                    output_path = os.path.join(processed_folder, output_filename)
                    
                    if os.path.exists(output_path):
                        print(f"Bild bereits verarbeitet, überspringe: {os.path.basename(image_path)}")
                        # Lade bereits verarbeitetes Bild in die Liste
                        image = cv2.imread(image_path)
                        if image is not None:
                            results = self.predictor.predict(image)
                            dart_positions = []
                            reference_points = []
                            
                            if results and len(results) > 0:
                                result = results[0]
                                if hasattr(result, 'boxes') and result.boxes is not None:
                                    boxes = result.boxes.xyxy.cpu().numpy()
                                    class_ids = result.boxes.cls.cpu().numpy()
                                    
                                    for box, class_id in zip(boxes, class_ids):
                                        x1, y1, x2, y2 = box
                                        center_x = int((x1 + x2) / 2)
                                        center_y = int((y1 + y2) / 2)
                                        
                                        if int(class_id) == 4:  # Dart
                                            dart_positions.append((center_x, center_y))
                                        else:  # Referenzpunkte
                                            reference_points.append((center_x, center_y))
                            
                            self.processed_images.append({
                                'original_path': image_path,
                                'processed_path': output_path,
                                'dart_count': len(dart_positions),
                                'dart_positions': dart_positions
                            })
                        continue
                    
                    # Bild laden
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Fehler beim Laden von {image_path}")
                        continue
                    
                    # YOLO-Erkennung
                    results = self.predictor.predict(image)
                    
                    # Erkennungen nach Typ trennen
                    dart_positions = []  # Class ID 4
                    reference_points = []  # Alle anderen Class IDs (0,1,2,3,5,6)
                    
                    if results and len(results) > 0:
                        result = results[0]
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            boxes = result.boxes.xyxy.cpu().numpy()
                            class_ids = result.boxes.cls.cpu().numpy()
                            
                            for box, class_id in zip(boxes, class_ids):
                                x1, y1, x2, y2 = box
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                
                                if int(class_id) == 4:  # Dart
                                    dart_positions.append((center_x, center_y))
                                else:  # Referenzpunkte für Dartboard-Kalibrierung
                                    reference_points.append((center_x, center_y))
                    
                    # Score-Prediction mit korrekter Dartboard-Kalibrierung
                    processed_image = image.copy()
                    scores = []
                    
                    # Dartboard mit Referenzpunkten kalibrieren (falls vorhanden)
                    if reference_points and len(reference_points) >= 3:
                        try:
                            # Kalibriere das Dartboard mit den Referenzpunkten
                            if self.score_predictor.calibrate_dartboard(reference_points):
                                print(f"Dartboard erfolgreich kalibriert mit {len(reference_points)} Referenzpunkten")
                                
                                # Overlay des Dartboard-Templates anzeigen
                                processed_image = self.score_predictor.overlay_dartboard_template(
                                    processed_image, 
                                    reference_points=reference_points,
                                    show_numbers=True,
                                    show_analysis=False
                                )
                                
                                # Berechne Scores für Dart-Positionen (falls vorhanden)
                                if dart_positions:
                                    processed_image, scores = self.score_predictor.process_dart_detections(
                                        processed_image, dart_positions, show_scores=True
                                    )
                            else:
                                print("Dartboard-Kalibrierung fehlgeschlagen")
                                # Fallback: Nur Erkennungen markieren
                                for pos in reference_points:
                                    cv2.circle(processed_image, pos, 8, (255, 0, 0), 2)  # Blau für Referenzpunkte
                                for pos in dart_positions:
                                    cv2.circle(processed_image, pos, 10, (0, 255, 0), 2)  # Grün für Darts
                                    cv2.putText(processed_image, "Dart", (pos[0]+15, pos[1]), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Fehler bei der Dartboard-Kalibrierung: {str(e)}")
                            # Fallback: Nur Erkennungen markieren
                            for pos in reference_points:
                                cv2.circle(processed_image, pos, 8, (255, 0, 0), 2)  # Blau für Referenzpunkte
                            for pos in dart_positions:
                                cv2.circle(processed_image, pos, 10, (0, 255, 0), 2)  # Grün für Darts
                                cv2.putText(processed_image, "Dart", (pos[0]+15, pos[1]), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif dart_positions:
                        # Nur Darts vorhanden, keine Kalibrierung möglich
                        print("Keine Referenzpunkte für Dartboard-Kalibrierung gefunden")
                        for pos in dart_positions:
                            cv2.circle(processed_image, pos, 10, (0, 255, 0), 2)  # Grün für Darts
                            cv2.putText(processed_image, "Dart", (pos[0]+15, pos[1]), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Verarbeitetes Bild speichern
                    cv2.imwrite(output_path, processed_image)
                    
                    self.processed_images.append({
                        'original_path': image_path,
                        'processed_path': output_path,
                        'dart_count': len(dart_positions),
                        'dart_positions': dart_positions
                    })
                    
                except Exception as e:
                    print(f"Fehler bei der Verarbeitung von {image_path}: {str(e)}")
                    continue
                    
        except KeyboardInterrupt:
            print(f"\nVerarbeitung unterbrochen. {len(self.processed_images)} Bilder bereits verarbeitet.")
            print("Starte GUI mit den bisher verarbeiteten Bildern...")
        
        print(f"Verarbeitung abgeschlossen. {len(self.processed_images)} Bilder erfolgreich verarbeitet.")
        return True
    
    def setup_gui(self):
        """Erstellt die GUI für die Kategorisierung."""
        self.root = tk.Tk()
        self.root.title("Dart-Bilder Kategorisierung")
        self.root.geometry("1200x800")
        
        # Hauptframe
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Informationsbereich
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.info_label = ttk.Label(info_frame, text="", font=("Arial", 12))
        self.info_label.pack(side=tk.LEFT)
        
        # Progress-Frame
        progress_frame = ttk.Frame(info_frame)
        progress_frame.pack(side=tk.RIGHT)
        
        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.pack()
        
        # Bildbereich
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(image_frame, bg='gray')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Kategorisierung-Bereich
        category_frame = ttk.LabelFrame(main_frame, text="Kategorisierung", padding=10)
        category_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Kategorie-Buttons in 3 Spalten
        categories_info = [
            ("1 Dart:", [("Q - Gut", "q"), ("W - Medium", "w"), ("E - Schlecht", "e")]),
            ("2 Darts:", [("A - Gut", "a"), ("S - Medium", "s"), ("D - Schlecht", "d")]),
            ("3 Darts:", [("Y - Gut", "y"), ("X - Medium", "x"), ("C - Schlecht", "c")])
        ]
        
        for col, (title, buttons) in enumerate(categories_info):
            col_frame = ttk.Frame(category_frame)
            col_frame.grid(row=0, column=col, padx=20, sticky="nsew")
            
            ttk.Label(col_frame, text=title, font=("Arial", 12, "bold")).pack(pady=(0, 5))
            
            for button_text, key in buttons:
                btn = ttk.Button(col_frame, text=button_text, 
                               command=lambda k=key: self.categorize_image(k))
                btn.pack(fill=tk.X, pady=2)
        
        # Navigation
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(nav_frame, text="← Vorheriges", command=self.previous_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="Nächstes →", command=self.next_image).pack(side=tk.RIGHT)
        ttk.Button(nav_frame, text="Überspringen", command=self.skip_image).pack()
        
        # Tastatur-Bindings
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        
        # Fenstergrößenänderung binden
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Erstes Bild laden
        if self.processed_images:
            self.load_current_image()
    
    def load_current_image(self):
        """Lädt das aktuelle Bild in die GUI."""
        if not self.processed_images or self.current_image_index >= len(self.processed_images):
            messagebox.showinfo("Fertig", "Alle Bilder wurden kategorisiert!")
            self.root.quit()
            return
        
        current_data = self.processed_images[self.current_image_index]
        self.current_image_path = current_data['processed_path']
        
        # Bild laden und anzeigen
        image = cv2.imread(self.current_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Canvas aktualisieren damit die Größe korrekt ermittelt wird
        self.root.update_idletasks()
        
        # Bildgröße an Canvas anpassen mit mehr Padding
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Fallback für initiale Größe falls Canvas noch nicht richtig gemessen wurde
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 800
            canvas_height = 600
        
        # Mehr Padding für bessere Darstellung
        max_width = canvas_width - 40
        max_height = canvas_height - 40
        
        # Bild skalieren falls es zu groß ist
        if pil_image.width > max_width or pil_image.height > max_height:
            pil_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Canvas leeren und Bild zentrieren
        self.canvas.delete("all")
        x = (canvas_width - pil_image.width) // 2
        y = (canvas_height - pil_image.height) // 2
        self.canvas.create_image(max(x, 20), max(y, 20), anchor=tk.NW, image=self.photo)
        
        # Informationen aktualisieren
        dart_count = current_data['dart_count']
        filename = os.path.basename(current_data['original_path'])
        self.info_label.config(text=f"Datei: {filename} | Erkannte Darts: {dart_count}")
        
        progress_text = f"Bild {self.current_image_index + 1} von {len(self.processed_images)}"
        self.progress_label.config(text=progress_text)
    
    def categorize_image(self, category_key):
        """Kategorisiert das aktuelle Bild."""
        if category_key not in self.categories:
            return
        
        category_path = self.categories[category_key]
        source_path = self.current_image_path
        
        # Zielordner
        dest_folder = os.path.join(self.output_base_folder, "categorized", category_path)
        dest_path = os.path.join(dest_folder, os.path.basename(source_path))
        
        try:
            # Bild kopieren
            shutil.copy2(source_path, dest_path)
            print(f"Bild kategorisiert: {category_path} - {os.path.basename(source_path)}")
            
            # Zum nächsten Bild
            self.next_image()
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Speichern: {str(e)}")
    
    def next_image(self):
        """Geht zum nächsten Bild."""
        self.current_image_index += 1
        self.load_current_image()
    
    def previous_image(self):
        """Geht zum vorherigen Bild."""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
    
    def skip_image(self):
        """Überspringt das aktuelle Bild."""
        self.next_image()
    
    def on_key_press(self, event):
        """Behandelt Tastatureingaben."""
        key = event.char.lower()
        if key in self.categories:
            self.categorize_image(key)
        elif event.keysym == 'Left':
            self.previous_image()
        elif event.keysym == 'Right':
            self.next_image()
        elif event.keysym == 'space':
            self.skip_image()
    
    def on_window_resize(self, event):
        """Behandelt Fenstergrößenänderungen und lädt das Bild neu."""
        # Nur auf Hauptfenster-Resize reagieren
        if event.widget == self.root and self.processed_images:
            # Kurze Verzögerung um mehrfache Aufrufe zu vermeiden
            self.root.after(100, self.load_current_image)
    
    def run(self):
        """Startet den kompletten Workflow."""
        print("Starte Bildverarbeitung...")
        
        # 1. Alle Bilder verarbeiten
        if not self.process_all_images():
            return
        
        if not self.processed_images:
            messagebox.showerror("Fehler", "Keine Bilder konnten verarbeitet werden.")
            return
        
        print("Starte GUI für Kategorisierung...")
        
        # 2. GUI für Kategorisierung starten
        self.setup_gui()
        self.root.mainloop()
        
        print("Kategorisierung abgeschlossen.")


def main():
    """Hauptfunktion."""
    try:
        processor = ImageProcessorGUI()
        processor.run()
    except Exception as e:
        print(f"Fehler: {str(e)}")
        messagebox.showerror("Fehler", f"Ein Fehler ist aufgetreten: {str(e)}")


if __name__ == "__main__":
    main()

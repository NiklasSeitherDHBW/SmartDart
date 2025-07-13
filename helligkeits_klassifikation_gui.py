import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import os
import shutil
from pathlib import Path
from PIL import Image, ImageTk

class HelligkeitsKlassifikationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Belichtungs-Klassifikation für Dartscheiben-Bilder")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Pfade
        self.source_folder = r"C:\Users\CARLO\Downloads\Darts-AL2-Stg3\images\val"
        self.output_base = r"c:\Users\CARLO\OneDrive\Desktop\Darts\klassifikation_dunkel"
        
        # Ausgabeordner erstellen
        self.create_output_folders()
        
        # Bildliste
        self.image_files = []
        self.current_index = 0
        self.load_images()
        
        # Statistiken
        self.stats = {"Normal": 0, "Besondere Belichtung": 0, "Übersprungen": 0}
        
        # GUI erstellen
        self.setup_gui()
        self.setup_keybindings()
        
        # Erstes Bild laden
        if self.image_files:
            self.show_current_image()
    
    def create_output_folders(self):
        """Erstellt die Ausgabeordner für die Klassifikation."""
        folders = ["normal", "besondere_belichtung"]
        for folder in folders:
            folder_path = os.path.join(self.output_base, folder)
            os.makedirs(folder_path, exist_ok=True)
    
    def load_images(self):
        """Lädt alle Bilddateien aus dem Quellordner."""
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        source_path = Path(self.source_folder)
        
        if not source_path.exists():
            messagebox.showerror("Fehler", f"Quellordner nicht gefunden: {self.source_folder}")
            return
        
        self.image_files = []
        for file_path in source_path.rglob('*'):
            if file_path.suffix.lower() in supported_formats:
                self.image_files.append(str(file_path))
        
        if not self.image_files:
            messagebox.showwarning("Warnung", "Keine Bilder im Quellordner gefunden!")
    
    def calculate_luminance(self, image_path):
        """Berechnet die neutrale Helligkeit basierend auf Luminanz."""
        img = cv2.imread(image_path)
        if img is None:
            return 0
        
        # Normalisierung auf 0-1 Bereich
        normalized = img.astype(np.float32) / 255.0
        
        # Gamma-Korrektur
        gamma = 2.2
        gamma_corrected = np.power(normalized, 1.0 / gamma)
        
        # Luminanz berechnen (ITU-R BT.709)
        luminance = (0.2126 * gamma_corrected[:,:,2] +  # R
                     0.7152 * gamma_corrected[:,:,1] +  # G
                     0.0722 * gamma_corrected[:,:,0])   # B
        
        return np.mean(luminance) * 255.0
    
    def setup_gui(self):
        """Erstellt die GUI-Elemente."""
        # Hauptframe
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Titel
        title_label = ttk.Label(main_frame, text="Belichtungs-Klassifikation", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Fortschrittsbereich
        progress_frame = ttk.LabelFrame(main_frame, text="Fortschritt", padding="10")
        progress_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        # Bildbereich
        image_frame = ttk.LabelFrame(main_frame, text="Aktuelles Bild", padding="10")
        image_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.grid(row=0, column=0)
        
        # Bildinformationen
        info_frame = ttk.Frame(image_frame)
        info_frame.grid(row=1, column=0, pady=(10, 0))
        
        self.filename_var = tk.StringVar()
        self.luminance_var = tk.StringVar()
        
        ttk.Label(info_frame, text="Datei:").grid(row=0, column=0, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.filename_var).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        ttk.Label(info_frame, text="Luminanz:").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(info_frame, textvariable=self.luminance_var).grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # Steuerungsbereich
        control_frame = ttk.LabelFrame(main_frame, text="Steuerung", padding="10")
        control_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Tastenkürzel-Anzeige
        shortcuts_frame = ttk.Frame(control_frame)
        shortcuts_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10))
        
        ttk.Label(shortcuts_frame, text="Tastenkürzel:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=3)
        ttk.Label(shortcuts_frame, text="[1] oder [N] - Normal").grid(row=1, column=0, sticky=tk.W)
        ttk.Label(shortcuts_frame, text="[2] oder [B] - Besondere Belichtung").grid(row=1, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(shortcuts_frame, text="[S] - Überspringen").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(shortcuts_frame, text="[←] - Vorheriges Bild").grid(row=2, column=1, sticky=tk.W, padx=(20, 0))
        ttk.Label(shortcuts_frame, text="[→] - Nächstes Bild").grid(row=2, column=2, sticky=tk.W, padx=(20, 0))
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(button_frame, text="Normal (1)", command=lambda: self.classify_image("normal")).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Besondere Belichtung (2)", command=lambda: self.classify_image("besondere_belichtung")).grid(row=0, column=1, padx=(0, 10))
        ttk.Button(button_frame, text="Überspringen (S)", command=self.skip_image).grid(row=0, column=2, padx=(10, 0))
        
        # Navigation
        nav_frame = ttk.Frame(control_frame)
        nav_frame.grid(row=2, column=0, columnspan=3, pady=(10, 0))
        
        ttk.Button(nav_frame, text="← Vorheriges", command=self.previous_image).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(nav_frame, text="Nächstes →", command=self.next_image).grid(row=0, column=1)
        
        # Statistiken
        stats_frame = ttk.LabelFrame(main_frame, text="Statistiken", padding="10")
        stats_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        self.stats_var = tk.StringVar()
        self.stats_label = ttk.Label(stats_frame, textvariable=self.stats_var)
        self.stats_label.grid(row=0, column=0)
        
        # Grid-Konfiguration
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
    
    def setup_keybindings(self):
        """Richtet die Tastenkürzel ein."""
        self.root.bind('<Key-1>', lambda e: self.classify_image("normal"))
        self.root.bind('<Key-2>', lambda e: self.classify_image("besondere_belichtung"))
        self.root.bind('<n>', lambda e: self.classify_image("normal"))
        self.root.bind('<N>', lambda e: self.classify_image("normal"))
        self.root.bind('<b>', lambda e: self.classify_image("besondere_belichtung"))
        self.root.bind('<B>', lambda e: self.classify_image("besondere_belichtung"))
        self.root.bind('<s>', lambda e: self.skip_image())
        self.root.bind('<S>', lambda e: self.skip_image())
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.focus_set()  # Fokus für Tasteneingaben
    
    def show_current_image(self):
        """Zeigt das aktuelle Bild in der GUI an."""
        if not self.image_files or self.current_index >= len(self.image_files):
            self.show_completion_message()
            return
        
        current_file = self.image_files[self.current_index]
        
        # Bild laden und skalieren
        try:
            # OpenCV für Bildladen
            img_cv = cv2.imread(current_file)
            if img_cv is None:
                raise Exception("Bild konnte nicht geladen werden")
            
            # Zu RGB konvertieren
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Skalieren (max 600x400)
            h, w = img_rgb.shape[:2]
            max_width, max_height = 600, 400
            
            if w > max_width or h > max_height:
                scale = min(max_width/w, max_height/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            
            # Zu PIL und Tkinter konvertieren
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil)
            
            # Bild anzeigen
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk  # Referenz behalten
            
            # Informationen aktualisieren
            filename = os.path.basename(current_file)
            self.filename_var.set(filename)
            
            luminance = self.calculate_luminance(current_file)
            self.luminance_var.set(f"{luminance:.1f}")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Laden des Bildes: {str(e)}")
            self.next_image()
            return
        
        # Fortschritt aktualisieren
        self.update_progress()
        self.update_stats()
    
    def classify_image(self, category):
        """Klassifiziert das aktuelle Bild und kopiert es in den entsprechenden Ordner."""
        if not self.image_files or self.current_index >= len(self.image_files):
            return
        
        current_file = self.image_files[self.current_index]
        filename = os.path.basename(current_file)
        
        # Zielordner bestimmen
        target_folder = os.path.join(self.output_base, category)
        target_file = os.path.join(target_folder, filename)
        
        try:
            # Datei kopieren
            shutil.copy2(current_file, target_file)
            
            # Statistik aktualisieren
            category_name = {"normal": "Normal", "besondere_belichtung": "Besondere Belichtung"}[category]
            self.stats[category_name] += 1
            
            print(f"Bild klassifiziert als '{category_name}': {filename}")
            
        except Exception as e:
            messagebox.showerror("Fehler", f"Fehler beim Kopieren: {str(e)}")
            return
        
        # Zum nächsten Bild
        self.next_image()
    
    def skip_image(self):
        """Überspringt das aktuelle Bild."""
        self.stats["Übersprungen"] += 1
        self.next_image()
    
    def next_image(self):
        """Geht zum nächsten Bild."""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_current_image()
        else:
            self.show_completion_message()
    
    def previous_image(self):
        """Geht zum vorherigen Bild."""
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()
    
    def update_progress(self):
        """Aktualisiert die Fortschrittsanzeige."""
        if not self.image_files:
            return
        
        total = len(self.image_files)
        current = self.current_index + 1
        
        self.progress_var.set(f"Bild {current} von {total}")
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = current
    
    def update_stats(self):
        """Aktualisiert die Statistikanzeige."""
        total_processed = sum(self.stats.values())
        stats_text = f"Bearbeitet: {total_processed} | "
        stats_text += f"Normal: {self.stats['Normal']} | "
        stats_text += f"Besondere Belichtung: {self.stats['Besondere Belichtung']} | "
        stats_text += f"Übersprungen: {self.stats['Übersprungen']}"
        
        self.stats_var.set(stats_text)
    
    def show_completion_message(self):
        """Zeigt eine Abschlussnachricht an."""
        total_classified = self.stats['Normal'] + self.stats['Besondere Belichtung']
        
        message = f"Klassifikation abgeschlossen!\n\n"
        message += f"Insgesamt klassifiziert: {total_classified} Bilder\n"
        message += f"Normal: {self.stats['Normal']}\n"
        message += f"Besondere Belichtung: {self.stats['Besondere Belichtung']}\n"
        message += f"Übersprungen: {self.stats['Übersprungen']}\n\n"
        message += f"Ergebnisse gespeichert in:\n{self.output_base}"
        
        messagebox.showinfo("Abgeschlossen", message)

def main():
    """Startet die Anwendung."""
    root = tk.Tk()
    app = HelligkeitsKlassifikationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

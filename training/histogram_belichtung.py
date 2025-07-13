import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

def calculate_brightness(image_path):
    """
    Berechnet die neutrale Helligkeit eines Bildes basierend auf Luminanz (Y-Wert).
    Unabhängig von Farbinformationen - nur reine Helligkeitsbewertung.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Normalisierung auf 0-1 Bereich (alle RGB-Kanäle)
    normalized = img.astype(np.float32) / 255.0
    
    # Gamma-Korrektur anwenden (Standard Gamma = 2.2 für sRGB)
    gamma = 2.2
    gamma_corrected = np.power(normalized, 1.0 / gamma)
    
    # Berechne Luminanz (Y-Wert) aus Gamma-korrigierten RGB-Werten
    # ITU-R BT.709 Standard für Luminanz: 0.2126*R + 0.7152*G + 0.0722*B
    luminance = (0.2126 * gamma_corrected[:,:,2] +  # R (BGR -> RGB)
                 0.7152 * gamma_corrected[:,:,1] +  # G
                 0.0722 * gamma_corrected[:,:,0])   # B
    
    # Durchschnittliche Luminanz (neutrale Helligkeit)
    brightness = np.mean(luminance) * 255.0
    return brightness

def calculate_exposure_value(image_path):
    """
    Berechnet einen neutralen Belichtungswert basierend auf Luminanz-Histogramm.
    Verwendet Luminanz für farbunabhängige Belichtungsbewertung.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Normalisierung auf 0-1 Bereich (alle RGB-Kanäle)
    normalized = img.astype(np.float32) / 255.0
    
    # Gamma-Korrektur anwenden
    gamma = 2.2
    gamma_corrected = np.power(normalized, 1.0 / gamma)
    
    # Berechne Luminanz (Y-Wert) aus den Gamma-korrigierten RGB-Werten
    # ITU-R BT.709 Standard für Luminanz: 0.2126*R + 0.7152*G + 0.0722*B
    luminance = (0.2126 * gamma_corrected[:,:,2] +  # R (BGR -> RGB)
                 0.7152 * gamma_corrected[:,:,1] +  # G  
                 0.0722 * gamma_corrected[:,:,0])   # B
    
    # Zurück zu 0-255 Bereich für Histogramm
    luminance_scaled = (luminance * 255).astype(np.uint8)
    
    # Histogramm der Luminanz berechnen
    hist = cv2.calcHist([luminance_scaled], [0], None, [256], [0, 256])
    
    # Gewichtete Durchschnittshelligkeit
    total_pixels = img.shape[0] * img.shape[1]
    weighted_sum = sum(i * hist[i][0] for i in range(256))
    exposure_value = weighted_sum / total_pixels
    
    return exposure_value

def analyze_images_in_folder(folder_path):
    """
    Analysiert alle Bilder in einem Ordner und gibt Listen der Belichtungswerte zurück.
    """
    brightness_values = []
    exposure_values = []
    image_paths = []
    
    # Unterstützte Bildformate
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Warnung: Ordner {folder_path} existiert nicht!")
        return brightness_values, exposure_values, image_paths
    
    for file_path in folder.rglob('*'):
        if file_path.suffix.lower() in supported_formats:
            brightness = calculate_brightness(str(file_path))
            exposure = calculate_exposure_value(str(file_path))
            
            if brightness is not None and exposure is not None:
                brightness_values.append(brightness)
                exposure_values.append(exposure)
                image_paths.append(str(file_path))
    
    return brightness_values, exposure_values, image_paths

def create_histogram():
    """
    Erstellt Histogramme für die Belichtungsverhältnisse der Dartscheiben-Bilder.
    """
    # Pfade zu den Bildordnern
    val_folder = r"C:\Users\CARLO\Downloads\Darts-AL2-Stg3\images\val"
    train_folder = r"C:\Users\CARLO\Downloads\Darts-AL2-Stg3\images\train"
    
    print("Analysiere Bilder...")
    
    # Analysiere Validierungsbilder
    val_brightness, val_exposure, val_paths = analyze_images_in_folder(val_folder)
    print(f"Validierungsbilder gefunden: {len(val_brightness)}")
    
    # Analysiere Trainingsbilder
    train_brightness, train_exposure, train_paths = analyze_images_in_folder(train_folder)
    print(f"Trainingsbilder gefunden: {len(train_brightness)}")
    
    # Kombiniere alle Werte
    all_brightness = val_brightness + train_brightness
    all_exposure = val_exposure + train_exposure
    
    if not all_brightness:
        print("Keine Bilder gefunden! Überprüfen Sie die Pfade.")
        return
    
    # Erstelle nur das Vergleichs-Diagramm: Training vs. Validierung
    output_base = r"c:\Users\CARLO\OneDrive\Desktop\Darts"
    
    # Vergleich: Validierungs- vs Trainingsbilder (Helligkeit)
    # Validierung wird in den Vordergrund gestellt
    if val_brightness and train_brightness:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Erst Trainingsbilder (im Hintergrund)
        ax.hist(train_brightness, bins=25, alpha=0.6, color='purple', 
                label=f'Training ({len(train_brightness)} Bilder)', edgecolor='black', linewidth=0.8)
        
        # Dann Validierungsbilder (im Vordergrund)
        ax.hist(val_brightness, bins=25, alpha=0.8, color='orange', 
                label=f'Validierung ({len(val_brightness)} Bilder)', edgecolor='black', linewidth=1.0)
        
        ax.set_title('Helligkeitsvergleich: Training vs. Validierung', fontsize=16, fontweight='bold')
        ax.set_xlabel('Helligkeit (0=Schwarz, 128=Mittelgrau, 255=Weiß)', fontsize=14)
        ax.set_ylabel('Anzahl Bilder', fontsize=14)
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Statistiken hinzufügen
        mean_val = np.mean(val_brightness)
        mean_train = np.mean(train_brightness)
        
        ax.axvline(mean_val, color='darkorange', linestyle='--', linewidth=2,
                   label=f'Mittelwert Validierung: {mean_val:.1f}')
        ax.axvline(mean_train, color='darkviolet', linestyle='--', linewidth=2,
                   label=f'Mittelwert Training: {mean_train:.1f}')
        
        # Legende aktualisieren
        ax.legend(fontsize=12, loc='upper right')
        
        plt.tight_layout()
        
        # Als PDF speichern
        pdf_path = os.path.join(output_base, "vergleich_train_val.pdf")
        plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
        print(f"📄 PDF gespeichert: {pdf_path}")
        
        plt.close()
        
    else:
        print("❌ Nicht genügend Daten für Vergleich verfügbar!")
        return
    
    # Zeige Statistiken
    print("\n" + "="*70)
    print("📊 BELICHTUNGSSTATISTIKEN")
    print("="*70)
    print(f"Gesamtanzahl Bilder: {len(all_brightness)}")
    print(f"Trainingsbilder: {len(train_brightness)}")
    print(f"Validierungsbilder: {len(val_brightness)}")
    print("\n📊 Erstelle Vergleichsdiagramm Training vs. Validierung...")
    
    if not val_brightness or not train_brightness:
        print("❌ Nicht genügend Daten für Vergleich verfügbar!")
        return
    
    print("✅ Vergleichsdiagramm erfolgreich erstellt!")
    print("\n" + "="*50)
    print("📄 VERGLEICHSDIAGRAMM ERSTELLT:")
    print("="*50)
    print("📊 vergleich_train_val.pdf - Training vs. Validierung")
    print("\n✅ Vergleichsdiagramm erfolgreich erstellt!")

if __name__ == "__main__":
    create_histogram()

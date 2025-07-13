import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

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
    
    # Erstelle Subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Neutrale Belichtungsanalyse der Dartscheiben-Bilder (Luminanz-basiert)', fontsize=16, fontweight='bold')
    
    # 1. Histogramm: Neutrale Helligkeit (alle Bilder)
    axes[0, 0].hist(all_brightness, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Neutrale Helligkeit - Luminanz (Alle Bilder)')
    axes[0, 0].set_xlabel('Luminanz-basierte Helligkeit (0-255)')
    axes[0, 0].set_ylabel('Anzahl Bilder')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Statistiken hinzufügen
    mean_brightness = np.mean(all_brightness)
    std_brightness = np.std(all_brightness)
    axes[0, 0].axvline(mean_brightness, color='red', linestyle='--', 
                       label=f'Mittelwert: {mean_brightness:.1f}')
    axes[0, 0].legend()
    
    # 2. Histogramm: Neutrale Belichtungswerte (alle Bilder)
    axes[0, 1].hist(all_exposure, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Neutrale Belichtungswerte - Luminanz (Alle Bilder)')
    axes[0, 1].set_xlabel('Luminanz-basierter Belichtungswert')
    axes[0, 1].set_ylabel('Anzahl Bilder')
    axes[0, 1].grid(True, alpha=0.3)
    
    mean_exposure = np.mean(all_exposure)
    axes[0, 1].axvline(mean_exposure, color='red', linestyle='--',
                       label=f'Mittelwert: {mean_exposure:.1f}')
    axes[0, 1].legend()
    
    # 3. Vergleich: Validierungs- vs Trainingsbilder (Helligkeit)
    if val_brightness and train_brightness:
        axes[1, 0].hist(val_brightness, bins=20, alpha=0.7, color='orange', 
                        label=f'Validierung ({len(val_brightness)})', edgecolor='black')
        axes[1, 0].hist(train_brightness, bins=20, alpha=0.7, color='purple', 
                        label=f'Training ({len(train_brightness)})', edgecolor='black')
        axes[1, 0].set_title('Neutrale Helligkeit: Training vs. Validierung')
        axes[1, 0].set_xlabel('Luminanz-basierte Helligkeit (0-255)')
        axes[1, 0].set_ylabel('Anzahl Bilder')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Belichtungskategorien
    categories = []
    for brightness in all_brightness:
        if brightness < 85:
            categories.append('Dunkel')
        elif brightness < 170:
            categories.append('Normal')
        else:
            categories.append('Hell')
    
    category_counts = {cat: categories.count(cat) for cat in ['Dunkel', 'Normal', 'Hell']}
    colors = ['darkblue', 'lightblue', 'yellow']
    
    axes[1, 1].bar(category_counts.keys(), category_counts.values(), 
                   color=colors, edgecolor='black', alpha=0.8)
    axes[1, 1].set_title('Belichtungskategorien')
    axes[1, 1].set_ylabel('Anzahl Bilder')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Prozentuale Anteile hinzufügen
    total_images = len(all_brightness)
    for i, (category, count) in enumerate(category_counts.items()):
        percentage = (count / total_images) * 100
        axes[1, 1].text(i, count + 1, f'{percentage:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Speichere das Histogramm
    output_path = r"c:\Users\CARLO\OneDrive\Desktop\Darts\belichtung_histogramm.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHistogramm gespeichert unter: {output_path}")
    
    # Zeige Statistiken
    print("\n" + "="*50)
    print("BELICHTUNGSSTATISTIKEN")
    print("="*50)
    print(f"Gesamtanzahl Bilder: {total_images}")
    print(f"Trainingsbilder: {len(train_brightness)}")
    print(f"Validierungsbilder: {len(val_brightness)}")
    print(f"\nDurchschnittliche neutrale Helligkeit (Luminanz): {mean_brightness:.2f} ± {std_brightness:.2f}")
    print(f"Minimale neutrale Helligkeit: {min(all_brightness):.2f}")
    print(f"Maximale neutrale Helligkeit: {max(all_brightness):.2f}")
    print(f"\nBelichtungskategorien (basierend auf neutraler Luminanz):")
    for category, count in category_counts.items():
        percentage = (count / total_images) * 100
        print(f"  {category}: {count} Bilder ({percentage:.1f}%)")
    
    # Zeige das Histogramm
    plt.show()

if __name__ == "__main__":
    create_histogram()

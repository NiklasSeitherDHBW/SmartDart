"""
Einfache Accuracy-Berechnung für YOLO Dart Detection Models

Berechnet Accuracy nach der Formel:
Accuracy = (Gut_1Dart + Gut_2Dart + Gut_3Dart) / (Total_1Dart + Total_2Dart + Total_3Dart)

Autor: Model Evaluation Script
Datum: 2025-01-13
"""

import os
import glob
from pathlib import Path


def count_images_in_folder(folder_path):
    """Zählt alle Bilddateien in einem Ordner."""
    if not os.path.exists(folder_path):
        return 0
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    count = 0
    
    for ext in image_extensions:
        # Nur Kleinbuchstaben zählen (um Duplikate zu vermeiden)
        files = glob.glob(os.path.join(folder_path, ext))
        count += len(files)
    
    return count


def analyze_model(model_name, base_path):
    """Analysiert ein einzelnes Modell."""
    model_folder = os.path.join(base_path, f"processed_images_dunkel_{model_name}")
    categorized_path = os.path.join(model_folder, "categorized")
    
    if not os.path.exists(categorized_path):
        print(f"❌ Ordner nicht gefunden: {categorized_path}")
        return None
    
    results = {}
    total_gut = 0
    total_bilder = 0
    
    # Für jede Dart-Anzahl (1Dart, 2Dart, 3Dart)
    for dart_count in ["1Dart", "2Dart", "3Dart"]:
        dart_path = os.path.join(categorized_path, dart_count)
        
        if os.path.exists(dart_path):
            # Zähle Bilder in jeder Kategorie
            gut = count_images_in_folder(os.path.join(dart_path, "gut"))
            medium = count_images_in_folder(os.path.join(dart_path, "medium"))
            schlecht = count_images_in_folder(os.path.join(dart_path, "schlecht"))
            
            total_dart = gut + medium + schlecht
            
            results[dart_count] = {
                'gut': gut,
                'medium': medium,
                'schlecht': schlecht,
                'total': total_dart
            }
            
            total_gut += gut
            total_bilder += total_dart
            
            print(f"  {dart_count}: {total_dart} Bilder (gut: {gut}, medium: {medium}, schlecht: {schlecht})")
        else:
            print(f"  {dart_count}: Ordner nicht gefunden")
            results[dart_count] = {'gut': 0, 'medium': 0, 'schlecht': 0, 'total': 0}
    
    # Accuracy berechnen
    if total_bilder > 0:
        accuracy = (total_gut / total_bilder) * 100
    else:
        accuracy = 0.0
    
    results['summary'] = {
        'total_gut': total_gut,
        'total_bilder': total_bilder,
        'accuracy': accuracy
    }
    
    return results


def main():
    """Hauptfunktion für die Accuracy-Berechnung."""
    base_path = r"C:\Users\CARLO\OneDrive\Desktop\Darts"
    
    # Finde alle Modell-Ordner
    model_folders = []
    for folder in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, folder)) and folder.startswith("processed_images_dunkel_"):
            # Extrahiere den Modellnamen (alles nach "processed_images_helligkeit_")
            model_name = folder.replace("processed_images_dunkel_", "")
            model_folders.append(model_name)
    
    if not model_folders:
        print("❌ Keine Modell-Ordner gefunden!")
        return
    
    model_folders.sort()
    
    print("🎯 ACCURACY-BERECHNUNG FÜR YOLO DART DETECTION MODELS")
    print("=" * 60)
    print()
    print("📋 FORMEL:")
    print("Accuracy = (Gut_1Dart + Gut_2Dart + Gut_3Dart) / (Total_1Dart + Total_2Dart + Total_3Dart)")
    print()
    
    all_results = {}
    
    # Analysiere jedes Modell
    for model_name in model_folders:
        print(f"📊 Analysiere Modell: {model_name}")
        
        results = analyze_model(model_name, base_path)
        
        if results:
            all_results[model_name] = results
            summary = results['summary']
            
            print(f"  📈 Ergebnis:")
            print(f"    Gesamt 'gut' Bilder: {summary['total_gut']}")
            print(f"    Gesamt Bilder: {summary['total_bilder']}")
            print(f"    🎯 ACCURACY: {summary['accuracy']:.2f}%")
        else:
            print(f"  ❌ Konnte Modell {model_name} nicht analysieren")
        
        print()
    
    # Zusammenfassung aller Modelle
    if all_results:
        print("📊 ZUSAMMENFASSUNG ALLER MODELLE:")
        print("=" * 60)
        
        # Sortiere nach Accuracy
        sorted_models = sorted(all_results.items(), 
                             key=lambda x: x[1]['summary']['accuracy'], 
                             reverse=True)
        
        print(f"{'Modell':<35} {'Gut':<8} {'Gesamt':<8} {'Accuracy':<12}")
        print("-" * 65)
        
        for model_name, results in sorted_models:
            summary = results['summary']
            print(f"{model_name:<35} {summary['total_gut']:<8} {summary['total_bilder']:<8} {summary['accuracy']:<12.2f}%")
        
        print()
        print("🏆 RANKING:")
        for i, (model_name, results) in enumerate(sorted_models, 1):
            accuracy = results['summary']['accuracy']
            print(f"  {i}. {model_name}: {accuracy:.2f}%")
        
        print()
        print("✅ Analyse abgeschlossen!")
    
    return all_results


if __name__ == "__main__":
    main()

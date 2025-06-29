import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import cv2
import threading
import time
import json
import math
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Import your existing modules
from utils import camera, calibration, predict, score_prediction


class Player:
    """Repräsentiert einen einzelnen Spieler im Spiel."""
    
    def __init__(self, name: str, player_id: int):
        self.name = name
        self.id = player_id
        self.current_score = 501  # Standard für 501-Spiel
        self.darts_thrown = 0
        self.turn_scores = []  # Liste der Punkte pro Runde
        self.game_history = []  # Historie aller Spiele
        self.legs_won = 0
        self.sets_won = 0
        
    def reset_game(self, starting_score: int = 501):
        """Spieler für ein neues Spiel zurücksetzen."""
        self.current_score = starting_score
        self.darts_thrown = 0
        self.turn_scores = []
    
    def add_turn_score(self, scores: List[int]):
        """Punkte aus einer Runde hinzufügen (bis zu 3 Darts)."""
        turn_total = sum(scores)
        self.turn_scores.append(scores)
        self.darts_thrown += len(scores)
        return turn_total
    
    def get_average(self) -> float:
        """Durchschnittliche Punkte pro Dart berechnen."""
        if self.darts_thrown == 0:
            return 0.0
        total_score = 501 - self.current_score
        return total_score / self.darts_thrown
    
    def can_finish_with_score(self, score: int) -> bool:
        """Prüfen ob der Spieler mit dieser Punktzahl finishen kann."""
        return self.current_score == score and score <= 170
    
    def is_bust(self, score: int) -> bool:
        """Prüfen ob die Punktzahl ein Bust ist."""
        remaining = self.current_score - score
        return remaining < 0 or remaining == 1
    
    def to_dict(self) -> dict:
        """Spieler in Dictionary für Speicherung konvertieren."""
        return {
            'name': self.name,
            'id': self.id,
            'current_score': self.current_score,
            'darts_thrown': self.darts_thrown,
            'turn_scores': self.turn_scores,
            'game_history': self.game_history,
            'legs_won': self.legs_won,
            'sets_won': self.sets_won
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Player':
        """Spieler aus Dictionary erstellen."""
        player = cls(data['name'], data['id'])
        player.current_score = data.get('current_score', 501)
        player.darts_thrown = data.get('darts_thrown', 0)
        player.turn_scores = data.get('turn_scores', [])
        player.game_history = data.get('game_history', [])
        player.legs_won = data.get('legs_won', 0)
        player.sets_won = data.get('sets_won', 0)
        return player


class GameState:
    """Verwaltet den gesamten Spielzustand."""
    
    def __init__(self):
        self.players: List[Player] = []
        self.current_player_index = 0
        self.current_dart_count = 0  # Geworfene Darts in aktueller Runde (0-3)
        self.current_turn_scores = []  # Punkte für aktuelle Runde
        self.game_mode = "501"  # 501, Cricket, Around the Clock, etc.
        self.game_active = False
        self.winner: Optional[Player] = None
        self.legs_to_win = 3
        self.sets_to_win = 2
        self.auto_advance = True  # Automatisch zur nächsten Runde nach 3 Darts
        
    def add_player(self, name: str) -> Player:
        """Neuen Spieler zum Spiel hinzufügen."""
        player_id = len(self.players) + 1
        player = Player(name, player_id)
        self.players.append(player)
        return player
    
    def remove_player(self, player: Player):
        """Spieler aus dem Spiel entfernen."""
        if player in self.players:
            self.players.remove(player)
            # Index anpassen falls nötig
            if self.current_player_index >= len(self.players):
                self.current_player_index = 0
    
    def start_game(self, game_mode: str = "501"):
        """Neues Spiel starten."""
        if len(self.players) < 2:
            raise ValueError("Mindestens 2 Spieler erforderlich")
        
        self.game_mode = game_mode
        self.current_player_index = 0
        self.current_dart_count = 0
        self.current_turn_scores = []
        self.game_active = True
        self.winner = None
        
        # Alle Spieler zurücksetzen
        starting_score = 501 if game_mode == "501" else 0
        for player in self.players:
            player.reset_game(starting_score)
    
    def get_current_player(self) -> Optional[Player]:
        """Aktuell aktiven Spieler abrufen."""
        if not self.players:
            return None
        return self.players[self.current_player_index]
    
    def add_dart_score(self, score: int, description: str = "") -> bool:
        """Dart-Punktzahl für aktuellen Spieler hinzufügen. Gibt True zurück wenn Runde komplett ist."""
        if not self.game_active or not self.players:
            print(f"add_dart_score: Spiel nicht aktiv oder keine Spieler")
            return False
        
        current_player = self.get_current_player()
        if not current_player:
            print(f"add_dart_score: Kein aktueller Spieler")
            return False
            
        print(f"add_dart_score: {score} für {current_player.name} (Dart {self.current_dart_count + 1}/3)")
        
        self.current_turn_scores.append(score)
        self.current_dart_count += 1
        
        # Prüfen ob Runde komplett ist (3 Darts oder Bust oder Finish)
        turn_complete = False
        if self.current_dart_count >= 3:
            turn_complete = True
            print(f"add_dart_score: Runde komplett (3 Darts)")
        elif self.game_mode == "501":
            # Prüfen auf Bust oder Finish
            turn_total = sum(self.current_turn_scores)
            if current_player.is_bust(turn_total):
                turn_complete = True
                print(f"add_dart_score: Bust mit {turn_total} Punkten")
            elif current_player.current_score == turn_total:
                # Gewonnen!
                current_player.current_score = 0
                self.winner = current_player
                self.game_active = False
                turn_complete = True
                print(f"add_dart_score: {current_player.name} hat gewonnen!")
        
        print(f"add_dart_score: turn_complete = {turn_complete}")
        return turn_complete
    
    def complete_turn(self):
        """Runde des aktuellen Spielers abschließen."""
        if not self.players:
            return
        
        current_player = self.get_current_player()
        turn_total = sum(self.current_turn_scores)
        
        print(f"complete_turn: {current_player.name}, {turn_total} Punkte, vorher: {current_player.current_score}")
        
        if self.game_mode == "501":
            if not current_player.is_bust(turn_total):
                current_player.current_score -= turn_total
                print(f"complete_turn: Nach Abzug: {current_player.current_score}")
            else:
                print(f"complete_turn: Bust! Punkte bleiben bei {current_player.current_score}")
        
        # Runde aufzeichnen
        current_player.add_turn_score(self.current_turn_scores.copy())
        
        # Zum nächsten Spieler wechseln
        old_player = self.current_player_index
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        new_player = self.get_current_player()
        print(f"complete_turn: Wechsel von {current_player.name} zu {new_player.name}")
        
        self.current_dart_count = 0
        self.current_turn_scores = []
    
    def undo_last_dart(self):
        """Letzten Dart rückgängig machen."""
        if self.current_dart_count > 0:
            self.current_turn_scores.pop()
            self.current_dart_count -= 1
    
    def save_game(self, filepath: str):
        """Aktuellen Spielzustand in Datei speichern."""
        data = {
            'players': [player.to_dict() for player in self.players],
            'current_player_index': self.current_player_index,
            'game_mode': self.game_mode,
            'game_active': self.game_active,
            'legs_to_win': self.legs_to_win,
            'sets_to_win': self.sets_to_win,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_game(self, filepath: str):
        """Spielzustand aus Datei laden."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.players = [Player.from_dict(p) for p in data['players']]
        self.current_player_index = data.get('current_player_index', 0)
        self.game_mode = data.get('game_mode', '501')
        self.game_active = data.get('game_active', False)
        self.legs_to_win = data.get('legs_to_win', 3)
        self.sets_to_win = data.get('sets_to_win', 2)


class DartsGUI:
    """Haupt-GUI-Anwendung für das Dart-Spiel."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 SmartDart")
        self.root.geometry("1600x900")
        self.root.configure(bg='#2C3E50')
        
        # Spielzustand
        self.game_state = GameState()
        self.debug_mode = True
        
        # Computer Vision Komponenten
        self.camera = None
        self.calibration = None
        self.predictor = None
        self.score_predictor = None
        self.camera_thread = None
        self.camera_running = False
        
        # GUI Variablen
        self.video_label = None
        self.player_frames = {}
        self.status_var = tk.StringVar(value="Willkommen zum Darts Turnier!")
        self.turn_info_var = tk.StringVar(value="Spieler hinzufügen zum Starten")
        self.dartboard_calibrated = False
        
        # Kamera Settings
        self.camera_source = 0
        self.use_image_folder = False
        self.image_folder_path = "training/data/transferlearning/stg1/raw"
        
        # Dart-Detection Cache für konsistente Anzeige
        self.last_dart_positions = []       # Für Anzeige zwischen Frames
        self.processed_dart_positions = []  # Für Anti-Duplikat-Logik
        self.blacklisted_dart_positions = []  # Für gesamten Zug ignorierte Dart-Positionen
        self.last_dart_scores = []
        
        # Dart-Erkennung Stabilisierung
        self.dart_detection_cooldown = 0  # Frames bis zur nächsten Detection
        self.stable_dart_positions = []  # Stabile Dart-Positionen
        self.detection_confirmation_frames = 3  # Frames für Bestätigung
        self.dart_position_tolerance = 30  # Pixel-Toleranz für gleiche Position
        
        # Anti-Spam für Dart-Verarbeitung
        self._currently_processing_dart = False  # Verhindert gleichzeitige Dart-Verarbeitung
        
        # Cooldown nach 3 Darts
        self.turn_complete_cooldown = 0  # Cooldown nach kompletter Runde (in Frames)
        self.turn_complete_cooldown_duration = 300  # 10 Sekunden bei 30 FPS
        self.board_empty_check_frames = 0  # Frames ohne Dart-Erkennung
        self.board_empty_required_frames = 30  # 1 Sekunde ohne Darts = Board leer
        self.turn_ready_to_complete = False  # Flag dass Runde bereit zum Abschluss ist
        
        # Create GUI
        self.setup_gui()
        self.setup_computer_vision()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_gui(self):
        """Haupt-GUI-Layout einrichten."""
        self.create_menu()
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
    def create_menu(self):
        """Menüleiste erstellen."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Spiel Menü
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Spiel", menu=game_menu)
        game_menu.add_command(label="Neues Spiel", command=self.new_game)
        game_menu.add_command(label="Spiel speichern", command=self.save_game)
        game_menu.add_command(label="Spiel laden", command=self.load_game)
        game_menu.add_separator()
        game_menu.add_command(label="Beenden", command=self.root.quit)
        
        # Spieler Menü
        players_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Spieler", menu=players_menu)
        players_menu.add_command(label="Spieler hinzufügen", command=self.add_player)
        players_menu.add_command(label="Spieler entfernen", command=self.remove_selected_player)
        
        # Einstellungen Menü
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Einstellungen", menu=settings_menu)
        settings_menu.add_command(label="Kamera-Einstellungen", command=self.camera_settings)
        settings_menu.add_command(label="Spiel-Einstellungen", command=self.game_settings)
        settings_menu.add_separator()
        settings_menu.add_command(label="Dartboard kalibrieren", command=self.calibrate_dartboard)
        settings_menu.add_command(label="Kalibrierung zurücksetzen", command=self.reset_calibration)
        settings_menu.add_separator()
        settings_menu.add_command(label="Debug umschalten", command=self.toggle_debug)
    
    def create_header(self):
        """Header mit Spielinfo erstellen."""
        header_frame = tk.Frame(self.root, bg='#34495E', height=70)
        header_frame.pack(fill='x', padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        # Spiel Titel
        title_label = tk.Label(header_frame, text="🎯 SmartDart", 
                              font=('Arial', 24, 'bold'), 
                              fg='white', bg='#34495E')
        title_label.pack(side='left', padx=20, pady=20)
        
        # Runden Info
        turn_frame = tk.Frame(header_frame, bg='#34495E')
        turn_frame.pack(side='right', padx=20, pady=20)
        
        turn_label = tk.Label(turn_frame, textvariable=self.turn_info_var,
                             font=('Arial', 16, 'bold'),
                             fg='#E74C3C', bg='#34495E')
        turn_label.pack()
    
    def create_main_content(self):
        """Hauptinhalt-Bereich erstellen."""
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Linkes Panel - Kamera und Steuerung
        left_panel = tk.Frame(main_frame, bg='#34495E', width=800)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.create_camera_panel(left_panel)
        self.create_control_panel(left_panel)
        
        # Rechtes Panel - Spieler und Punkte
        right_panel = tk.Frame(main_frame, bg='#34495E', width=700)
        right_panel.pack(side='right', fill='both', padx=(5, 0))
        right_panel.pack_propagate(False)
        
        self.create_players_panel(right_panel)
    
    def create_camera_panel(self, parent):
        """Kamera-Anzeige-Panel erstellen."""
        camera_frame = tk.LabelFrame(parent, text="Kamera Feed", 
                                   font=('Arial', 14, 'bold'),
                                   fg='white', bg='#34495E', height=450)
        camera_frame.pack(fill='x', expand=False, padx=10, pady=10)
        camera_frame.pack_propagate(False)  # Größe fixieren
        
        # Kamera Anzeige
        self.video_label = tk.Label(camera_frame, bg='black', 
                                   text="Kamera nicht initialisiert\nKlicken Sie 'Kamera starten'",
                                   fg='white', font=('Arial', 16))
        self.video_label.pack(padx=5, pady=5)
    
    def create_control_panel(self, parent):
        """Spiel-Steuerungs-Panel erstellen."""
        control_frame = tk.LabelFrame(parent, text="Spiel Steuerung", 
                                    font=('Arial', 14, 'bold'),
                                    fg='white', bg='#34495E', height=140)
        control_frame.pack(fill='x', padx=10, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        # Button Frame
        button_frame = tk.Frame(control_frame, bg='#34495E')
        button_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Kamera Steuerung
        camera_btn_frame = tk.Frame(button_frame, bg='#34495E')
        camera_btn_frame.pack(side='left', fill='y')
        
        self.camera_btn = tk.Button(camera_btn_frame, text="Kamera starten", 
                                   font=('Arial', 11, 'bold'),
                                   bg='#27AE60', fg='white',
                                   command=self.toggle_camera, width=14)
        self.camera_btn.pack(pady=2)
        
        calibrate_btn = tk.Button(camera_btn_frame, text="Kalibrieren", 
                                font=('Arial', 11, 'bold'),
                                bg='#3498DB', fg='white',
                                command=self.calibrate_dartboard, width=14)
        calibrate_btn.pack(pady=2)
        
        reset_cal_btn = tk.Button(camera_btn_frame, text="Kalibrierung Reset", 
                                font=('Arial', 9, 'bold'),
                                bg='#E67E22', fg='white',
                                command=self.reset_calibration, width=14)
        reset_cal_btn.pack(pady=2)
        
        # Spiel Steuerung
        game_btn_frame = tk.Frame(button_frame, bg='#34495E')
        game_btn_frame.pack(side='left', fill='y', padx=(20, 0))
        
        new_game_btn = tk.Button(game_btn_frame, text="Neues Spiel", 
                               font=('Arial', 11, 'bold'),
                               bg='#E67E22', fg='white',
                               command=self.new_game, width=14)
        new_game_btn.pack(pady=2)
        
        self.next_turn_btn = tk.Button(game_btn_frame, text="Nächste Runde", 
                                     font=('Arial', 11, 'bold'),
                                     bg='#9B59B6', fg='white',
                                     command=self.next_turn, width=14)
        self.next_turn_btn.pack(pady=2)
        
        # Manuelle Punkteeingabe
        manual_frame = tk.Frame(button_frame, bg='#34495E')
        manual_frame.pack(side='right', fill='y')
        
        tk.Label(manual_frame, text="Manuelle Punkte:", 
                font=('Arial', 10), fg='white', bg='#34495E').pack()
        
        score_entry_frame = tk.Frame(manual_frame, bg='#34495E')
        score_entry_frame.pack()
        
        self.manual_score_var = tk.StringVar()
        score_entry = tk.Entry(score_entry_frame, textvariable=self.manual_score_var,
                              font=('Arial', 11), width=8)
        score_entry.pack(side='left')
        
        add_score_btn = tk.Button(score_entry_frame, text="➕", 
                                font=('Arial', 10, 'bold'),
                                bg='#2ECC71', fg='white',
                                command=self.add_manual_score, width=3)
        add_score_btn.pack(side='left', padx=(2, 0))
        
        undo_btn = tk.Button(manual_frame, text="↶ Rückgängig", 
                           font=('Arial', 10, 'bold'),
                           bg='#E74C3C', fg='white',
                           command=self.undo_last_dart, width=12)
        undo_btn.pack(pady=(5, 0))
    
    def create_players_panel(self, parent):
        """Spieler-Panel erstellen."""
        players_frame = tk.LabelFrame(parent, text="Spieler & Punktestände", 
                                    font=('Arial', 14, 'bold'),
                                    fg='white', bg='#34495E')
        players_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrollbarer Bereich für Spieler
        canvas = tk.Canvas(players_frame, bg='#34495E', highlightthickness=0)
        scrollbar = ttk.Scrollbar(players_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg='#34495E')
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scrollbar.pack(side="right", fill="y")
        
        # Spieler hinzufügen Button
        add_player_frame = tk.Frame(self.scrollable_frame, bg='#34495E')
        add_player_frame.pack(fill='x', pady=5)
        
        add_player_btn = tk.Button(add_player_frame, text="➕ Spieler hinzufügen", 
                                 font=('Arial', 12, 'bold'),
                                 bg='#3498DB', fg='white',
                                 command=self.add_player)
        add_player_btn.pack(pady=5)
        
    def create_status_bar(self):
        """Status-Leiste erstellen."""
        status_frame = tk.Frame(self.root, bg='#34495E', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=('Arial', 10), fg='white', bg='#34495E')
        status_label.pack(side='left', padx=10, pady=5)
        
        # Dartboard Status
        self.dartboard_status_var = tk.StringVar(value="Dartboard: Nicht kalibriert")
        dartboard_status = tk.Label(status_frame, textvariable=self.dartboard_status_var,
                                   font=('Arial', 10), fg='orange', bg='#34495E')
        dartboard_status.pack(side='right', padx=10, pady=5)
    
    def setup_computer_vision(self):
        """Computer Vision Komponenten einrichten."""
        try:
            # Initialize components but don't start camera yet
            self.predictor = predict.Predictor(model_path="models/yolo8n-pretrained-al2-stg3.pt")
            self.score_predictor = score_prediction.DartboardScorePredictor()
            self.update_status("Computer Vision Komponenten geladen")
        except Exception as e:
            self.update_status(f"Fehler beim Laden der CV-Komponenten: {e}")
    
    def toggle_camera(self):
        """Kamera ein/aus schalten."""
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Kamera starten."""
        try:
            # Stop existing camera first
            if self.camera_running:
                self.stop_camera()
                time.sleep(0.5)  # Give time for cleanup
            
            if self.use_image_folder:
                # Use image folder as source
                self.camera = camera.VideoStreamViewer(source=Path(self.image_folder_path))
            else:
                # Use webcam
                self.camera = camera.VideoStreamViewer(source=self.camera_source)
            
            self.camera.open_connection()
            if not self.camera.isOpened():
                messagebox.showerror("Fehler", "Kamera konnte nicht geöffnet werden")
                return
            
            # Test if camera can provide a frame
            test_frame = self.camera.get_frame_raw()
            if test_frame is None:
                messagebox.showerror("Fehler", "Kamera liefert keine Bilder")
                self.camera.release()
                return
            
            self.camera_running = True
            self.camera_btn.configure(text="Kamera stoppen", bg='#E74C3C')
            
            # Initialize calibration components (but don't calibrate yet)
            self.calibration = calibration.CameraCalibration(
                ref_img="resources/dartboard-gerade.jpg", 
                debug=self.debug_mode
            )
            
            # Set initial status (not calibrated)
            self.dartboard_calibrated = False
            self.dartboard_status_var.set("Dartboard: Nicht kalibriert")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.update_status("Kamera gestartet - Drücken Sie 'Kalibrieren' für Dartboard-Kalibrierung")
            
        except Exception as e:
            error_msg = f"Fehler beim Starten der Kamera: {e}"
            messagebox.showerror("Fehler", error_msg)
            self.update_status(f"Kamera-Fehler: {e}")
            
            # Cleanup on error
            try:
                if hasattr(self, 'camera') and self.camera:
                    self.camera.release()
                self.camera_running = False
                self.camera_btn.configure(text="Kamera starten", bg='#27AE60')
            except:
                pass
    
    def stop_camera(self):
        """Kamera stoppen."""
        self.camera_running = False
        if self.camera:
            self.camera.release()
        
        self.camera_btn.configure(text="Kamera starten", bg='#27AE60')
        self.video_label.configure(image='', text="Kamera gestoppt\nKlicken Sie 'Kamera starten'")
        
        self.update_status("Kamera gestoppt")
    
    def camera_loop(self):
        """Hauptschleife für Kamera-Verarbeitung."""
        frame_count = 0
        
        while self.camera_running:
            try:
                frame = self.camera.get_frame_raw()
                if frame is None:
                    time.sleep(0.033)  # Warte kurz wenn kein Frame verfügbar
                    continue
                
                frame_count += 1
                
                # Frame verarbeiten - entweder kalibriert oder original
                if self.dartboard_calibrated and self.calibration and self.calibration.H is not None:
                    # Verwende gespeicherte Kalibrierung für alle Frames
                    try:
                        processed_frame = self.calibration.warp_frame(frame)
                        if processed_frame is None:
                            processed_frame = frame
                    except Exception as warp_e:
                        print(f"Warping-Fehler: {warp_e}")
                        processed_frame = frame
                else:
                    # Zeige Original-Frame wenn nicht kalibriert
                    processed_frame = frame
                
                # Performance: Nur jeden 3. Frame vollständig verarbeiten für YOLO
                full_processing = (frame_count % 3 == 0)
                
                # Cooldown-Logik verwalten
                if self.dart_detection_cooldown > 0:
                    self.dart_detection_cooldown -= 1
                
                if self.turn_complete_cooldown > 0:
                    self.turn_complete_cooldown -= 1
                    # Update Turn-Display während Cooldown
                    if frame_count % 10 == 0:  # Alle 10 Frames (ca. 3x pro Sekunde)
                        self.root.after(0, self.update_turn_display)
                    
                    # Automatisches Ende des Cooldowns nach Ablauf der Zeit
                    if self.turn_complete_cooldown <= 0:
                        print("⏰ Turn-Complete-Cooldown automatisch abgelaufen!")
                        self.board_empty_check_frames = 0
                        self.reset_dart_detection_state()
                        
                        # Prüfe ob Runde bereit zum Abschluss ist und schließe sie ab
                        if self.turn_ready_to_complete:
                            print("⏰ Cooldown abgelaufen und Runde bereit - schließe Runde ab")
                            self.turn_ready_to_complete = False
                            self.root.after(0, self.complete_current_turn)
                        
                        # Update Turn-Display sofort
                        self.root.after(0, self.update_turn_display)
                
                # Create display frame
                display_frame = processed_frame.copy()
                
                # Overlay dartboard template IMMER wenn kalibriert (nicht nur bei YOLO-Frames)
                if self.score_predictor and self.score_predictor.is_calibrated():
                    try:
                        display_frame = self.score_predictor.overlay_dartboard_template(
                            display_frame, 
                            show_numbers=True,
                            template_color=(0, 255, 255)
                        )
                    except Exception as overlay_e:
                        print(f"Overlay-Fehler: {overlay_e}")
                        display_frame = processed_frame
                
                # Zeige gespeicherte Dart-Scores bei ALLEN Frames (nicht nur bei YOLO-Frames)
                if (self.score_predictor and self.score_predictor.is_calibrated() and 
                    self.last_dart_positions and not full_processing):
                    try:
                        # Verwende gespeicherte Dart-Positionen für konsistente Anzeige
                        display_frame, _ = self.score_predictor.process_dart_detections(
                            display_frame, 
                            self.last_dart_positions, 
                            show_scores=True
                        )
                    except Exception as cached_detection_e:
                        print(f"Cached Dart-Detection Fehler: {cached_detection_e}")
                
                # Nur bei jedem 3. Frame YOLO-Verarbeitung und Dart-Erkennung
                if not full_processing:
                    self.update_video_display(display_frame)
                    continue
                
                # YOLO Prediction nur auf verarbeitetem Frame
                try:
                    results = self.predictor.predict(processed_frame)
                except Exception as pred_e:
                    print(f"YOLO Prediction Fehler: {pred_e}")
                    self.update_video_display(processed_frame)
                    continue
                
                # Extract dart positions and dartboard points
                dart_positions = []
                dartboard_points = []
                
                for result in results:
                    if result.boxes is not None:
                        for box in result.boxes:
                            try:
                                x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                                y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                                class_id = int(box.cls[0])
                                
                                if class_id == 4:  # Dart class
                                    dart_positions.append((x_center, y_center))
                                else:
                                    dartboard_points.append((x_center, y_center))
                            except Exception as box_e:
                                print(f"Box-Verarbeitung Fehler: {box_e}")
                                continue
                
                # Calibrate score predictor if needed
                if (len(dartboard_points) >= 3 and 
                    self.score_predictor and 
                    not self.score_predictor.is_calibrated()):
                    try:
                        if self.score_predictor.calibrate_dartboard(dartboard_points):
                            self.root.after(0, lambda: self.update_status("Dartboard-Punkte-System kalibriert!"))
                    except Exception as score_cal_e:
                        print(f"Score-Predictor Kalibrierung Fehler: {score_cal_e}")
                
                # Process dart detections und Score-Anzeige hinzufügen
                if self.score_predictor and self.score_predictor.is_calibrated() and dart_positions:
                    try:
                        # Filtere und stabilisiere Dart-Positionen
                        filtered_positions = self.filter_duplicate_darts(dart_positions)
                        
                        # Reset board_empty_check_frames da Darts erkannt wurden
                        self.board_empty_check_frames = 0
                        
                        # IMMER Score-Berechnung und Anzeige für aktuelle Positionen
                        display_frame, dart_scores = self.score_predictor.process_dart_detections(
                            display_frame, 
                            filtered_positions, 
                            show_scores=True
                        )
                        
                        # Speichere Positionen für konsistente Anzeige zwischen Frames
                        self.last_dart_positions = filtered_positions.copy()
                        
                        # Prüfe ob Detection verarbeitet werden soll (Anti-Duplikat-Logik)
                        should_process = self.should_process_dart_detection(filtered_positions)
                        
                        # Process detected darts automatically - nur bei neuen Erkennungen
                        if should_process and dart_scores:
                            print(f"✓ should_process=True, {len(dart_scores)} Dart-Scores: {dart_scores}")
                            print(f"  game_active: {self.game_state.game_active}")
                            print(f"  current_player: {self.game_state.get_current_player().name if self.game_state.get_current_player() else 'None'}")
                            print(f"  current_dart_count: {self.game_state.current_dart_count}")
                            print(f"  turn_complete_cooldown: {self.turn_complete_cooldown}")
                            
                            # Setze längeren Cooldown um Spam zu verhindern
                            self.dart_detection_cooldown = 60  # 2 Sekunden bei 30 FPS
                            
                            self.last_dart_scores = dart_scores.copy() if dart_scores else []
                            self.root.after(0, lambda scores=dart_scores: self.process_detected_darts(scores))
                            
                        else:
                            print(f"✗ should_process={should_process}, dart_scores={len(dart_scores) if dart_scores else 0}, game_active={self.game_state.game_active}")
                            if self.dart_detection_cooldown > 0:
                                print(f"  Dart-Detection-Cooldown aktiv: {self.dart_detection_cooldown}")
                            if self.turn_complete_cooldown > 0:
                                print(f"  Turn-Complete-Cooldown aktiv: {self.turn_complete_cooldown}")
                            if not dart_scores:
                                print(f"  Keine Dart-Scores")
                            if not self.game_state.game_active:
                                print(f"  Spiel nicht aktiv")
                        
                    except Exception as detection_e:
                        print(f"Dart-Detection Fehler: {detection_e}")
                elif self.score_predictor and self.score_predictor.is_calibrated() and not dart_positions:
                    # Keine Darts erkannt - zähle Frames für "leeres Board"
                    self.board_empty_check_frames += 1
                    
                    # Nach ausreichend Frames ohne Darts UND aktiver Turn-Complete-Cooldown: Board als leer betrachten
                    if (self.board_empty_check_frames >= self.board_empty_required_frames and 
                        self.turn_complete_cooldown > 0):
                        print(f"🔄 Board ist leer nach {self.board_empty_check_frames} Frames! Reset Turn-Complete-Cooldown (war {self.turn_complete_cooldown})")
                        self.turn_complete_cooldown = 0
                        self.board_empty_check_frames = 0
                        self.reset_dart_detection_state()
                        
                        # Prüfe ob Runde bereit zum Abschluss ist und schließe sie ab
                        if self.turn_ready_to_complete:
                            print("🔄 Board ist leer und Runde bereit - schließe Runde ab")
                            self.turn_ready_to_complete = False
                            self.root.after(0, self.complete_current_turn)
                        
                        # Update Turn-Display sofort
                        self.root.after(0, self.update_turn_display)
                    
                    # Lösche Dart-Cache nach einer Weile wenn kein Cooldown aktiv
                    if self.dart_detection_cooldown <= 0 and self.turn_complete_cooldown <= 0:
                        self.reset_dart_detection_state()
                
                # Show YOLO detections in debug mode
                if self.debug_mode and results:
                    try:
                        for result in results:
                            annotated = result.plot()
                            # Could show this in a separate window if needed
                    except Exception as debug_e:
                        print(f"Debug-Anzeige Fehler: {debug_e}")
                
                self.update_video_display(display_frame)
                
            except Exception as e:
                print(f"Fehler in camera_loop: {e}")
                # Bei größeren Fehlern, verwende Original-Frame
                try:
                    if 'frame' in locals() and frame is not None:
                        self.update_video_display(frame)
                except:
                    pass
                time.sleep(0.1)
    
    def update_video_display(self, frame):
        """Video-Anzeige aktualisieren."""
        if frame is None:
            return
            
        try:
            # Check if frame is valid
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"Ungültiger Frame: {frame.shape}")
                return
                
            # IMMER Frame auf maximale Display-Größe skalieren
            height, width = frame.shape[:2]
            max_width, max_height = 540, 420  # Weitere Reduzierung für bessere GUI-Sichtbarkeit
            
            # Berechne Skalierungsfaktor um Seitenverhältnis beizubehalten
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Frame IMMER skalieren (auch wenn er kleiner ist, für Konsistenz)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert to RGB and create PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Update label in main thread
            self.root.after(0, self._update_video_label, photo)
            
        except Exception as e:
            print(f"Fehler bei Video-Anzeige: {e}")
            # Try to show error message in video display
            try:
                self.root.after(0, lambda: self.video_label.configure(
                    image='', 
                    text=f"Video-Fehler:\n{str(e)[:50]}...",
                    fg='red'
                ))
            except:
                pass
    
    def _update_video_label(self, photo):
        """Video-Label in Haupt-Thread aktualisieren."""
        try:
            if self.video_label and photo:
                self.video_label.configure(image=photo, text="", fg='white')
                self.video_label.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            print(f"Fehler beim Video-Label Update: {e}")
            try:
                if self.video_label:
                    self.video_label.configure(
                        image='', 
                        text="Video-Display Fehler", 
                        fg='red'
                    )
            except:
                pass
    
    def process_detected_darts(self, dart_scores: List[Tuple[int, str]]):
        """Erkannte Darts verarbeiten."""
        print(f"process_detected_darts: Eingegangen mit {len(dart_scores) if dart_scores else 0} Dart-Scores")
        print(f"process_detected_darts: game_active = {self.game_state.game_active}")
        
        if not dart_scores:
            print("process_detected_darts: Keine Dart-Scores")
            return
            
        # Prüfe ob wir gerade einen Dart verarbeiten (Anti-Spam)
        if hasattr(self, '_currently_processing_dart') and self._currently_processing_dart:
            print("process_detected_darts: Dart wird bereits verarbeitet, ignoriere")
            return
            
        # Setze Processing-Flag
        self._currently_processing_dart = True
        
        try:
            if not self.game_state.game_active:
                print("process_detected_darts: Spiel nicht aktiv - starte automatisch ein Spiel")
                # Automatisch ein Spiel starten wenn Spieler vorhanden sind
                if len(self.game_state.players) >= 2:
                    self.game_state.start_game()
                    self.update_status("Spiel automatisch gestartet wegen Dart-Erkennung")
                    self.update_all_displays()
                else:
                    print("process_detected_darts: Nicht genug Spieler für automatischen Start")
                    return
            
            # Only process if we're waiting for darts
            current_player = self.game_state.get_current_player()
            if not current_player:
                print("process_detected_darts: Kein aktueller Spieler")
                return
            
            # Prüfe ob wir noch Darts für diese Runde erwarten
            if self.game_state.current_dart_count >= 3:
                print(f"process_detected_darts: Runde bereits komplett ({self.game_state.current_dart_count}/3)")
                return
            
            # Nimm nur den ersten erkannten Dart UND entferne ähnliche aus processed_dart_positions
            score, description = dart_scores[0]
            print(f"Verarbeite EINEN Dart: {score} ({description}) für {current_player.name} (Dart {self.game_state.current_dart_count + 1}/3)")
            
            # Finde die entsprechende neue Position und markiere sie als verarbeitet
            # Suche die erste Position die NICHT bereits verarbeitet wurde
            new_position_found = None
            for i, (dart_score, dart_desc) in enumerate(dart_scores):
                # Versuche die entsprechende Position zu finden
                if i < len(self.last_dart_positions):
                    candidate_pos = self.last_dart_positions[i]
                    # Prüfe ob diese Position bereits verarbeitet wurde
                    is_already_processed = any(
                        self.are_positions_similar(candidate_pos, processed_pos, self.dart_position_tolerance)
                        for processed_pos in self.processed_dart_positions
                    )
                    if not is_already_processed:
                        new_position_found = candidate_pos
                        # Verwende den entsprechenden Score, nicht nur den ersten
                        score, description = dart_score, dart_desc
                        print(f"Verwende Position {new_position_found} mit Score {score} ({description})")
                        break
            
            # Füge die neue Position zu verarbeiteten hinzu
            if new_position_found:
                self.processed_dart_positions.append(new_position_found)
                print(f"Position {new_position_found} zu processed_dart_positions hinzugefügt")
            
            turn_complete = self.game_state.add_dart_score(score, description)
            
            # Setze längeren Cooldown nach Score-Hinzufügung (wird in camera_loop auf 60 gesetzt)
            print(f"Score hinzugefügt, Cooldown auf {self.dart_detection_cooldown}")
            
            if turn_complete:
                print("process_detected_darts: Runde komplett, starte Board-Clearing-Pause")
                # AKTIVIERE 10-Sekunden-Cooldown nach kompletter Runde
                self.turn_complete_cooldown = self.turn_complete_cooldown_duration
                self.board_empty_check_frames = 0
                print(f"Turn-Complete-Cooldown aktiviert: {self.turn_complete_cooldown} Frames (10 Sekunden)")
                # Setze Flag dass Runde bereit zum Abschluss ist, aber warte auf Cooldown-Ende
                self.turn_ready_to_complete = True
            else:
                print(f"process_detected_darts: Runde noch nicht komplett, warte auf weitere Darts")
            
            # Update displays
            self.root.after(0, self.update_all_displays)
            
            # Lösche Detection-Cache nach kurzer Verzögerung
            self.root.after(100, self.clear_dart_cache_after_processing)
            
        finally:
            # Processing-Flag nach kurzer Verzögerung zurücksetzen
            self.root.after(200, self.reset_processing_flag)
    
    def reset_processing_flag(self):
        """Processing-Flag zurücksetzen."""
        self._currently_processing_dart = False
        print("Processing-Flag zurückgesetzt")
    
    def complete_current_turn(self):
        """Aktuelle Runde abschließen - wird nur aufgerufen wenn Cooldown abgelaufen ist."""
        print("complete_current_turn: Schließe Runde ab (nach Cooldown)")
        
        # Processing-Flag zurücksetzen
        self._currently_processing_dart = False
        
        if self.game_state.winner:
            print("complete_current_turn: Spiel beendet, es gibt einen Gewinner")
            self.end_game()
        else:
            self.game_state.complete_turn()
            # Nach Rundenwechsel alle Dart-Positionen löschen
            print(f"Rundenwechsel: Lösche {len(self.processed_dart_positions)} verarbeitete Dart-Positionen")
            print(f"Rundenwechsel: Lösche {len(self.blacklisted_dart_positions)} blacklisted Dart-Positionen")
            self.processed_dart_positions = []
            self.blacklisted_dart_positions = []  # Blacklist bei Rundenwechsel löschen
            self.reset_dart_detection_state()  # Kompletter Reset nach Rundenwechsel
            self.update_all_displays()
    
    def add_player(self):
        """Neuen Spieler hinzufügen."""
        name = simpledialog.askstring("Spieler hinzufügen", "Name des Spielers:")
        if name and name.strip():
            player = self.game_state.add_player(name.strip())
            self.create_player_widget(player)
            self.update_status(f"Spieler {name} hinzugefügt")
            self.update_turn_display()
    
    def remove_selected_player(self):
        """Ausgewählten Spieler entfernen."""
        if not self.game_state.players:
            messagebox.showwarning("Warnung", "Keine Spieler vorhanden")
            return
        
        # Show dialog to select player
        names = [p.name for p in self.game_state.players]
        selected = simpledialog.askstring("Spieler entfernen", f"Verfügbare Spieler: {', '.join(names)}\nName eingeben:")
        
        if selected:
            for player in self.game_state.players:
                if player.name == selected:
                    self.game_state.remove_player(player)
                    self.refresh_players_display()
                    self.update_status(f"Spieler {selected} entfernt")
                    self.update_turn_display()
                    return
            messagebox.showwarning("Warnung", "Spieler nicht gefunden")
    
    def create_player_widget(self, player: Player):
        """Player Widget erstellen."""
        # Remove existing widget if any
        if player.id in self.player_frames:
            self.player_frames[player.id].destroy()
        
        # Create new widget
        player_frame = tk.Frame(self.scrollable_frame, bg='#2C3E50', relief='raised', bd=2)
        player_frame.pack(fill='x', padx=5, pady=5)
        
        # Player header
        header_frame = tk.Frame(player_frame, bg='#2C3E50')
        header_frame.pack(fill='x', padx=10, pady=5)
        
        # Player name and current status
        name_label = tk.Label(header_frame, text=player.name, 
                             font=('Arial', 16, 'bold'), 
                             fg='white', bg='#2C3E50')
        name_label.pack(side='left')
        
        # Current score
        score_label = tk.Label(header_frame, text=f"Punkte: {player.current_score}", 
                              font=('Arial', 14, 'bold'), 
                              fg='#E74C3C', bg='#2C3E50')
        score_label.pack(side='right')
        
        # Stats frame
        stats_frame = tk.Frame(player_frame, bg='#2C3E50')
        stats_frame.pack(fill='x', padx=10, pady=5)
        
        # Statistics
        darts_label = tk.Label(stats_frame, text=f"Darts: {player.darts_thrown}", 
                              font=('Arial', 10), fg='#BDC3C7', bg='#2C3E50')
        darts_label.pack(side='left')
        
        avg_label = tk.Label(stats_frame, text=f"Durchschnitt: {player.get_average():.1f}", 
                            font=('Arial', 10), fg='#BDC3C7', bg='#2C3E50')
        avg_label.pack(side='right')
        
        # Current turn scores
        if (self.game_state.get_current_player() == player and 
            self.game_state.current_turn_scores):
            turn_frame = tk.Frame(player_frame, bg='#2C3E50')
            turn_frame.pack(fill='x', padx=10, pady=5)
            
            turn_label = tk.Label(turn_frame, 
                                 text=f"Aktuelle Runde: {' + '.join(map(str, self.game_state.current_turn_scores))}", 
                                 font=('Arial', 12), fg='#F39C12', bg='#2C3E50')
            turn_label.pack()
        
        # Highlight current player
        if self.game_state.get_current_player() == player:
            player_frame.configure(bg='#16A085')
            header_frame.configure(bg='#16A085')
            stats_frame.configure(bg='#16A085')
            if 'turn_frame' in locals():
                turn_frame.configure(bg='#16A085')
        
        self.player_frames[player.id] = player_frame
    
    def refresh_players_display(self):
        """Spieler-Anzeige aktualisieren."""
        # Clear all player widgets
        for widget in self.player_frames.values():
            widget.destroy()
        self.player_frames.clear()
        
        # Recreate all player widgets
        for player in self.game_state.players:
            self.create_player_widget(player)
    
    def update_all_displays(self):
        """Alle Anzeigen aktualisieren."""
        self.refresh_players_display()
        self.update_turn_display()
    
    def update_turn_display(self):
        """Runden-Anzeige aktualisieren."""
        if self.turn_complete_cooldown > 0:
            # Berechne verbleibende Sekunden
            remaining_seconds = self.turn_complete_cooldown / 30  # 30 FPS
            if remaining_seconds > 1.0:
                self.turn_info_var.set(f"⏳ Wartepause - Board leeren! ({remaining_seconds:.1f}s)")
            else:
                self.turn_info_var.set(f"⏳ Wartepause - Board leeren! (0.{int(remaining_seconds*10)}s)")
            
            # Deaktiviere relevante Buttons während Cooldown
            if hasattr(self, 'next_turn_btn'):
                self.next_turn_btn.configure(state='disabled')
                
        elif self.game_state.players and self.game_state.game_active:
            current = self.game_state.get_current_player()
            dart_info = f"Dart {self.game_state.current_dart_count + 1}/3" if current else ""
            self.turn_info_var.set(f"{current.name} ist dran - {dart_info}")
            
            # Reaktiviere Buttons wenn kein Cooldown
            if hasattr(self, 'next_turn_btn'):
                self.next_turn_btn.configure(state='normal')
                
        elif self.game_state.players:
            self.turn_info_var.set(f"{len(self.game_state.players)} Spieler bereit - Spiel starten")
            if hasattr(self, 'next_turn_btn'):
                self.next_turn_btn.configure(state='normal')
        else:
            self.turn_info_var.set("Spieler hinzufügen zum Starten")
            if hasattr(self, 'next_turn_btn'):
                self.next_turn_btn.configure(state='normal')
    
    def new_game(self):
        """Neues Spiel starten."""
        if len(self.game_state.players) < 2:
            messagebox.showwarning("Warnung", "Mindestens 2 Spieler erforderlich")
            return
        
        result = messagebox.askyesno("Neues Spiel", "Neues Spiel starten?")
        if result:
            # Reset aller Flags bei neuem Spiel
            self.turn_ready_to_complete = False
            self.turn_complete_cooldown = 0
            self.board_empty_check_frames = 0
            self.blacklisted_dart_positions = []  # Blacklist bei neuem Spiel löschen
            self.reset_dart_detection_state()
            
            self.game_state.start_game()
            self.update_status("Neues Spiel gestartet!")
            self.update_all_displays()
    
    def next_turn(self):
        """Zur nächsten Runde wechseln."""
        if not self.game_state.game_active:
            return
        
        # Prüfe Turn-Complete-Cooldown auch für manuellen Rundenwechsel
        if self.turn_complete_cooldown > 0:
            remaining_seconds = self.turn_complete_cooldown / 30
            messagebox.showwarning("Warnung", f"Board muss erst geleert werden! Noch {remaining_seconds:.1f} Sekunden warten.")
            return
        
        # Reset dart detection state bei Rundenwechsel
        self.reset_dart_detection_state()
        self.blacklisted_dart_positions = []  # Blacklist bei Rundenwechsel löschen
        self.turn_ready_to_complete = False  # Reset Flag bei manuellem Rundenwechsel
        print("Nächste Runde - Dart-Detection-State zurückgesetzt")
        print(f"Nächste Runde - Blacklist geleert")
        
        self.game_state.complete_turn()
        if self.game_state.winner:
            self.end_game()
        else:
            self.update_all_displays()
    
    def undo_last_dart(self):
        """Letzten Dart rückgängig machen."""
        if not self.game_state.game_active:
            messagebox.showwarning("Warnung", "Kein aktives Spiel!")
            return
            
        # Rückgängig ist IMMER erlaubt, auch während Board-Clearing-Pause
        # Das ist der ganze Sinn der Pause - Zeit für Korrekturen zu haben
            
        if self.game_state.current_dart_count > 0:
            # Merke ob wir vom dritten Dart zurückgehen (für spezielle Behandlung)
            was_third_dart = (self.game_state.current_dart_count == 3)
            
            # Reset entsprechende Detection-States wenn ein Dart rückgängig gemacht wird
            removed_position = None
            if self.processed_dart_positions:
                removed_position = self.processed_dart_positions.pop()
                print(f"Rückgängig: Entferne Position {removed_position} aus processed_dart_positions")
                
                # WICHTIG: Füge die Position zur Blacklist hinzu für den gesamten Zug
                self.blacklisted_dart_positions.append(removed_position)
                print(f"Rückgängig: Füge Position {removed_position} zur Blacklist hinzu")
                print(f"Blacklisted Positionen: {len(self.blacklisted_dart_positions)}")
            
            # KRITISCH: Entferne die entsprechende Position aus last_dart_positions und last_dart_scores
            # Finde die Position, die der entfernten processed_position entspricht
            if removed_position and self.last_dart_positions:
                # Suche die entsprechende Position in last_dart_positions
                for i, pos in enumerate(self.last_dart_positions):
                    if self.are_positions_similar(pos, removed_position, self.dart_position_tolerance):
                        removed_cached_position = self.last_dart_positions.pop(i)
                        print(f"Rückgängig: Entferne auch Position {removed_cached_position} aus last_dart_positions (Index {i})")
                        if i < len(self.last_dart_scores):
                            removed_cached_score = self.last_dart_scores.pop(i)
                            print(f"Rückgängig: Entferne auch Score {removed_cached_score} aus last_dart_scores (Index {i})")
                        break
                else:
                    print(f"Rückgängig: Konnte Position {removed_position} nicht in last_dart_positions finden")
            
            # WICHTIG: Wenn wir vom dritten Dart zurückgehen, deaktiviere Turn-Complete-Cooldown
            if was_third_dart:
                print("Rückgängig vom dritten Dart: Deaktiviere Turn-Complete-Cooldown und Flags")
                self.turn_complete_cooldown = 0
                self.turn_ready_to_complete = False
                self.board_empty_check_frames = 0
                print(f"Turn-Complete-Cooldown deaktiviert, Zug ist wieder unvollständig")
            
            # Reset Turn-Ready-Flag generell bei Rückgängig
            if self.turn_ready_to_complete:
                print("Rückgängig: Reset turn_ready_to_complete")
                self.turn_ready_to_complete = False
            
            self.game_state.undo_last_dart()
            self.update_all_displays()
            self.update_status("Letzter Dart rückgängig gemacht")
        else:
            messagebox.showinfo("Information", "Kein Dart zum Rückgängigmachen vorhanden")
    
    def add_manual_score(self):
        """Manuelle Punktzahl hinzufügen."""
        try:
            score_text = self.manual_score_var.get().strip()
            if not score_text:
                return
            
            score = int(score_text)
            if score < 0 or score > 180:
                messagebox.showerror("Fehler", "Punktzahl muss zwischen 0 und 180 liegen")
                return
            
            # Prüfe ob Spiel aktiv ist und ob wir einen aktuellen Spieler haben
            if not self.game_state.game_active:
                messagebox.showwarning("Warnung", "Bitte zuerst ein Spiel starten!")
                return
            
            current_player = self.game_state.get_current_player()
            if not current_player:
                messagebox.showwarning("Warnung", "Kein aktiver Spieler!")
                return
            
            # Prüfe Turn-Complete-Cooldown (gleiche Logik wie bei automatischer Erkennung)
            if self.turn_complete_cooldown > 0:
                remaining_seconds = self.turn_complete_cooldown / 30
                messagebox.showwarning("Warnung", f"Board muss erst geleert werden! Noch {remaining_seconds:.1f} Sekunden warten.")
                return
            
            # Debug-Ausgabe
            print(f"Füge manuell {score} Punkte für {current_player.name} hinzu")
            
            turn_complete = self.game_state.add_dart_score(score, f"Manuell: {score}")
            
            # Gleiche Behandlung wie bei automatischer Erkennung
            if turn_complete:
                print("add_manual_score: Runde komplett, starte Board-Clearing-Pause")
                # AKTIVIERE 10-Sekunden-Cooldown nach kompletter Runde (gleich wie bei Auto-Erkennung)
                self.turn_complete_cooldown = self.turn_complete_cooldown_duration
                self.board_empty_check_frames = 0
                print(f"Turn-Complete-Cooldown aktiviert: {self.turn_complete_cooldown} Frames (10 Sekunden)")
                # Setze Flag dass Runde bereit zum Abschluss ist, aber warte auf Cooldown-Ende
                self.turn_ready_to_complete = True
            else:
                print(f"add_manual_score: Runde noch nicht komplett, warte auf weitere Darts")
            
            self.manual_score_var.set("")
            self.update_all_displays()
            self.update_status(f"Punkte {score} für {current_player.name} hinzugefügt")
            
        except ValueError:
            messagebox.showerror("Fehler", "Bitte gültige Zahl eingeben")
    
    def end_game(self):
        """Spiel beenden."""
        winner = self.game_state.winner
        if winner:
            messagebox.showinfo("Spiel beendet!", f"🎉 {winner.name} hat gewonnen! 🎉")
            self.game_state.game_active = False
            self.update_status(f"Spiel beendet - Gewinner: {winner.name}")
            self.update_all_displays()
    
    def calibrate_dartboard(self):
        """Dartboard manuell kalibrieren."""
        if not self.camera_running:
            messagebox.showwarning("Warnung", "Bitte zuerst Kamera starten")
            return
        
        # Get current frame for calibration
        if not self.camera:
            messagebox.showerror("Fehler", "Kamera nicht verfügbar")
            return
        
        try:
            # Hole aktuellen Frame
            current_frame = self.camera.get_frame_raw()
            if current_frame is None:
                messagebox.showerror("Fehler", "Kann keinen Frame von der Kamera abrufen")
                return
            
            # Reset calibration status
            self.dartboard_calibrated = False
            self.dartboard_status_var.set("Dartboard: Kalibrierung läuft...")
            self.update_status("Starte manuelle Dartboard-Kalibrierung...")
            
            # Versuche Kalibrierung mit aktuellem Frame
            success, result = self.calibration.initial_calibration(current_frame)
            
            if success:
                self.dartboard_calibrated = True
                self.dartboard_status_var.set("Dartboard: Kalibriert ✓")
                self.update_status("Dartboard erfolgreich kalibriert!")
                
                # Reset score predictor für neue Kalibrierung
                self.score_predictor = score_prediction.DartboardScorePredictor()
                
                messagebox.showinfo(
                    "Erfolg", 
                    "Dartboard wurde erfolgreich kalibriert!\n\n"
                    "Alle folgenden Frames werden nun automatisch transformiert."
                )
            else:
                self.dartboard_calibrated = False
                self.dartboard_status_var.set("Dartboard: Kalibrierung fehlgeschlagen")
                self.update_status("Dartboard-Kalibrierung fehlgeschlagen")
                
                messagebox.showwarning(
                    "Kalibrierung fehlgeschlagen", 
                    "Die Dartboard konnte nicht kalibriert werden.\n\n"
                    "Stellen Sie sicher, dass:\n"
                    "• Die Dartboard vollständig im Bild sichtbar ist\n"
                    "• Das Referenzbild verfügbar ist\n"
                    "• Genügend markante Punkte erkannt werden\n\n"
                    "Versuchen Sie es erneut mit besserer Positionierung."
                )
                
        except Exception as e:
            self.dartboard_calibrated = False
            self.dartboard_status_var.set("Dartboard: Kalibrierungsfehler")
            error_msg = f"Fehler bei der Kalibrierung: {e}"
            self.update_status(error_msg)
            messagebox.showerror("Kalibrierungsfehler", error_msg)
    
    def reset_calibration(self):
        """Kalibrierung zurücksetzen."""
        self.dartboard_calibrated = False
        self.dartboard_status_var.set("Dartboard: Nicht kalibriert")
        
        # Reset calibration matrix
        if self.calibration:
            self.calibration.H = None
        
        # Reset score predictor
        self.score_predictor = score_prediction.DartboardScorePredictor()
        
        self.update_status("Kalibrierung zurückgesetzt")
        messagebox.showinfo("Kalibrierung zurückgesetzt", "Die Dartboard-Kalibrierung wurde zurückgesetzt.")
    
    def update_status(self, message: str):
        """Status-Nachricht aktualisieren."""
        self.status_var.set(message)
        print(f"Status: {message}")
    
    def save_game(self):
        """Spiel speichern."""
        if not self.game_state.players:
            messagebox.showwarning("Warnung", "Keine Spieler vorhanden")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Spiel speichern"
        )
        
        if filename:
            try:
                self.game_state.save_game(filename)
                self.update_status(f"Spiel gespeichert: {filename}")
                messagebox.showinfo("Erfolg", "Spiel erfolgreich gespeichert!")
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Speichern: {e}")
    
    def load_game(self):
        """Spiel laden."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Spiel laden"
        )
        
        if filename:
            try:
                self.game_state.load_game(filename)
                self.refresh_players_display()
                self.update_turn_display()
                self.update_status(f"Spiel geladen: {filename}")
                messagebox.showinfo("Erfolg", "Spiel erfolgreich geladen!")
            except Exception as e:
                messagebox.showerror("Fehler", f"Fehler beim Laden: {e}")
    
    def camera_settings(self):
        """Kamera-Einstellungen Dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Kamera-Einstellungen")
        dialog.geometry("400x300")
        dialog.configure(bg='#34495E')
        
        # Camera source selection
        tk.Label(dialog, text="Kamera-Quelle:", font=('Arial', 12, 'bold'),
                fg='white', bg='#34495E').pack(pady=10)
        
        source_var = tk.StringVar(value="webcam" if not self.use_image_folder else "folder")
        
        webcam_radio = tk.Radiobutton(dialog, text="Webcam", variable=source_var, 
                                     value="webcam", font=('Arial', 11),
                                     fg='white', bg='#34495E', selectcolor='#2C3E50')
        webcam_radio.pack()
        
        folder_radio = tk.Radiobutton(dialog, text="Bildordner", variable=source_var, 
                                     value="folder", font=('Arial', 11),
                                     fg='white', bg='#34495E', selectcolor='#2C3E50')
        folder_radio.pack()
        
        # Webcam ID
        tk.Label(dialog, text="Webcam ID:", font=('Arial', 11),
                fg='white', bg='#34495E').pack(pady=(20, 5))
        
        webcam_var = tk.StringVar(value=str(self.camera_source))
        webcam_entry = tk.Entry(dialog, textvariable=webcam_var, font=('Arial', 11))
        webcam_entry.pack()
        
        # Image folder path
        tk.Label(dialog, text="Bildordner Pfad:", font=('Arial', 11),
                fg='white', bg='#34495E').pack(pady=(20, 5))
        
        folder_var = tk.StringVar(value=self.image_folder_path)
        folder_entry = tk.Entry(dialog, textvariable=folder_var, font=('Arial', 11), width=40)
        folder_entry.pack()
        
        browse_btn = tk.Button(dialog, text="Durchsuchen", 
                              command=lambda: self.browse_folder(folder_var),
                              bg='#3498DB', fg='white')
        browse_btn.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#34495E')
        button_frame.pack(pady=20)
        
        def apply_settings():
            self.use_image_folder = (source_var.get() == "folder")
            self.camera_source = int(webcam_var.get()) if webcam_var.get().isdigit() else 0
            self.image_folder_path = folder_var.get()
            dialog.destroy()
            self.update_status("Kamera-Einstellungen aktualisiert")
        
        apply_btn = tk.Button(button_frame, text="Anwenden", command=apply_settings,
                             bg='#27AE60', fg='white', font=('Arial', 11))
        apply_btn.pack(side='left', padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Abbrechen", command=dialog.destroy,
                              bg='#E74C3C', fg='white', font=('Arial', 11))
        cancel_btn.pack(side='left', padx=5)
    
    def browse_folder(self, folder_var):
        """Ordner durchsuchen."""
        folder = filedialog.askdirectory(title="Bildordner auswählen")
        if folder:
            folder_var.set(folder)
    
    def game_settings(self):
        """Spiel-Einstellungen Dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Spiel-Einstellungen")
        dialog.geometry("350x250")
        dialog.configure(bg='#34495E')
        
        # Game mode
        tk.Label(dialog, text="Spielmodus:", font=('Arial', 12, 'bold'),
                fg='white', bg='#34495E').pack(pady=10)
        
        mode_var = tk.StringVar(value=self.game_state.game_mode)
        mode_combo = ttk.Combobox(dialog, textvariable=mode_var, 
                                 values=["501", "301", "Cricket"], state="readonly")
        mode_combo.pack()
        
        # Auto advance
        auto_var = tk.BooleanVar(value=self.game_state.auto_advance)
        auto_check = tk.Checkbutton(dialog, text="Automatisch zur nächsten Runde", 
                                   variable=auto_var, font=('Arial', 11),
                                   fg='white', bg='#34495E', selectcolor='#2C3E50')
        auto_check.pack(pady=10)
        
        # Legs to win
        tk.Label(dialog, text="Legs zum Gewinnen:", font=('Arial', 11),
                fg='white', bg='#34495E').pack(pady=(10, 5))
        
        legs_var = tk.StringVar(value=str(self.game_state.legs_to_win))
        legs_entry = tk.Entry(dialog, textvariable=legs_var, font=('Arial', 11), width=10)
        legs_entry.pack()
        
        # Buttons
        button_frame = tk.Frame(dialog, bg='#34495E')
        button_frame.pack(pady=20)
        
        def apply_settings():
            self.game_state.game_mode = mode_var.get()
            self.game_state.auto_advance = auto_var.get()
            try:
                self.game_state.legs_to_win = int(legs_var.get())
            except ValueError:
                pass
            dialog.destroy()
            self.update_status("Spiel-Einstellungen aktualisiert")
        
        apply_btn = tk.Button(button_frame, text="Anwenden", command=apply_settings,
                             bg='#27AE60', fg='white', font=('Arial', 11))
        apply_btn.pack(side='left', padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Abbrechen", command=dialog.destroy,
                              bg='#E74C3C', fg='white', font=('Arial', 11))
        cancel_btn.pack(side='left', padx=5)
    
    def toggle_debug(self):
        """Debug-Modus umschalten."""
        self.debug_mode = not self.debug_mode
        status = "aktiviert" if self.debug_mode else "deaktiviert"
        self.update_status(f"Debug-Modus {status}")
        messagebox.showinfo("Debug-Modus", f"Debug-Modus {status}")
    
    def run(self):
        """GUI ausführen."""
        self.root.mainloop()
    
    def on_closing(self):
        """Beim Schließen der Anwendung."""
        self.stop_camera()
        self.root.destroy()
    
    def are_positions_similar(self, pos1: Tuple[int, int], pos2: Tuple[int, int], tolerance: int = 30) -> bool:
        """Prüft ob zwei Positionen ähnlich genug sind (gleicher Dart)."""
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance <= tolerance
    
    def filter_duplicate_darts(self, dart_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filtert doppelte/ähnliche Dart-Positionen heraus."""
        if not dart_positions:
            return []
        
        filtered_positions = []
        
        for pos in dart_positions:
            # ERST prüfen ob Position auf der Blacklist steht
            is_blacklisted = any(
                self.are_positions_similar(pos, blacklisted_pos, self.dart_position_tolerance)
                for blacklisted_pos in self.blacklisted_dart_positions
            )
            
            if is_blacklisted:
                print(f"Position {pos} ist blacklisted, ignoriere für gesamten Zug")
                continue
            
            # Prüfe ob diese Position zu ähnlich zu bereits gefilterten ist
            is_duplicate = False
            for existing_pos in filtered_positions:
                if self.are_positions_similar(pos, existing_pos, self.dart_position_tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_positions.append(pos)
        
        return filtered_positions
    
    def should_process_dart_detection(self, dart_positions: List[Tuple[int, int]]) -> bool:
        """Entscheidet ob Dart-Detection verarbeitet werden soll (Anti-Spam)."""
        print(f"should_process_dart_detection: Input {len(dart_positions)} Dart-Positionen")
        
        # Turn-Complete-Cooldown check (wichtiger als normaler Cooldown)
        if self.turn_complete_cooldown > 0:
            print(f"  Turn-Complete-Cooldown aktiv: {self.turn_complete_cooldown} Frames")
            return False
        
        # Normaler Cooldown check
        if self.dart_detection_cooldown > 0:
            print(f"  Cooldown aktiv: {self.dart_detection_cooldown}")
            return False
        
        # Keine Darts erkannt
        if not dart_positions:
            print(f"  Keine Dart-Positionen")
            return False
        
        # Einfache Duplikat-Prüfung: Prüfe nur gegen bereits verarbeitete Positionen
        new_positions = []
        for pos in dart_positions:
            is_already_processed = any(
                self.are_positions_similar(pos, processed_pos, self.dart_position_tolerance)
                for processed_pos in self.processed_dart_positions
            )
            if not is_already_processed:
                new_positions.append(pos)
        
        print(f"  Neue Positionen: {len(new_positions)} von {len(dart_positions)}")
        
        # Nur verarbeiten wenn wir NEUE Positionen haben
        if len(new_positions) > 0:
            print(f"  ✓ Neue Dart-Positionen gefunden: {len(new_positions)}")
            return True
        
        # Keine neuen Positionen
        print(f"  ✗ Keine neuen Positionen")
        return False
    
    def reset_dart_detection_state(self):
        """Setzt den Dart-Detection-Zustand zurück."""
        print(f"reset_dart_detection_state: Lösche {len(self.last_dart_positions)} last_dart_positions, {len(self.processed_dart_positions)} processed_dart_positions")
        self.last_dart_positions = []
        self.processed_dart_positions = []
        # WICHTIG: Blacklist wird NICHT hier gelöscht, nur bei Rundenwechsel!
        self.last_dart_scores = []
        self.dart_detection_cooldown = 0
    
    def clear_dart_cache_after_processing(self):
        """Löscht den Dart-Cache nach Verarbeitung, um doppelte Zählungen zu verhindern."""
        print("Lösche Dart-Cache nach Verarbeitung")
        print(f"  Vor dem Löschen: last_dart_positions={len(self.last_dart_positions)}, processed={len(self.processed_dart_positions)}")
        
        # Behalte last_dart_positions für die Anzeige, lösche nur wenn alle verarbeitet wurden
        all_positions_processed = True
        if self.last_dart_positions:
            for pos in self.last_dart_positions:
                is_processed = any(
                    self.are_positions_similar(pos, processed_pos, self.dart_position_tolerance)
                    for processed_pos in self.processed_dart_positions
                )
                if not is_processed:
                    all_positions_processed = False
                    break
        
        if all_positions_processed and self.last_dart_positions:
            print("  Alle Positionen verarbeitet - lösche Anzeige-Cache")
            self.last_dart_positions = []
            self.last_dart_scores = []
        else:
            print(f"  Noch unverarbeitete Positionen - behalte Anzeige-Cache")
        
        print(f"  Nach dem Löschen: last_dart_positions={len(self.last_dart_positions)}, processed_dart_positions={len(self.processed_dart_positions)}")

if __name__ == "__main__":
    # Start the Darts GUI application
    app = DartsGUI()
    app.run()
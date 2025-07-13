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

from utils import camera, calibration, predict, score_prediction


class Player:
    """Represents a single player in the game."""
    
    def __init__(self, name: str, player_id: int):
        self.name = name
        self.id = player_id
        self.current_score = 501  # Standard for 501 game
        self.darts_thrown = 0
        self.turn_scores = []  # List of points per round
        self.game_history = []  # History of all games
        self.legs_won = 0
        self.sets_won = 0
        
    def reset_game(self, starting_score: int = 501):
        """Reset player for a new game."""
        self.current_score = starting_score
        self.darts_thrown = 0
        self.turn_scores = []
    
    def add_turn_score(self, scores: List[int]):
        """Add points from a round (up to 3 darts)."""
        turn_total = sum(scores)
        self.turn_scores.append(scores)
        self.darts_thrown += len(scores)
        return turn_total
    
    def get_average(self) -> float:
        """Calculate average points per dart."""
        if self.darts_thrown == 0:
            return 0.0
        total_score = 501 - self.current_score
        return total_score / self.darts_thrown
    
    def can_finish_with_score(self, score: int) -> bool:
        """Check if the player can finish with this score."""
        return self.current_score == score and score <= 170
    
    def is_bust(self, score: int) -> bool:
        """Check if the score is a bust."""
        remaining = self.current_score - score
        return remaining < 0 or remaining == 1
    
    def to_dict(self) -> dict:
        """Convert player to dictionary for storage."""
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
        """Create player from dictionary."""
        player = cls(data['name'], data['id'])
        player.current_score = data.get('current_score', 501)
        player.darts_thrown = data.get('darts_thrown', 0)
        player.turn_scores = data.get('turn_scores', [])
        player.game_history = data.get('game_history', [])
        player.legs_won = data.get('legs_won', 0)
        player.sets_won = data.get('sets_won', 0)
        return player


class GameState:
    """Manages the entire game state."""
    
    def __init__(self):
        self.players: List[Player] = []
        self.current_player_index = 0
        self.current_dart_count = 0  # Thrown darts in current round (0-3)
        self.current_turn_scores = []  # Points for current round
        self.game_mode = "501" 
        self.game_active = False
        self.winner: Optional[Player] = None
        self.legs_to_win = 3
        self.sets_to_win = 2
        self.auto_advance = True  # Automatically advance to next round after 3 darts
        
    def add_player(self, name: str) -> Player:
        """Add new player to the game."""
        player_id = len(self.players) + 1
        player = Player(name, player_id)
        self.players.append(player)
        return player
    
    def remove_player(self, player: Player):
        """Remove player from the game."""
        if player in self.players:
            self.players.remove(player)
            
            if self.current_player_index >= len(self.players):
                self.current_player_index = 0
    
    def start_game(self, game_mode: str = "501"):
        """Start new game."""
        if len(self.players) < 2:
            raise ValueError("At least 2 players required")
        
        self.game_mode = game_mode
        self.current_player_index = 0
        self.current_dart_count = 0
        self.current_turn_scores = []
        self.game_active = True
        self.winner = None
        
        # Reset all players
        starting_score = 501 if game_mode == "501" else 0
        for player in self.players:
            player.reset_game(starting_score)
    
    def get_current_player(self) -> Optional[Player]:
        """Get currently active player."""
        if not self.players:
            return None
        return self.players[self.current_player_index]
    
    def add_dart_score(self, score: int, description: str = "") -> bool:
        """Add dart score for current player. Returns True if round is complete."""
        if not self.game_active or not self.players:
            print(f"add_dart_score: Game not active or no players")
            return False
        
        current_player = self.get_current_player()
        if not current_player:
            print(f"add_dart_score: No current player")
            return False
            
        print(f"add_dart_score: {score} for {current_player.name} (Dart {self.current_dart_count + 1}/3)")
        
        self.current_turn_scores.append(score)
        self.current_dart_count += 1
        
        # Check if round is complete
        turn_complete = False
        if self.current_dart_count >= 3:
            turn_complete = True
            print(f"add_dart_score: Round complete (3 darts)")
        elif self.game_mode == "501":
            # Check for bust or finish
            turn_total = sum(self.current_turn_scores)
            if current_player.is_bust(turn_total):
                turn_complete = True
                print(f"add_dart_score: Bust with {turn_total} points")
            elif current_player.current_score == turn_total:
                
                current_player.current_score = 0
                self.winner = current_player
                self.game_active = False
                turn_complete = True
                print(f"add_dart_score: {current_player.name} has won!")
        
        print(f"add_dart_score: turn_complete = {turn_complete}")
        return turn_complete
    
    def complete_turn(self):
        """Complete the current player's turn."""
        if not self.players:
            return
        
        current_player = self.get_current_player()
        turn_total = sum(self.current_turn_scores)
        
        print(f"complete_turn: {current_player.name}, {turn_total} points, before: {current_player.current_score}")
        
        if self.game_mode == "501":
            if not current_player.is_bust(turn_total):
                current_player.current_score -= turn_total
                print(f"complete_turn: After deduction: {current_player.current_score}")
            else:
                print(f"complete_turn: Bust! Points remain at {current_player.current_score}")
        
        # Save round
        current_player.add_turn_score(self.current_turn_scores.copy())
        
        # Switch to next player
        old_player = self.current_player_index
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        new_player = self.get_current_player()
        print(f"complete_turn: Switch from {current_player.name} to {new_player.name}")
        
        self.current_dart_count = 0
        self.current_turn_scores = []
    
    def undo_last_dart(self):
        """Undo last dart."""
        if self.current_dart_count > 0:
            self.current_turn_scores.pop()
            self.current_dart_count -= 1
    
    def save_game(self, filepath: str):
        """Save current game state to file."""
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
        """Load game state from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.players = [Player.from_dict(p) for p in data['players']]
        self.current_player_index = data.get('current_player_index', 0)
        self.game_mode = data.get('game_mode', '501')
        self.game_active = data.get('game_active', False)
        self.legs_to_win = data.get('legs_to_win', 3)
        self.sets_to_win = data.get('sets_to_win', 2)


class DartsGUI:
    """Main GUI application for the dart game."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🎯 SmartDart")
        self.root.geometry("1600x900")
        self.root.configure(bg='#2C3E50')
        
        # Game state
        self.game_state = GameState()
        self.debug_mode = True
        
        # Computer Vision components
        self.camera = None
        self.calibration = None
        self.predictor = None
        self.score_predictor = None
        self.camera_thread = None
        self.camera_running = False
        
        # GUI variables
        self.video_label = None
        self.player_frames = {}
        self.status_var = tk.StringVar(value="Welcome to Darts Tournament!")
        self.turn_info_var = tk.StringVar(value="Add players to start")
        self.dartboard_calibrated = False
        
        # Camera settings
        self.camera_source = 1
        self.use_image_folder = False
        self.image_folder_path = "training/data/transferlearning/stg1/raw"
        
        # Dart detection cache for consistent display
        self.last_dart_positions = []       # For display between frames
        self.processed_dart_positions = []  # For anti-duplicate logic
        self.blacklisted_dart_positions = []  # For entire turn ignored dart positions
        self.last_dart_scores = []
        
        # Dart detection stabilization
        self.dart_detection_cooldown = 0  # Frames until next detection
        self.stable_dart_positions = []  # Stable dart positions
        self.detection_confirmation_frames = 3  # Frames for confirmation
        self.dart_position_tolerance = 30  # Pixel-tolerance for same position

        self._currently_processing_dart = False  # Prevents simultaneous dart processing

        # Cooldown after 3 darts
        self.turn_complete_cooldown = 0  # Cooldown after complete turn
        self.turn_complete_cooldown_duration = 100
        self.board_empty_check_frames = 0  # Frames without dart detection
        self.board_empty_required_frames = 10  # 1 second without darts = board empty
        self.turn_ready_to_complete = False  # Flag that turn is ready to complete

        # Create GUI
        self.setup_gui()
        self.setup_computer_vision()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_gui(self):
        """Set up main GUI layout."""
        self.create_menu()
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
    def create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # game menu
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self.new_game)
        game_menu.add_command(label="Save Game", command=self.save_game)
        game_menu.add_command(label="Load Game", command=self.load_game)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.quit)
        
        # player menu
        players_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Player", menu=players_menu)
        players_menu.add_command(label="Add Player", command=self.add_player)
        players_menu.add_command(label="Remove Player", command=self.remove_selected_player)

        # settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Camera Settings", command=self.camera_settings)
        settings_menu.add_command(label="Game Settings", command=self.game_settings)
        settings_menu.add_separator()
        settings_menu.add_command(label="Calibrate Dartboard", command=self.calibrate_dartboard)
        settings_menu.add_command(label="Reset Calibration", command=self.reset_calibration)
        settings_menu.add_separator()
        settings_menu.add_command(label="Toggle Debug", command=self.toggle_debug)

    def create_header(self):
        """Create header with game info."""
        header_frame = tk.Frame(self.root, bg='#34495E', height=70)
        header_frame.pack(fill='x', padx=5, pady=5)
        header_frame.pack_propagate(False)

        # Game Title
        title_label = tk.Label(header_frame, text="🎯 SmartDart", 
                              font=('Arial', 24, 'bold'), 
                              fg='white', bg='#34495E')
        title_label.pack(side='left', padx=20, pady=20)

        # Turn Info
        turn_frame = tk.Frame(header_frame, bg='#34495E')
        turn_frame.pack(side='right', padx=20, pady=20)
        
        turn_label = tk.Label(turn_frame, textvariable=self.turn_info_var,
                             font=('Arial', 16, 'bold'),
                             fg='#E74C3C', bg='#34495E')
        turn_label.pack()
    
    def create_main_content(self):
        """Create main content area."""
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # Camera and Controls
        left_panel = tk.Frame(main_frame, bg='#34495E', width=800)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.create_camera_panel(left_panel)
        self.create_control_panel(left_panel)

        # Players and Scores
        right_panel = tk.Frame(main_frame, bg='#34495E', width=700)
        right_panel.pack(side='right', fill='both', padx=(5, 0))
        right_panel.pack_propagate(False)
        
        self.create_players_panel(right_panel)
    
    def create_camera_panel(self, parent):
        """Create camera feed panel."""
        camera_frame = tk.LabelFrame(parent, text="Camera Feed", 
                                   font=('Arial', 14, 'bold'),
                                   fg='white', bg='#34495E', height=450)
        camera_frame.pack(fill='x', expand=False, padx=10, pady=10)
        camera_frame.pack_propagate(False)

        # Camera Display
        self.video_label = tk.Label(camera_frame, bg='black',
                                   text="Camera not initialized\nClick 'Start Camera'",
                                   fg='white', font=('Arial', 16))
        self.video_label.pack(padx=5, pady=5)
    
    def create_control_panel(self, parent):
        """Create control panel."""
        control_frame = tk.LabelFrame(parent, text="Control Panel",
                                    font=('Arial', 14, 'bold'),
                                    fg='white', bg='#34495E', height=140)
        control_frame.pack(fill='x', padx=10, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        # Button Frame
        button_frame = tk.Frame(control_frame, bg='#34495E')
        button_frame.pack(expand=True, fill='both', padx=10, pady=10)

        # Camera Controls
        camera_btn_frame = tk.Frame(button_frame, bg='#34495E')
        camera_btn_frame.pack(side='left', fill='y')

        self.camera_btn = tk.Button(camera_btn_frame, text="Start Camera", 
                                   font=('Arial', 11, 'bold'),
                                   bg='#27AE60', fg='white',
                                   command=self.toggle_camera, width=14)
        self.camera_btn.pack(pady=2)

        calibrate_btn = tk.Button(camera_btn_frame, text="Calibrate", 
                                font=('Arial', 11, 'bold'),
                                bg='#3498DB', fg='white',
                                command=self.calibrate_dartboard, width=14)
        calibrate_btn.pack(pady=2)

        reset_cal_btn = tk.Button(camera_btn_frame, text="Calibration Reset", 
                                font=('Arial', 9, 'bold'),
                                bg='#E67E22', fg='white',
                                command=self.reset_calibration, width=14)
        reset_cal_btn.pack(pady=2)

        # Game Controls
        game_btn_frame = tk.Frame(button_frame, bg='#34495E')
        game_btn_frame.pack(side='left', fill='y', padx=(20, 0))

        new_game_btn = tk.Button(game_btn_frame, text="New Game", 
                               font=('Arial', 11, 'bold'),
                               bg='#E67E22', fg='white',
                               command=self.new_game, width=14)
        new_game_btn.pack(pady=2)

        self.next_turn_btn = tk.Button(game_btn_frame, text="Next Turn", 
                                     font=('Arial', 11, 'bold'),
                                     bg='#9B59B6', fg='white',
                                     command=self.next_turn, width=14)
        self.next_turn_btn.pack(pady=2)

        # Manual Score Entry
        manual_frame = tk.Frame(button_frame, bg='#34495E')
        manual_frame.pack(side='right', fill='y')

        tk.Label(manual_frame, text="Manual Points:", 
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
        
        undo_btn = tk.Button(manual_frame, text="↶ Undo", 
                           font=('Arial', 10, 'bold'),
                           bg='#E74C3C', fg='white',
                           command=self.undo_last_dart, width=12)
        undo_btn.pack(pady=(5, 0))
    
    def create_players_panel(self, parent):
        """Create players panel."""
        players_frame = tk.LabelFrame(parent, text="Players & Scores", 
                                    font=('Arial', 14, 'bold'),
                                    fg='white', bg='#34495E')
        players_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Scrollable area for players
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

        # Add Player Button
        add_player_frame = tk.Frame(self.scrollable_frame, bg='#34495E')
        add_player_frame.pack(fill='x', pady=5)

        add_player_btn = tk.Button(add_player_frame, text="➕ Add Player", 
                                 font=('Arial', 12, 'bold'),
                                 bg='#3498DB', fg='white',
                                 command=self.add_player)
        add_player_btn.pack(pady=5)
        
    def create_status_bar(self):
        """Create status bar."""
        status_frame = tk.Frame(self.root, bg='#34495E', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                               font=('Arial', 10), fg='white', bg='#34495E')
        status_label.pack(side='left', padx=10, pady=5)
        
        # Dartboard Status
        self.dartboard_status_var = tk.StringVar(value="Dartboard not calibrated")
        dartboard_status = tk.Label(status_frame, textvariable=self.dartboard_status_var,
                                   font=('Arial', 10), fg='orange', bg='#34495E')
        dartboard_status.pack(side='right', padx=10, pady=5)
    
    def setup_computer_vision(self):
        """Set up computer vision components."""
        try:
            # Initialize components but don't start camera yet
            self.predictor = predict.Predictor(model_path="models/yolo8n-pretrained-al2-stg3.pt")
            self.score_predictor = score_prediction.DartboardScorePredictor()
            self.update_status("Computer vision components loaded")
        except Exception as e:
            self.update_status(f"Error loading CV components: {e}")
    
    def toggle_camera(self):
        """Switch camera on/off."""
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()
    
    def start_camera(self):
        """Start camera."""
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
                messagebox.showerror("Error", "Camera could not be opened")
                return
            
            # Test if camera can provide a frame
            test_frame = self.camera.get_frame_raw()
            if test_frame is None:
                messagebox.showerror("Error", "Camera is not providing frames")
                self.camera.release()
                return
            
            self.camera_running = True
            self.camera_btn.configure(text="Stop Camera", bg='#E74C3C')
            
            # Initialize calibration components
            self.calibration = calibration.CameraCalibration(
                ref_img="resources/dartboard-gerade.jpg", 
                debug=self.debug_mode
            )
            
            # Set initial status
            self.dartboard_calibrated = False
            self.dartboard_status_var.set("Dartboard not calibrated")

            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

            self.update_status("Camera started - Press 'Calibrate' for dartboard calibration")
            
        except Exception as e:
            error_msg = f"Error starting camera: {e}"
            messagebox.showerror("Error", error_msg)
            self.update_status(f"Camera error: {e}")
            
            try:
                if hasattr(self, 'camera') and self.camera:
                    self.camera.release()
                self.camera_running = False
                self.camera_btn.configure(text="Start Camera", bg='#27AE60')
            except:
                pass
    
    def stop_camera(self):
        """Stop camera."""
        self.camera_running = False
        if self.camera:
            self.camera.release()

        self.camera_btn.configure(text="Start Camera", bg='#27AE60')
        self.video_label.configure(image='', text="Camera stopped\nClick 'Start Camera'")

        self.update_status("Camera stopped")

    def camera_loop(self):
        """Main loop for camera processing."""
        frame_count = 0
        
        while self.camera_running:
            try:
                frame = self.camera.get_frame_raw()
                if frame is None:
                    time.sleep(0.033)  # Wait briefly if no frame is available
                    continue
                
                frame_count += 1

                # Process frame
                if self.dartboard_calibrated and self.calibration and self.calibration.H is not None:
                    # Use stored calibration for all frames
                    try:
                        processed_frame = self.calibration.warp_frame(frame)
                        if processed_frame is None:
                            processed_frame = frame
                    except Exception as warp_e:
                        print(f"Warping error: {warp_e}")
                        processed_frame = frame
                else:
                    # Show original frame if not calibrated
                    processed_frame = frame

                # Performance: Only fully process every 3rd frame for YOLO
                full_processing = (frame_count % 3 == 0)

                # Manage cooldown logic
                if self.dart_detection_cooldown > 0:
                    self.dart_detection_cooldown -= 1
                
                if self.turn_complete_cooldown > 0:
                    self.turn_complete_cooldown -= 1
            
                    if frame_count % 10 == 0:
                        self.root.after(0, self.update_turn_display)
                    
                    # Automatic end of cooldown after time has elapsed
                    if self.turn_complete_cooldown <= 0:
                        print("Turn-Complete-Cooldown automatically expired!")
                        self.board_empty_check_frames = 0
                        self.reset_dart_detection_state()
                        
                        # Check if round is ready to complete and finish it
                        if self.turn_ready_to_complete:
                            print("Cooldown expired and round ready - finish round")
                            self.turn_ready_to_complete = False
                            self.root.after(0, self.complete_current_turn)
                        
                        # Update turn display immediately
                        self.root.after(0, self.update_turn_display)
                
                # Create display frame
                display_frame = processed_frame.copy()
                
                # Overlay dartboard template 
                if self.score_predictor and self.score_predictor.is_calibrated():
                    try:
                        display_frame = self.score_predictor.overlay_dartboard_template(
                            display_frame, 
                            show_numbers=True,
                            template_color=(0, 255, 255)
                        )
                    except Exception as overlay_e:
                        print(f"Overlay error: {overlay_e}")
                        display_frame = processed_frame
                
                # Show saved dart scores
                if (self.score_predictor and self.score_predictor.is_calibrated() and 
                    self.last_dart_positions and not full_processing):
                    try:
                        # Use saved dart positions for consistent display
                        display_frame, _ = self.score_predictor.process_dart_detections(
                            display_frame, 
                            self.last_dart_positions, 
                            show_scores=True
                        )
                    except Exception as cached_detection_e:
                        print(f"Cached dart detection error: {cached_detection_e}")
                
                # Only every 3rd frame YOLO processing and dart detection
                if not full_processing:
                    self.update_video_display(display_frame)
                    continue
                
                # YOLO prediction only on processed frame
                try:
                    results = self.predictor.predict(processed_frame)
                except Exception as pred_e:
                    print(f"YOLO prediction error: {pred_e}")
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
                                print(f"Box processing error: {box_e}")
                                continue
                
                # Calibrate score predictor if needed
                if (len(dartboard_points) >= 3 and 
                    self.score_predictor and 
                    not self.score_predictor.is_calibrated()):
                    try:
                        if self.score_predictor.calibrate_dartboard(dartboard_points):
                            self.root.after(0, lambda: self.update_status("Dartboard point system calibrated!"))
                    except Exception as score_cal_e:
                        print(f"Score predictor calibration error: {score_cal_e}")
                
                # Process dart detections and add score display
                if self.score_predictor and self.score_predictor.is_calibrated() and dart_positions:
                    try:
                        # Filter and stabilize dart positions
                        filtered_positions = self.filter_duplicate_darts(dart_positions)
                        
                        # Reset board_empty_check_frames since darts were detected
                        self.board_empty_check_frames = 0
                        
                        # Score calculation and display for current positions
                        display_frame, dart_scores = self.score_predictor.process_dart_detections(
                            display_frame, 
                            filtered_positions, 
                            show_scores=True
                        )
                        
                        # Save positions for consistent display between frames
                        self.last_dart_positions = filtered_positions.copy()
                        
                        # Check if detection should be processed
                        should_process = self.should_process_dart_detection(filtered_positions)
                        
                        # Process detected darts automatically
                        if should_process and dart_scores:
                            print(f"✓ should_process=True, {len(dart_scores)} dart scores: {dart_scores}")
                            print(f"  game_active: {self.game_state.game_active}")
                            print(f"  current_player: {self.game_state.get_current_player().name if self.game_state.get_current_player() else 'None'}")
                            print(f"  current_dart_count: {self.game_state.current_dart_count}")
                            print(f"  turn_complete_cooldown: {self.turn_complete_cooldown}")
                            
                            # Set longer cooldown to prevent spam
                            self.dart_detection_cooldown = 60 
                            
                            self.last_dart_scores = dart_scores.copy() if dart_scores else []
                            self.root.after(0, lambda scores=dart_scores: self.process_detected_darts(scores))
                            
                        else:
                            print(f"✗ should_process={should_process}, dart_scores={len(dart_scores) if dart_scores else 0}, game_active={self.game_state.game_active}")
                            if self.dart_detection_cooldown > 0:
                                print(f"  Dart detection cooldown active: {self.dart_detection_cooldown}")
                            if self.turn_complete_cooldown > 0:
                                print(f"  Turn complete cooldown active: {self.turn_complete_cooldown}")
                            if not dart_scores:
                                print(f"  No dart scores")
                            if not self.game_state.game_active:
                                print(f"  Game not active")
                        
                    except Exception as detection_e:
                        print(f"Dart detection error: {detection_e}")
                elif self.score_predictor and self.score_predictor.is_calibrated() and not dart_positions:
                    # No darts detected 
                    self.board_empty_check_frames += 1
                    
                    # After sufficient frames without darts and active turn-complete-cooldown consider board as empty
                    if (self.board_empty_check_frames >= self.board_empty_required_frames and 
                        self.turn_complete_cooldown > 0):
                        print(f"🔄 Board is empty after {self.board_empty_check_frames} frames! Reset turn-complete-cooldown (was {self.turn_complete_cooldown})")
                        self.turn_complete_cooldown = 0
                        self.board_empty_check_frames = 0
                        self.reset_dart_detection_state()
                        
                        # Check if round is ready to complete and finish it
                        if self.turn_ready_to_complete:
                            print("🔄 Board is empty and round ready - finish round")
                            self.turn_ready_to_complete = False
                            self.root.after(0, self.complete_current_turn)
                        
                        # Update turn display
                        self.root.after(0, self.update_turn_display)
                    
                    # Clear dart cache after a while if no cooldown is active
                    if self.dart_detection_cooldown <= 0 and self.turn_complete_cooldown <= 0:
                        self.reset_dart_detection_state()
                
                # Show YOLO detections in debug mode
                if self.debug_mode and results:
                    try:
                        for result in results:
                            annotated = result.plot()
                            # Could show this in a separate window if needed
                    except Exception as debug_e:
                        print(f"Debug display error: {debug_e}")
                
                self.update_video_display(display_frame)
                
            except Exception as e:
                print(f"Error in camera_loop: {e}")
                # For major errors use original frame
                try:
                    if 'frame' in locals() and frame is not None:
                        self.update_video_display(frame)
                except:
                    pass
                time.sleep(0.1)
    
    def update_video_display(self, frame):
        """Update video display."""
        if frame is None:
            return
            
        try:
            # Check if frame is valid
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                print(f"Invalid frame: {frame.shape}")
                return
                
            # Scale frame to maximum display size
            height, width = frame.shape[:2]
            max_width, max_height = 540, 420  
            
            # Calculate scaling factor to maintain aspect ratio
            scale = min(max_width/width, max_height/height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Scale frame
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert to RGB and create PhotoImage
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image)
            
            # Update label in main thread
            self.root.after(0, self._update_video_label, photo)
            
        except Exception as e:
            print(f"Error in video display: {e}")
            
            try:
                self.root.after(0, lambda: self.video_label.configure(
                    image='', 
                    text=f"Video error:\n{str(e)[:50]}...",
                    fg='red'
                ))
            except:
                pass
    
    def _update_video_label(self, photo):
        """Update video label in main thread."""
        try:
            if self.video_label and photo:
                self.video_label.configure(image=photo, text="", fg='white')
                self.video_label.image = photo  # Keep reference to prevent garbage collection
        except Exception as e:
            print(f"Error updating video label: {e}")
            try:
                if self.video_label:
                    self.video_label.configure(
                        image='', 
                        text="Video Display Error", 
                        fg='red'
                    )
            except:
                pass
    
    def process_detected_darts(self, dart_scores: List[Tuple[int, str]]):
        """Process detected darts."""
        print(f"process_detected_darts: Received {len(dart_scores) if dart_scores else 0} dart scores")
        print(f"process_detected_darts: game_active = {self.game_state.game_active}")
        
        if not dart_scores:
            print("process_detected_darts: No dart scores")
            return
            
        # Check if we're currently processing a dart
        if hasattr(self, '_currently_processing_dart') and self._currently_processing_dart:
            print("process_detected_darts: Dart is already being processed, ignore")
            return
            
        # Set processing flag
        self._currently_processing_dart = True
        
        try:
            if not self.game_state.game_active:
                print("process_detected_darts: Game not active - start game automatically")
                # Automatically start a game if players are available
                if len(self.game_state.players) >= 2:
                    self.game_state.start_game()
                    self.update_status("Game automatically started due to dart detection")
                    self.update_all_displays()
                else:
                    print("process_detected_darts: Not enough players for automatic start")
                    return
            
            # Only process if waiting for darts
            current_player = self.game_state.get_current_player()
            if not current_player:
                print("process_detected_darts: No current player")
                return
            
            # Check if still expecting darts for this round
            if self.game_state.current_dart_count >= 3:
                print(f"process_detected_darts: Round already complete ({self.game_state.current_dart_count}/3)")
                return
            
            # Take only the first detected dart and remove similar ones from processed_dart_positions
            score, description = dart_scores[0]
            print(f"Process ONE dart: {score} ({description}) for {current_player.name} (Dart {self.game_state.current_dart_count + 1}/3)")
            
            # Find the corresponding new position and mark it as processed
            # Search for the first position that has not already been processed
            new_position_found = None
            for i, (dart_score, dart_desc) in enumerate(dart_scores):
                # Try to find the corresponding position
                if i < len(self.last_dart_positions):
                    candidate_pos = self.last_dart_positions[i]
                    # Check if this position has already been processed
                    is_already_processed = any(
                        self.are_positions_similar(candidate_pos, processed_pos, self.dart_position_tolerance)
                        for processed_pos in self.processed_dart_positions
                    )
                    if not is_already_processed:
                        new_position_found = candidate_pos
                        # Use the corresponding score
                        score, description = dart_score, dart_desc
                        print(f"Use position {new_position_found} with score {score} ({description})")
                        break
            
            # Add the new position to processed ones
            if new_position_found:
                self.processed_dart_positions.append(new_position_found)
                print(f"Position {new_position_found} added to processed_dart_positions")
            
            turn_complete = self.game_state.add_dart_score(score, description)
            
    
            print(f"Score added, cooldown to {self.dart_detection_cooldown}")
            
            if turn_complete:
                print("process_detected_darts: Round complete, start board-clearing pause")
                # 10-second-cooldown after complete round
                self.turn_complete_cooldown = self.turn_complete_cooldown_duration
                self.board_empty_check_frames = 0
                print(f"Turn-Complete-Cooldown activated: {self.turn_complete_cooldown} frames (10 seconds)")
                # Set flag that round is ready to complete, but wait for cooldown to end
                self.turn_ready_to_complete = True
            else:
                print(f"process_detected_darts: Round not yet complete, wait for more darts")
            
            # Update displays
            self.root.after(0, self.update_all_displays)
            
            # Clear detection cache after short delay
            self.root.after(100, self.clear_dart_cache_after_processing)
            
        finally:
            # Reset processing flag after short delay
            self.root.after(200, self.reset_processing_flag)
    
    def reset_processing_flag(self):
        """Reset processing flag."""
        self._currently_processing_dart = False
        print("Processing flag reset")
    
    def complete_current_turn(self):
        """Complete current turn - only called when cooldown has expired."""
        print("complete_current_turn: Complete round (after cooldown)")
        
        # Reset processing flag
        self._currently_processing_dart = False
        
        if self.game_state.winner:
            print("complete_current_turn: Game ended, there is a winner")
            self.end_game()
        else:
            self.game_state.complete_turn()
            # After turn change delete all dart positions
            print(f"Turn change: Delete {len(self.processed_dart_positions)} processed dart positions")
            print(f"Turn change: Delete {len(self.blacklisted_dart_positions)} blacklisted dart positions")
            self.processed_dart_positions = []
            self.blacklisted_dart_positions = []  # Clear blacklist on turn change
            self.reset_dart_detection_state()  # Complete reset after turn change
            self.update_all_displays()
    
    def add_player(self):
        """Add new player."""
        name = simpledialog.askstring("Add Player", "Player name:")
        if name and name.strip():
            player = self.game_state.add_player(name.strip())
            self.create_player_widget(player)
            self.update_status(f"Player {name} added")
            self.update_turn_display()
    
    def remove_selected_player(self):
        """Remove selected player."""
        if not self.game_state.players:
            messagebox.showwarning("Warning", "No players available")
            return
        
        # Show dialog to select player
        names = [p.name for p in self.game_state.players]
        selected = simpledialog.askstring("Remove Player", f"Available players: {', '.join(names)}\nEnter name:")
        
        if selected:
            for player in self.game_state.players:
                if player.name == selected:
                    self.game_state.remove_player(player)
                    self.refresh_players_display()
                    self.update_status(f"Player {selected} removed")
                    self.update_turn_display()
                    return
            messagebox.showwarning("Warning", "Player not found")
    
    def create_player_widget(self, player: Player):
        """Create player widget."""
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
        score_label = tk.Label(header_frame, text=f"Points: {player.current_score}", 
                              font=('Arial', 14, 'bold'), 
                              fg='#E74C3C', bg='#2C3E50')
        score_label.pack(side='right')
        
        # Stats frame
        stats_frame = tk.Frame(player_frame, bg='#2C3E50')
        stats_frame.pack(fill='x', padx=10, pady=5)
    
        darts_label = tk.Label(stats_frame, text=f"Darts: {player.darts_thrown}", 
                              font=('Arial', 10), fg='#BDC3C7', bg='#2C3E50')
        darts_label.pack(side='left')
        
        avg_label = tk.Label(stats_frame, text=f"Average: {player.get_average()*3:.1f}", 
                            font=('Arial', 10), fg='#BDC3C7', bg='#2C3E50')
        avg_label.pack(side='right')
        
        # Current turn scores
        if (self.game_state.get_current_player() == player and 
            self.game_state.current_turn_scores):
            turn_frame = tk.Frame(player_frame, bg='#2C3E50')
            turn_frame.pack(fill='x', padx=10, pady=5)
            
            turn_label = tk.Label(turn_frame, 
                                 text=f"Current Round: {' + '.join(map(str, self.game_state.current_turn_scores))}", 
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
        """Update player display."""
        # Clear all player widgets
        for widget in self.player_frames.values():
            widget.destroy()
        self.player_frames.clear()
        
        # Recreate all player widgets
        for player in self.game_state.players:
            self.create_player_widget(player)
    
    def update_all_displays(self):
        """Update all displays."""
        self.refresh_players_display()
        self.update_turn_display()
    
    def update_turn_display(self):
        """Update turn display."""
        if self.turn_complete_cooldown > 0:
            # Calculate remaining seconds
            remaining_seconds = self.turn_complete_cooldown / 30 
            if remaining_seconds > 1.0:
                self.turn_info_var.set(f"Waiting pause - Clear board! ({remaining_seconds:.1f}s)")
            else:
                self.turn_info_var.set(f"Waiting pause - Clear board! (0.{int(remaining_seconds*10)}s)")
            
            # Disable relevant buttons during cooldown
            if hasattr(self, 'next_turn_btn'):
                self.next_turn_btn.configure(state='disabled')
                
        elif self.game_state.players and self.game_state.game_active:
            current = self.game_state.get_current_player()
            dart_info = f"Dart {self.game_state.current_dart_count + 1}/3" if current else ""
            self.turn_info_var.set(f"{current.name} is up - {dart_info}")
            
            # Reactivate buttons when no cooldown
            if hasattr(self, 'next_turn_btn'):
                self.next_turn_btn.configure(state='normal')
                
        elif self.game_state.players:
            self.turn_info_var.set(f"{len(self.game_state.players)} players ready - Start game")
            if hasattr(self, 'next_turn_btn'):
                self.next_turn_btn.configure(state='normal')
        else:
            self.turn_info_var.set("Add players to start")
            if hasattr(self, 'next_turn_btn'):
                self.next_turn_btn.configure(state='normal')
    
    def new_game(self):
        """Start new game."""
        if len(self.game_state.players) < 2:
            messagebox.showwarning("Warning", "At least 2 players required")
            return
        
        result = messagebox.askyesno("New Game", "Start new game?")
        if result:
            # Reset all flags for new game
            self.turn_ready_to_complete = False
            self.turn_complete_cooldown = 0
            self.board_empty_check_frames = 0
            self.blacklisted_dart_positions = []  # Clear blacklist for new game
            self.reset_dart_detection_state()
            
            self.game_state.start_game()
            self.update_status("New game started!")
            self.update_all_displays()
    
    def next_turn(self):
        """Switch to next turn."""
        if not self.game_state.game_active:
            return
        
        # Check turn-complete-cooldown also for manual turn change
        if self.turn_complete_cooldown > 0:
            remaining_seconds = self.turn_complete_cooldown / 30
            messagebox.showwarning("Warning", f"Board must be cleared first! Wait {remaining_seconds:.1f} more seconds.")
            return
        
        # Reset dart detection state on turn change
        self.reset_dart_detection_state()
        self.blacklisted_dart_positions = []  # Clear blacklist on turn change
        self.turn_ready_to_complete = False  # Reset flag on manual turn change
        print("Next round - Dart detection state reset")
        print(f"Next round - Blacklist cleared")
        
        self.game_state.complete_turn()
        if self.game_state.winner:
            self.end_game()
        else:
            self.update_all_displays()
    
    def undo_last_dart(self):
        """Undo last dart."""
        if not self.game_state.game_active:
            messagebox.showwarning("Warning", "No active game!")
            return
            
        # Undo is allowed even during board-clearing pause
            
        if self.game_state.current_dart_count > 0:
            # Note if we're going back from the third dart
            was_third_dart = (self.game_state.current_dart_count == 3)
            
            # Reset corresponding detection states when a dart is undone
            removed_position = None
            if self.processed_dart_positions:
                removed_position = self.processed_dart_positions.pop()
                print(f"Undo: Remove position {removed_position} from processed_dart_positions")
                
                # Add the position to the blacklist for the entire turn
                self.blacklisted_dart_positions.append(removed_position)
                print(f"Undo: Add position {removed_position} to blacklist")
                print(f"Blacklisted positions: {len(self.blacklisted_dart_positions)}")
            
            # Remove the corresponding position from last_dart_positions and last_dart_scores
            # Find the position that corresponds to the removed processed_position
            if removed_position and self.last_dart_positions:
                # Search for the corresponding position in last_dart_positions
                for i, pos in enumerate(self.last_dart_positions):
                    if self.are_positions_similar(pos, removed_position, self.dart_position_tolerance):
                        removed_cached_position = self.last_dart_positions.pop(i)
                        print(f"Undo: Also remove position {removed_cached_position} from last_dart_positions (Index {i})")
                        if i < len(self.last_dart_scores):
                            removed_cached_score = self.last_dart_scores.pop(i)
                            print(f"Undo: Also remove score {removed_cached_score} from last_dart_scores (Index {i})")
                        break
                else:
                    print(f"Undo: Could not find position {removed_position} in last_dart_positions")
            
            if was_third_dart:
                print("Undo from third dart: Disable Turn-Complete-Cooldown and flags")
                self.turn_complete_cooldown = 0
                self.turn_ready_to_complete = False
                self.board_empty_check_frames = 0
                print(f"Turn-Complete-Cooldown disabled, turn is incomplete again")
            
            # Reset Turn-Ready-Flag generally on undo
            if self.turn_ready_to_complete:
                print("Undo: Reset turn_ready_to_complete")
                self.turn_ready_to_complete = False
            
            self.game_state.undo_last_dart()
            self.update_all_displays()
            self.update_status("Last dart undone")
        else:
            messagebox.showinfo("Information", "No dart available to undo")
    
    def add_manual_score(self):
        """Add manual score."""
        try:
            score_text = self.manual_score_var.get().strip()
            if not score_text:
                return
            
            score = int(score_text)
            if score < 0 or score > 180:
                messagebox.showerror("Error", "Score must be between 0 and 180")
                return
            
            # Check if game is active and if we have a current player
            if not self.game_state.game_active:
                messagebox.showwarning("Warning", "Please start a game first!")
                return
            
            current_player = self.game_state.get_current_player()
            if not current_player:
                messagebox.showwarning("Warning", "No active player!")
                return
            
            # Check turn-complete-cooldown 
            if self.turn_complete_cooldown > 0:
                remaining_seconds = self.turn_complete_cooldown / 30
                messagebox.showwarning("Warning", f"Board must be cleared first! Wait {remaining_seconds:.1f} more seconds.")
                return
            
            # Debug output
            print(f"Add manually {score} points for {current_player.name}")
            
            turn_complete = self.game_state.add_dart_score(score, f"Manual: {score}")
            
            # Same treatment as automatic detection
            if turn_complete:
                print("add_manual_score: Round complete, start board-clearing pause")
                # 10-second-cooldown after complete round
                self.turn_complete_cooldown = self.turn_complete_cooldown_duration
                self.board_empty_check_frames = 0
                print(f"Turn-Complete-Cooldown activated: {self.turn_complete_cooldown} frames (10 seconds)")
                # Set flag that round is ready to complete, but wait for cooldown to end
                self.turn_ready_to_complete = True
            else:
                print(f"add_manual_score: Round not yet complete, wait for more darts")
            
            self.manual_score_var.set("")
            self.update_all_displays()
            self.update_status(f"Points {score} added for {current_player.name}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")
    
    def end_game(self):
        """End game."""
        winner = self.game_state.winner
        if winner:
            messagebox.showinfo("Game Over!", f"🎉 {winner.name} has won! 🎉")
            self.game_state.game_active = False
            self.update_status(f"Game over - Winner: {winner.name}")
            self.update_all_displays()
    
    def calibrate_dartboard(self):
        """Manually calibrate dartboard."""
        if not self.camera_running:
            messagebox.showwarning("Warning", "Please start camera first")
            return
        
        # Get current frame for calibration
        if not self.camera:
            messagebox.showerror("Error", "Camera not available")
            return
        
        try:
            # Get current frame
            current_frame = self.camera.get_frame_raw()
            if current_frame is None:
                messagebox.showerror("Error", "Cannot get frame from camera")
                return
            
            # Reset calibration status
            self.dartboard_calibrated = False
            self.dartboard_status_var.set("Dartboard: Calibration running...")
            self.update_status("Starting manual dartboard calibration...")
            
            # Try calibration with current frame
            success, result = self.calibration.initial_calibration(current_frame)
            
            if success:
                self.dartboard_calibrated = True
                self.dartboard_status_var.set("Dartboard: Calibrated ✓")
                self.update_status("Dartboard successfully calibrated!")
                
                # Reset score predictor for new calibration
                self.score_predictor = score_prediction.DartboardScorePredictor()
                
                messagebox.showinfo(
                    "Success", 
                    "Dartboard was successfully calibrated!\n\n"
                    "All following frames will now be automatically transformed."
                )
            else:
                self.dartboard_calibrated = False
                self.dartboard_status_var.set("Dartboard: Calibration failed")
                self.update_status("Dartboard calibration failed")
                
                messagebox.showwarning(
                    "Calibration failed", 
                    "The dartboard could not be calibrated.\n\n"
                    "Make sure that:\n"
                    "• The dartboard is fully visible in the image\n"
                    "• The reference image is available\n"
                    "• Sufficient distinctive points are detected\n\n"
                    "Try again with better positioning."
                )
                
        except Exception as e:
            self.dartboard_calibrated = False
            self.dartboard_status_var.set("Dartboard: Calibration error")
            error_msg = f"Error during calibration: {e}"
            self.update_status(error_msg)
            messagebox.showerror("Calibration error", error_msg)
    
    def reset_calibration(self):
        """Reset calibration."""
        self.dartboard_calibrated = False
        self.dartboard_status_var.set("Dartboard: Not calibrated")
        
        # Reset calibration matrix
        if self.calibration:
            self.calibration.H = None
        
        # Reset score predictor
        self.score_predictor = score_prediction.DartboardScorePredictor()
        
        self.update_status("Calibration reset")
        messagebox.showinfo("Calibration reset", "The dartboard calibration has been reset.")
    
    def update_status(self, message: str):
        """Update status message."""
        self.status_var.set(message)
        print(f"Status: {message}")
    
    def save_game(self):
        """Save game."""
        if not self.game_state.players:
            messagebox.showwarning("Warning", "No players available")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save game"
        )
        
        if filename:
            try:
                self.game_state.save_game(filename)
                self.update_status(f"Game saved: {filename}")
                messagebox.showinfo("Success", "Game successfully saved!")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving: {e}")
    
    def load_game(self):
        """Load game."""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load game"
        )
        
        if filename:
            try:
                self.game_state.load_game(filename)
                self.refresh_players_display()
                self.update_turn_display()
                self.update_status(f"Game loaded: {filename}")
                messagebox.showinfo("Success", "Game successfully loaded!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading: {e}")
    
    def camera_settings(self):
        """Camera settings dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Camera Settings")
        dialog.geometry("400x300")
        dialog.configure(bg='#34495E')
        
        # Camera source selection
        tk.Label(dialog, text="Camera Source:", font=('Arial', 12, 'bold'),
                fg='white', bg='#34495E').pack(pady=10)
        
        source_var = tk.StringVar(value="webcam" if not self.use_image_folder else "folder")
        
        webcam_radio = tk.Radiobutton(dialog, text="Webcam", variable=source_var, 
                                     value="webcam", font=('Arial', 11),
                                     fg='white', bg='#34495E', selectcolor='#2C3E50')
        webcam_radio.pack()
        
        folder_radio = tk.Radiobutton(dialog, text="Image Folder", variable=source_var, 
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
        tk.Label(dialog, text="Image Folder Path:", font=('Arial', 11),
                fg='white', bg='#34495E').pack(pady=(20, 5))
        
        folder_var = tk.StringVar(value=self.image_folder_path)
        folder_entry = tk.Entry(dialog, textvariable=folder_var, font=('Arial', 11), width=40)
        folder_entry.pack()
        
        browse_btn = tk.Button(dialog, text="Browse", 
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
            self.update_status("Camera settings updated")
        
        apply_btn = tk.Button(button_frame, text="Apply", command=apply_settings,
                             bg='#27AE60', fg='white', font=('Arial', 11))
        apply_btn.pack(side='left', padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                              bg='#E74C3C', fg='white', font=('Arial', 11))
        cancel_btn.pack(side='left', padx=5)
    
    def browse_folder(self, folder_var):
        """Browse folder."""
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            folder_var.set(folder)
    
    def game_settings(self):
        """Game settings dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Game Settings")
        dialog.geometry("350x250")
        dialog.configure(bg='#34495E')
        
        # Game mode
        tk.Label(dialog, text="Game Mode:", font=('Arial', 12, 'bold'),
                fg='white', bg='#34495E').pack(pady=10)
        
        mode_var = tk.StringVar(value=self.game_state.game_mode)
        mode_combo = ttk.Combobox(dialog, textvariable=mode_var, 
                                 values=["501", "301", "Cricket"], state="readonly")
        mode_combo.pack()
        
        # Auto advance
        auto_var = tk.BooleanVar(value=self.game_state.auto_advance)
        auto_check = tk.Checkbutton(dialog, text="Automatically advance to next round", 
                                   variable=auto_var, font=('Arial', 11),
                                   fg='white', bg='#34495E', selectcolor='#2C3E50')
        auto_check.pack(pady=10)
        
        # Legs to win
        tk.Label(dialog, text="Legs to win:", font=('Arial', 11),
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
            self.update_status("Game settings updated")
        
        apply_btn = tk.Button(button_frame, text="Apply", command=apply_settings,
                             bg='#27AE60', fg='white', font=('Arial', 11))
        apply_btn.pack(side='left', padx=5)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=dialog.destroy,
                              bg='#E74C3C', fg='white', font=('Arial', 11))
        cancel_btn.pack(side='left', padx=5)
    
    def toggle_debug(self):
        """Toggle debug mode."""
        self.debug_mode = not self.debug_mode
        status = "enabled" if self.debug_mode else "disabled"
        self.update_status(f"Debug mode {status}")
        messagebox.showinfo("Debug Mode", f"Debug mode {status}")
    
    def run(self):
        """Run GUI."""
        self.root.mainloop()
    
    def on_closing(self):
        """On application closing."""
        self.stop_camera()
        self.root.destroy()
    
    def are_positions_similar(self, pos1: Tuple[int, int], pos2: Tuple[int, int], tolerance: int = 30) -> bool:
        """Check if two positions are similar enough (same dart)."""
        distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        return distance <= tolerance
    
    def filter_duplicate_darts(self, dart_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filter duplicate/similar dart positions."""
        if not dart_positions:
            return []
        
        filtered_positions = []
        
        for pos in dart_positions:
            # Check if position is on blacklist
            is_blacklisted = any(
                self.are_positions_similar(pos, blacklisted_pos, self.dart_position_tolerance)
                for blacklisted_pos in self.blacklisted_dart_positions
            )
            
            if is_blacklisted:
                print(f"Position {pos} is blacklisted, ignore for entire turn")
                continue
            
            # Check if this position is too similar to already filtered ones
            is_duplicate = False
            for existing_pos in filtered_positions:
                if self.are_positions_similar(pos, existing_pos, self.dart_position_tolerance):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_positions.append(pos)
        
        return filtered_positions
    
    def should_process_dart_detection(self, dart_positions: List[Tuple[int, int]]) -> bool:
        """Decide if dart detection should be processed (anti-spam)."""
        print(f"should_process_dart_detection: Input {len(dart_positions)} dart positions")
        
        # Turn-Complete-Cooldown check
        if self.turn_complete_cooldown > 0:
            print(f"  Turn-Complete-Cooldown active: {self.turn_complete_cooldown} Frames")
            return False
        
        # Normal cooldown check
        if self.dart_detection_cooldown > 0:
            print(f"  Cooldown active: {self.dart_detection_cooldown}")
            return False
        
        # No darts detected
        if not dart_positions:
            print(f"  No dart positions")
            return False
        
        # Check only against already processed positions
        new_positions = []
        for pos in dart_positions:
            is_already_processed = any(
                self.are_positions_similar(pos, processed_pos, self.dart_position_tolerance)
                for processed_pos in self.processed_dart_positions
            )
            if not is_already_processed:
                new_positions.append(pos)
        
        print(f"  New positions: {len(new_positions)} of {len(dart_positions)}")
        
        # Only process if we have new positions
        if len(new_positions) > 0:
            print(f"  ✓ New dart positions found: {len(new_positions)}")
            return True
        
        # No new positions
        print(f"  ✗ No new positions")
        return False
    
    def reset_dart_detection_state(self):
        """Reset dart detection state."""
        print(f"reset_dart_detection_state: Delete {len(self.last_dart_positions)} last_dart_positions, {len(self.processed_dart_positions)} processed_dart_positions")
        self.last_dart_positions = []
        self.processed_dart_positions = []
        
        self.last_dart_scores = []
        self.dart_detection_cooldown = 0
    
    def clear_dart_cache_after_processing(self):
        """Clear dart cache after processing to prevent double counting."""
        print("Clear dart cache after processing")
        print(f"  Before clearing: last_dart_positions={len(self.last_dart_positions)}, processed={len(self.processed_dart_positions)}")
        
        # Keep last_dart_positions for display, only clear when all have been processed
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
            print("  All positions processed - clear display cache")
            self.last_dart_positions = []
            self.last_dart_scores = []
        else:
            print(f"  Still unprocessed positions - keep display cache")
        
        print(f"  After clearing: last_dart_positions={len(self.last_dart_positions)}, processed_dart_positions={len(self.processed_dart_positions)}")

if __name__ == "__main__":
    # Start the Darts GUI application
    app = DartsGUI()
    app.run()
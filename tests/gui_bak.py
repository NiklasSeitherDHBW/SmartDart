import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))  # Adjust path to include utils module
from utils import camera, calibration, predict, score_prediction


class Player:
    """Represents a single player in the game."""
    
    def __init__(self, name: str, player_id: int):
        self.name = name
        self.id = player_id
        self.current_score = 501  # Default for 501 game
        self.darts_thrown = 0
        self.turn_scores = []  # List of scores per turn
        self.game_history = []  # History of all games
        
    def reset_game(self, starting_score: int = 501):
        """Reset player for a new game."""
        self.current_score = starting_score
        self.darts_thrown = 0
        self.turn_scores = []
    
    def add_turn_score(self, scores: List[int]):
        """Add scores from a turn (up to 3 darts)."""
        turn_total = sum(scores)
        self.turn_scores.append(scores)
        self.darts_thrown += len(scores)
        return turn_total
    
    def get_average(self) -> float:
        """Calculate player's average score per dart."""
        if self.darts_thrown == 0:
            return 0.0
        total_score = 501 - self.current_score
        return total_score / self.darts_thrown
    
    def to_dict(self) -> dict:
        """Convert player to dictionary for saving."""
        return {
            'name': self.name,
            'id': self.id,
            'current_score': self.current_score,
            'darts_thrown': self.darts_thrown,
            'turn_scores': self.turn_scores,
            'game_history': self.game_history
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Player':
        """Create player from dictionary."""
        player = cls(data['name'], data['id'])
        player.current_score = data.get('current_score', 501)
        player.darts_thrown = data.get('darts_thrown', 0)
        player.turn_scores = data.get('turn_scores', [])
        player.game_history = data.get('game_history', [])
        return player


class GameState:
    """Manages the overall game state."""
    
    def __init__(self):
        self.players: List[Player] = []
        self.current_player_index = 0
        self.current_dart_count = 0  # Darts thrown in current turn (0-3)
        self.current_turn_scores = []  # Scores for current turn
        self.game_mode = "501"  # 501, Cricket, Around the Clock, etc.
        self.game_active = False
        self.winner: Optional[Player] = None
        
    def add_player(self, name: str) -> Player:
        """Add a new player to the game."""
        player_id = len(self.players) + 1
        player = Player(name, player_id)
        self.players.append(player)
        return player
    
    def remove_player(self, player: Player):
        """Remove a player from the game."""
        if player in self.players:
            self.players.remove(player)
    
    def start_game(self, game_mode: str = "501"):
        """Start a new game."""
        if len(self.players) < 2:
            raise ValueError("Need at least 2 players to start a game")
        
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
        """Get the currently active player."""
        if not self.players:
            return None
        return self.players[self.current_player_index]
    
    def add_dart_score(self, score: int, description: str) -> bool:
        """Add a dart score for the current player. Returns True if turn is complete."""
        if not self.game_active or not self.players:
            return False
        
        current_player = self.get_current_player()
        self.current_turn_scores.append(score)
        self.current_dart_count += 1
        
        # Check if turn is complete (3 darts or bust)
        turn_complete = False
        if self.current_dart_count >= 3:
            turn_complete = True
        elif self.game_mode == "501":
            # Check for bust in 501
            turn_total = sum(self.current_turn_scores)
            new_score = current_player.current_score - turn_total
            if new_score < 0 or (new_score == 0 and score % 2 != 0):  # Must finish on double
                turn_complete = True
        
        if turn_complete:
            self.complete_turn()
        
        return turn_complete
    
    def complete_turn(self):
        """Complete the current player's turn."""
        if not self.players:
            return
        
        current_player = self.get_current_player()
        turn_total = sum(self.current_turn_scores)
        
        if self.game_mode == "501":
            new_score = current_player.current_score - turn_total
            last_dart = self.current_turn_scores[-1] if self.current_turn_scores else 0
            
            # Check for valid finish (must be exactly 0 and finish on double)
            if new_score == 0 and last_dart % 2 == 0 and last_dart > 0:
                current_player.current_score = 0
                self.winner = current_player
                self.game_active = False
            elif new_score < 0 or (new_score == 0 and last_dart % 2 != 0):
                # Bust - no score change
                pass
            else:
                current_player.current_score = new_score
        
        # Record the turn
        current_player.add_turn_score(self.current_turn_scores.copy())
        
        # Move to next player
        self.current_player_index = (self.current_player_index + 1) % len(self.players)
        self.current_dart_count = 0
        self.current_turn_scores = []
    
    def save_game(self, filepath: str):
        """Save current game state to file."""
        data = {
            'players': [player.to_dict() for player in self.players],
            'current_player_index': self.current_player_index,
            'game_mode': self.game_mode,
            'game_active': self.game_active,
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


class DartsGUI:
    """Main GUI application for the darts game."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Darts Game - Computer Vision Edition")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2C3E50')
        
        # Game state
        self.game_state = GameState()
        self.debug_mode = False
        
        # Computer vision components
        self.camera = None
        self.calibration = None
        self.predictor = None
        self.score_predictor = None
        self.camera_thread = None
        self.camera_running = False
        
        # GUI variables
        self.video_label = None
        self.player_frames = {}
        self.status_var = tk.StringVar(value="Welcome to Darts Game!")
        self.turn_info_var = tk.StringVar(value="Add players to start")
        
        # Create GUI
        self.setup_gui()
        self.setup_computer_vision()
        
    def setup_gui(self):
        """Set up the main GUI layout."""
        # Create main frames
        self.create_menu()
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
    def create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Game menu
        game_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Game", menu=game_menu)
        game_menu.add_command(label="New Game", command=self.new_game)
        game_menu.add_command(label="Save Game", command=self.save_game)
        game_menu.add_command(label="Load Game", command=self.load_game)
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.quit)
        
        # Players menu
        players_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Players", menu=players_menu)
        players_menu.add_command(label="Add Player", command=self.add_player)
        players_menu.add_command(label="Remove Player", command=self.remove_player)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Camera Settings", command=self.camera_settings)
        settings_menu.add_command(label="Game Settings", command=self.game_settings)
        settings_menu.add_command(label="Toggle Debug", command=self.toggle_debug)
    
    def create_header(self):
        """Create the header with game info."""
        header_frame = tk.Frame(self.root, bg='#34495E', height=60)
        header_frame.pack(fill='x', padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        # Game title
        title_label = tk.Label(header_frame, text="🎯 DARTS GAME", 
                              font=('Arial', 20, 'bold'), 
                              fg='white', bg='#34495E')
        title_label.pack(side='left', padx=20, pady=15)
        
        # Turn info
        turn_frame = tk.Frame(header_frame, bg='#34495E')
        turn_frame.pack(side='right', padx=20, pady=15)
        
        turn_label = tk.Label(turn_frame, textvariable=self.turn_info_var,
                             font=('Arial', 14, 'bold'),
                             fg='#E74C3C', bg='#34495E')
        turn_label.pack()
    
    def create_main_content(self):
        """Create the main content area."""
        main_frame = tk.Frame(self.root, bg='#2C3E50')
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left panel - Camera and controls
        left_panel = tk.Frame(main_frame, bg='#34495E', width=700)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))
        left_panel.pack_propagate(False)
        
        self.create_camera_panel(left_panel)
        self.create_control_panel(left_panel)
        
        # Right panel - Players and scores
        right_panel = tk.Frame(main_frame, bg='#34495E', width=600)
        right_panel.pack(side='right', fill='both', padx=(5, 0))
        right_panel.pack_propagate(False)
        
        self.create_players_panel(right_panel)
    
    def create_camera_panel(self, parent):
        """Create the camera display panel."""
        camera_frame = tk.LabelFrame(parent, text="Camera Feed", 
                                   font=('Arial', 12, 'bold'),
                                   fg='white', bg='#34495E')
        camera_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Camera display
        self.video_label = tk.Label(camera_frame, bg='black', 
                                   text="Camera not initialized\nClick 'Start Camera' to begin",
                                   fg='white', font=('Arial', 14))
        self.video_label.pack(expand=True, fill='both', padx=5, pady=5)
    
    def create_control_panel(self, parent):
        """Create the game control panel."""
        control_frame = tk.LabelFrame(parent, text="Game Controls", 
                                    font=('Arial', 12, 'bold'),
                                    fg='white', bg='#34495E', height=120)
        control_frame.pack(fill='x', padx=10, pady=(0, 10))
        control_frame.pack_propagate(False)
        
        # Button frame
        button_frame = tk.Frame(control_frame, bg='#34495E')
        button_frame.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Camera controls
        camera_btn_frame = tk.Frame(button_frame, bg='#34495E')
        camera_btn_frame.pack(side='left', fill='y')
        
        self.camera_btn = tk.Button(camera_btn_frame, text="Start Camera", 
                                   font=('Arial', 10, 'bold'),
                                   bg='#27AE60', fg='white',
                                   command=self.toggle_camera, width=12)
        self.camera_btn.pack(pady=2)
        
        calibrate_btn = tk.Button(camera_btn_frame, text="Calibrate", 
                                font=('Arial', 10, 'bold'),
                                bg='#3498DB', fg='white',
                                command=self.calibrate_dartboard, width=12)
        calibrate_btn.pack(pady=2)
        
        # Game controls
        game_btn_frame = tk.Frame(button_frame, bg='#34495E')
        game_btn_frame.pack(side='left', fill='y', padx=(20, 0))
        
        new_game_btn = tk.Button(game_btn_frame, text="New Game", 
                               font=('Arial', 10, 'bold'),
                               bg='#E67E22', fg='white',
                               command=self.new_game, width=12)
        new_game_btn.pack(pady=2)
        
        next_turn_btn = tk.Button(game_btn_frame, text="Next Turn", 
                                font=('Arial', 10, 'bold'),
                                bg='#9B59B6', fg='white',
                                command=self.next_turn, width=12)
        next_turn_btn.pack(pady=2)
        
        # Manual score entry
        manual_frame = tk.Frame(button_frame, bg='#34495E')
        manual_frame.pack(side='right', fill='y')
        
        tk.Label(manual_frame, text="Manual Score:", 
                font=('Arial', 9), fg='white', bg='#34495E').pack()
        
        score_entry_frame = tk.Frame(manual_frame, bg='#34495E')
        score_entry_frame.pack()
        
        self.manual_score_var = tk.StringVar()
        score_entry = tk.Entry(score_entry_frame, textvariable=self.manual_score_var,
                              font=('Arial', 10), width=8)
        score_entry.pack(side='left', padx=(0, 5))
        
        add_score_btn = tk.Button(score_entry_frame, text="Add", 
                                font=('Arial', 9, 'bold'),
                                bg='#E74C3C', fg='white',
                                command=self.add_manual_score, width=6)
        add_score_btn.pack(side='left')
    
    def create_players_panel(self, parent):
        """Create the players and scoring panel."""
        players_frame = tk.LabelFrame(parent, text="Players & Scores", 
                                    font=('Arial', 12, 'bold'),
                                    fg='white', bg='#34495E')
        players_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Players list with scrollbar
        self.players_canvas = tk.Canvas(players_frame, bg='#2C3E50')
        self.players_scrollbar = ttk.Scrollbar(players_frame, orient="vertical", 
                                             command=self.players_canvas.yview)
        self.players_scrollable_frame = tk.Frame(self.players_canvas, bg='#2C3E50')
        
        self.players_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.players_canvas.configure(scrollregion=self.players_canvas.bbox("all"))
        )
        
        self.players_canvas.create_window((0, 0), window=self.players_scrollable_frame, anchor="nw")
        self.players_canvas.configure(yscrollcommand=self.players_scrollbar.set)
        
        self.players_canvas.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.players_scrollbar.pack(side="right", fill="y", pady=5)
        
        # Add player button
        add_player_frame = tk.Frame(players_frame, bg='#34495E')
        add_player_frame.pack(fill='x', padx=5, pady=(0, 5))
        
        add_player_btn = tk.Button(add_player_frame, text="+ Add Player", 
                                 font=('Arial', 11, 'bold'),
                                 bg='#1ABC9C', fg='white',
                                 command=self.add_player)
        add_player_btn.pack(expand=True, fill='x')
    
    def create_status_bar(self):
        """Create the status bar."""
        status_frame = tk.Frame(self.root, bg='#34495E', height=30)
        status_frame.pack(fill='x', side='bottom')
        status_frame.pack_propagate(False)
        
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                              font=('Arial', 10), fg='white', bg='#34495E')
        status_label.pack(side='left', padx=10, pady=5)
    
    def setup_computer_vision(self):
        """Initialize computer vision components."""
        try:
            # Initialize camera (using folder for testing, can be changed to camera index)
            self.camera = camera.VideoStreamViewer(source=Path("training/data/train/okay_images"))
            
            # Initialize calibration
            self.calibration = calibration.CameraCalibration(
                ref_img="resources/dartboard-gerade.jpg", 
                debug=self.debug_mode
            )
            
            # Initialize YOLO predictor
            self.predictor = predict.Predictor(
                model_path="training/runs/train/PerfectTrainingdata-pretrained2/weights/best.pt"
            )
            
            # Initialize dartboard score predictor
            self.score_predictor = score_prediction.DartboardScorePredictor()
            
            self.update_status("Computer vision components initialized successfully")
            
        except Exception as e:
            self.update_status(f"Error initializing CV components: {str(e)}")
            messagebox.showerror("Initialization Error", f"Failed to initialize computer vision components:\n{str(e)}")
    
    def toggle_camera(self):
        """Start or stop the camera feed."""
        if not self.camera_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera feed."""
        try:
            if self.camera is None:
                raise Exception("Camera not initialized")
            
            self.camera.open_connection()
            if not self.camera.isOpened():
                raise Exception("Failed to open camera connection")
            
            self.camera_running = True
            self.camera_btn.config(text="Stop Camera", bg='#E74C3C')
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
            self.update_status("Camera started successfully")
            
        except Exception as e:
            self.update_status(f"Error starting camera: {str(e)}")
            messagebox.showerror("Camera Error", f"Failed to start camera:\n{str(e)}")
    
    def stop_camera(self):
        """Stop the camera feed."""
        self.camera_running = False
        self.camera_btn.config(text="Start Camera", bg='#27AE60')
        
        if self.camera:
            self.camera.release()
        
        # Clear video display
        self.video_label.config(image='', text="Camera stopped", fg='white')
        
        self.update_status("Camera stopped")
    
    def camera_loop(self):
        """Main camera processing loop."""
        initial_calibration_done = False
        
        while self.camera_running:
            try:
                frame = self.camera.get_frame_raw()
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Perform initial calibration on first frame
                if not initial_calibration_done and self.calibration:
                    success, result = self.calibration.initial_calibration(frame)
                    if success:
                        initial_calibration_done = True
                        self.update_status("Initial dartboard calibration completed")
                    else:
                        self.update_status("Calibration failed, trying again...")
                
                # Warp frame if calibration is available
                if initial_calibration_done and self.calibration:
                    warped_frame = self.calibration.warp_frame(frame)
                    if warped_frame is not None:
                        frame = warped_frame
                
                # Run YOLO detection
                if self.predictor:
                    results = self.predictor.predict(frame)
                    
                    # Extract dart positions and dartboard points
                    dart_positions = []
                    dartboard_points = []
                    
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                                y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                                class_id = int(box.cls[0])
                                
                                if class_id == 4:  # Darts
                                    dart_positions.append((x_center, y_center))
                                else:  # Dartboard features
                                    dartboard_points.append((x_center, y_center))
                    
                    # Calibrate dartboard if needed
                    if (len(dartboard_points) >= 3 and 
                        self.score_predictor and 
                        not self.score_predictor.is_calibrated()):
                        
                        if self.score_predictor.calibrate_dartboard(dartboard_points):
                            self.update_status("Dartboard scoring calibrated!")
                    
                    # Process detections and calculate scores
                    display_frame = frame.copy()
                    
                    if self.score_predictor and self.score_predictor.is_calibrated():
                        # Overlay dartboard template
                        display_frame = self.score_predictor.overlay_dartboard_template(
                            display_frame, 
                            show_numbers=True,
                            template_color=(0, 255, 255)
                        )
                        
                        # Process dart detections
                        if dart_positions and self.game_state.game_active:
                            display_frame, dart_scores = self.score_predictor.process_dart_detections(
                                display_frame, 
                                dart_positions, 
                                show_scores=True
                            )
                            
                            # Auto-add detected scores to current player
                            if dart_scores:
                                self.process_detected_darts(dart_scores)
                    
                    # Show debug info if enabled
                    if self.debug_mode:
                        for result in results:
                            annotated = result.plot()
                            # Could display this in a separate debug window
                
                # Convert frame for tkinter display
                self.update_video_display(frame)
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                self.update_status(f"Camera loop error: {str(e)}")
                time.sleep(0.1)
    
    def update_video_display(self, frame):
        """Update the video display with the current frame."""
        try:
            # Resize frame to fit display
            height, width = frame.shape[:2]
            display_width = 640
            display_height = int(height * (display_width / width))
            
            frame_resized = cv2.resize(frame, (display_width, display_height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image and then to PhotoImage
            pil_image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(pil_image)
            
            # Update label (must be done in main thread)
            self.root.after(0, self._update_video_label, photo)
            
        except Exception as e:
            print(f"Error updating video display: {e}")
    
    def _update_video_label(self, photo):
        """Update video label in main thread."""
        try:
            self.video_label.config(image=photo, text='')
            self.video_label.image = photo  # Keep a reference
        except:
            pass  # Label might be destroyed
    
    def process_detected_darts(self, dart_scores: List[Tuple[int, str]]):
        """Process automatically detected dart scores."""
        if not self.game_state.game_active:
            return
        
        # Auto-add scores for current player
        for score, description in dart_scores:
            if self.game_state.current_dart_count < 3:
                turn_complete = self.game_state.add_dart_score(score, description)
                self.update_turn_display()
                
                if turn_complete:
                    self.check_game_end()
                    break
    
    def add_player(self):
        """Add a new player to the game."""
        name = simpledialog.askstring("Add Player", "Enter player name:")
        if name and name.strip():
            player = self.game_state.add_player(name.strip())
            self.create_player_widget(player)
            self.update_status(f"Added player: {name}")
            self.update_turn_display()
    
    def remove_player(self):
        """Remove a player from the game."""
        if not self.game_state.players:
            messagebox.showinfo("No Players", "No players to remove")
            return
        
        # Create selection dialog
        player_names = [p.name for p in self.game_state.players]
        selected = simpledialog.askstring("Remove Player", 
                                        f"Enter player name to remove:\n{', '.join(player_names)}")
        
        if selected:
            player_to_remove = None
            for player in self.game_state.players:
                if player.name.lower() == selected.lower():
                    player_to_remove = player
                    break
            
            if player_to_remove:
                self.game_state.remove_player(player_to_remove)
                self.remove_player_widget(player_to_remove)
                self.update_status(f"Removed player: {selected}")
                self.update_turn_display()
            else:
                messagebox.showerror("Player Not Found", f"Player '{selected}' not found")
    
    def create_player_widget(self, player: Player):
        """Create a widget for displaying player information."""
        player_frame = tk.Frame(self.players_scrollable_frame, bg='#34495E', 
                              relief='raised', bd=2)
        player_frame.pack(fill='x', padx=5, pady=5)
        
        # Player header
        header_frame = tk.Frame(player_frame, bg='#3498DB')
        header_frame.pack(fill='x')
        
        name_label = tk.Label(header_frame, text=f"🎯 {player.name}", 
                            font=('Arial', 14, 'bold'),
                            fg='white', bg='#3498DB')
        name_label.pack(side='left', padx=10, pady=5)
        
        # Score display
        score_frame = tk.Frame(player_frame, bg='#34495E')
        score_frame.pack(fill='x', padx=10, pady=5)
        
        # Current score
        current_score_label = tk.Label(score_frame, text=f"Score: {player.current_score}", 
                                     font=('Arial', 20, 'bold'),
                                     fg='#E74C3C', bg='#34495E')
        current_score_label.pack()
        
        # Stats
        stats_frame = tk.Frame(player_frame, bg='#34495E')
        stats_frame.pack(fill='x', padx=10, pady=(0, 5))
        
        darts_label = tk.Label(stats_frame, text=f"Darts: {player.darts_thrown}", 
                             font=('Arial', 10),
                             fg='white', bg='#34495E')
        darts_label.pack(side='left')
        
        avg_label = tk.Label(stats_frame, text=f"Avg: {player.get_average():.1f}", 
                           font=('Arial', 10),
                           fg='white', bg='#34495E')
        avg_label.pack(side='right')
        
        # Store references for updates
        self.player_frames[player.id] = {
            'frame': player_frame,
            'score_label': current_score_label,
            'darts_label': darts_label,
            'avg_label': avg_label,
            'header_frame': header_frame
        }
        
        self.update_player_display(player)
    
    def remove_player_widget(self, player: Player):
        """Remove a player widget."""
        if player.id in self.player_frames:
            self.player_frames[player.id]['frame'].destroy()
            del self.player_frames[player.id]
    
    def update_player_display(self, player: Player):
        """Update the display for a specific player."""
        if player.id not in self.player_frames:
            return
        
        widgets = self.player_frames[player.id]
        
        # Update score
        widgets['score_label'].config(text=f"Score: {player.current_score}")
        
        # Update stats
        widgets['darts_label'].config(text=f"Darts: {player.darts_thrown}")
        widgets['avg_label'].config(text=f"Avg: {player.get_average():.1f}")
        
        # Highlight current player
        current_player = self.game_state.get_current_player()
        if current_player and current_player.id == player.id and self.game_state.game_active:
            widgets['header_frame'].config(bg='#E74C3C')
        else:
            widgets['header_frame'].config(bg='#3498DB')
    
    def update_all_player_displays(self):
        """Update all player displays."""
        for player in self.game_state.players:
            self.update_player_display(player)
    
    def update_turn_display(self):
        """Update the turn information display."""
        if not self.game_state.players:
            self.turn_info_var.set("Add players to start")
            return
        
        if not self.game_state.game_active:
            self.turn_info_var.set("Click 'New Game' to start")
            return
        
        current_player = self.game_state.get_current_player()
        if current_player:
            dart_count = self.game_state.current_dart_count
            turn_scores = self.game_state.current_turn_scores
            turn_total = sum(turn_scores) if turn_scores else 0
            
            turn_text = f"{current_player.name}'s Turn - Dart {dart_count + 1}/3"
            if turn_scores:
                turn_text += f" | Turn Total: {turn_total}"
            
            self.turn_info_var.set(turn_text)
    
    def new_game(self):
        """Start a new game."""
        if len(self.game_state.players) < 2:
            messagebox.showwarning("Not Enough Players", "You need at least 2 players to start a game")
            return
        
        try:
            self.game_state.start_game("501")
            self.update_all_player_displays()
            self.update_turn_display()
            self.update_status("New game started!")
            
        except Exception as e:
            messagebox.showerror("Game Start Error", f"Failed to start game:\n{str(e)}")
    
    def next_turn(self):
        """Manually advance to the next player's turn."""
        if self.game_state.game_active:
            self.game_state.complete_turn()
            self.update_all_player_displays()
            self.update_turn_display()
            self.check_game_end()
    
    def add_manual_score(self):
        """Add a manual score entry."""
        try:
            score_text = self.manual_score_var.get().strip()
            if not score_text:
                return
            
            score = int(score_text)
            if score < 0 or score > 180:
                messagebox.showerror("Invalid Score", "Score must be between 0 and 180")
                return
            
            if self.game_state.game_active:
                turn_complete = self.game_state.add_dart_score(score, f"Manual {score}")
                self.update_all_player_displays()
                self.update_turn_display()
                
                if turn_complete:
                    self.check_game_end()
            
            self.manual_score_var.set("")
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number")
    
    def check_game_end(self):
        """Check if the game has ended."""
        if self.game_state.winner:
            winner_name = self.game_state.winner.name
            messagebox.showinfo("Game Over!", f"🎉 {winner_name} wins! 🎉")
            self.update_status(f"Game over - {winner_name} is the winner!")
            
            # Save game to history
            for player in self.game_state.players:
                game_record = {
                    'date': datetime.now().isoformat(),
                    'winner': winner_name,
                    'final_score': player.current_score,
                    'darts_thrown': player.darts_thrown,
                    'average': player.get_average()
                }
                player.game_history.append(game_record)
    
    def calibrate_dartboard(self):
        """Manually trigger dartboard calibration."""
        if self.score_predictor:
            self.update_status("Dartboard calibration will occur automatically when dartboard features are detected")
        else:
            self.update_status("Score predictor not initialized")
    
    def camera_settings(self):
        """Open camera settings dialog."""
        messagebox.showinfo("Camera Settings", "Camera settings dialog would open here")
    
    def game_settings(self):
        """Open game settings dialog."""
        messagebox.showinfo("Game Settings", "Game settings dialog would open here")
    
    def toggle_debug(self):
        """Toggle debug mode."""
        self.debug_mode = not self.debug_mode
        self.update_status(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
    
    def save_game(self):
        """Save the current game state."""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                self.game_state.save_game(filename)
                self.update_status(f"Game saved to {filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save game:\n{str(e)}")
    
    def load_game(self):
        """Load a saved game state."""
        try:
            from tkinter import filedialog
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                # Clear current players
                for player in self.game_state.players.copy():
                    self.remove_player_widget(player)
                
                # Load game
                self.game_state.load_game(filename)
                
                # Recreate player widgets
                for player in self.game_state.players:
                    self.create_player_widget(player)
                
                self.update_all_player_displays()
                self.update_turn_display()
                self.update_status(f"Game loaded from {filename}")
                
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load game:\n{str(e)}")
    
    def update_status(self, message: str):
        """Update the status bar message."""
        self.status_var.set(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
    
    def run(self):
        """Start the GUI application."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing."""
        if self.camera_running:
            self.stop_camera()
        self.root.destroy()


if __name__ == "__main__":
    app = DartsGUI()
    app.run()

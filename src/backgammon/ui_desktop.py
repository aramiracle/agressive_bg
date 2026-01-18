# Kivy-based Backgammon UI - Consistent with ws_server.py and bg_engine.py
# Desktop alternative to the HTML WebSocket client

import threading
import os
import time
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.spinner import Spinner
from kivy.graphics import Color, Line, Triangle, Ellipse, Rectangle, RoundedRectangle
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp

import torch

from .engine import BackgammonGame
from .model import get_model
from .mcts import MCTS
from .config import Config


# --- VISUAL CONSTANTS (Matching HTML UI realistic theme) ---
# Green felt playing surface
COLOR_FELT = (0.18, 0.29, 0.23, 1)
COLOR_FELT_DARK = (0.12, 0.20, 0.16, 1)

# Mahogany wood tones
COLOR_WOOD_DARK = (0.24, 0.15, 0.09, 1)
COLOR_WOOD_MID = (0.36, 0.23, 0.14, 1)
COLOR_WOOD_LIGHT = (0.48, 0.31, 0.20, 1)
COLOR_WOOD_FRAME = (0.55, 0.35, 0.17, 1)

# Points (triangles)
COLOR_POINT_LIGHT = (0.91, 0.86, 0.78, 1)  # Cream/ivory
COLOR_POINT_DARK = (0.42, 0.27, 0.14, 1)   # Dark walnut

# Checkers
COLOR_WHITE_CHECKER = (0.96, 0.94, 0.90, 1)  # Ivory
COLOR_BLACK_CHECKER = (0.17, 0.14, 0.13, 1)  # Dark ebony

# UI elements
COLOR_HIGHLIGHT = (0.35, 0.60, 0.43, 0.5)
COLOR_SELECTED = (0.79, 0.65, 0.36, 0.5)
COLOR_DICE_BG = (0.96, 0.94, 0.88, 1)
COLOR_DICE_DOT = (0.17, 0.14, 0.13, 1)
COLOR_CUBE = (0.96, 0.94, 0.90, 1)
COLOR_CUBE_TEXT = (0.17, 0.14, 0.13, 1)

# Checkpoints directory (relative to project root)
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "checkpoints")


class DiceWidget(Widget):
    """Draws graphical dice faces based on a list of values."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dice_values = []
        self.bind(pos=self.redraw, size=self.redraw)

    def set_dice(self, values):
        self.dice_values = values if values else []
        self.redraw()

    def redraw(self, *args):
        self.canvas.clear()
        if not self.dice_values:
            return

        die_size = dp(40)
        gap = dp(10)
        total_width = (len(self.dice_values) * die_size) + ((len(self.dice_values) - 1) * gap)
        start_x = self.center_x - (total_width / 2)
        y = self.center_y - (die_size / 2)

        for i, val in enumerate(self.dice_values):
            x = start_x + i * (die_size + gap)
            
            with self.canvas:
                Color(*COLOR_DICE_BG)
                RoundedRectangle(pos=(x, y), size=(die_size, die_size), radius=[dp(5)])
                
                Color(*COLOR_DICE_DOT)
                dot_size = die_size / 5
                
                l, c, r = 0.25, 0.5, 0.75
                
                coords = []
                if val == 1: coords = [(c, c)]
                elif val == 2: coords = [(l, r), (r, l)]
                elif val == 3: coords = [(l, r), (c, c), (r, l)]
                elif val == 4: coords = [(l, l), (l, r), (r, l), (r, r)]
                elif val == 5: coords = [(l, l), (l, r), (c, c), (r, l), (r, r)]
                elif val == 6: coords = [(l, l), (l, c), (l, r), (r, l), (r, c), (r, r)]
                
                for (cx, cy) in coords:
                    dot_x = x + (cx * die_size) - (dot_size / 2)
                    dot_y = y + (cy * die_size) - (dot_size / 2)
                    Ellipse(pos=(dot_x, dot_y), size=(dot_size, dot_size))


class BackgammonBoard(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.engine = BackgammonGame()
        self.selected_index = None
        self.legal_moves = []
        
        # AI Components
        self.model = None
        self.mcts = None
        self.model_filename = None
        self.is_ai_thinking = False
        
        # Game Settings (matching ws_server.py)
        self.game_mode = 'human_vs_ai'  # human_vs_ai, ai_vs_human, ai_vs_ai
        self.game_over = False
        self.winner = 0
        self.has_rolled = False  # Track if current player has rolled
        self.match_target = Config.MATCH_TARGET
        
        # AI vs AI controls
        self.ai_auto_play = False
        self.ai_speed = 1.0
        self.ai_auto_event = None
        
        self.bind(pos=self.redraw, size=self.redraw)
        
        # Try to load default model
        self._try_load_default_model()

    def _try_load_default_model(self):
        """Try to load default model from checkpoints folder"""
        paths_to_try = [
            os.path.join(CHECKPOINTS_DIR, "best_model.pt"),
            os.path.join(CHECKPOINTS_DIR, "latest_model.pt")
        ]
        
        for path in paths_to_try:
            if not os.path.exists(path):
                continue
            try:
                self._load_model_sync(path)
                print(f"✅ Model loaded: {path}")
                return
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
        
        print(f"⚠️ No compatible model found")

    def _load_model_sync(self, filepath):
        """Load model synchronously"""
        device = torch.device(Config.DEVICE)
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        # Check model type
        checkpoint_model_type = None
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            checkpoint_model_type = checkpoint['config'].get('model_type', None)
        
        if checkpoint_model_type and checkpoint_model_type != Config.MODEL_TYPE:
            raise ValueError(f"Model type mismatch: {checkpoint_model_type} vs {Config.MODEL_TYPE}")
        
        model = get_model().to(device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        self.model = model
        self.mcts = MCTS(self.model, device=device)
        self.model_filename = os.path.basename(filepath)

    def new_match(self, target=None):
        """Start a completely new match (resets scores) - matches ws_server.py"""
        if target is not None:
            self.match_target = max(1, min(21, int(target)))
        self.engine.match_scores = {1: 0, -1: 0}
        self.engine.crawford = False
        self.engine.crawford_used = False
        self.engine.reset()
        self.game_over = False
        self.winner = 0
        self.has_rolled = False
        self.selected_index = None
        self.legal_moves = []
        self.redraw()

    def new_game(self):
        """Start a new game (keeps match scores) - matches ws_server.py"""
        self._update_crawford_status()
        self.engine.reset()
        self.game_over = False
        self.winner = 0
        self.has_rolled = False
        self.selected_index = None
        self.legal_moves = []
        self.redraw()

    def _update_crawford_status(self):
        """Update Crawford status based on match scores"""
        score_w = self.engine.match_scores.get(1, 0)
        score_b = self.engine.match_scores.get(-1, 0)
        
        if not self.engine.crawford_used:
            if score_w == self.match_target - 1 or score_b == self.match_target - 1:
                self.engine.crawford = True
            else:
                self.engine.crawford = False
        else:
            self.engine.crawford = False

    def _handle_game_win(self, winner, mult):
        """Handle game win - update match scores (matches ws_server.py)"""
        points = mult * self.engine.cube
        self.engine.match_scores[winner] = self.engine.match_scores.get(winner, 0) + points
        
        if self.engine.crawford:
            self.engine.crawford_used = True
        
        self.game_over = True
        self.winner = winner
        
        winner_name = "White" if winner == 1 else "Black"
        mult_text = {1: "", 2: " (Gammon!)", 3: " (Backgammon!)"}
        
        score_w = self.engine.match_scores.get(1, 0)
        score_b = self.engine.match_scores.get(-1, 0)
        
        match_won = self.engine.match_scores[winner] >= self.match_target
        
        if match_won:
            return f"🏆 {winner_name} WINS THE MATCH{mult_text.get(mult, '')}! Final: {score_w}-{score_b}"
        else:
            return f"{winner_name} wins{mult_text.get(mult, '')} (+{points}pts)! Score: {score_w}-{score_b}"

    def is_human_turn(self):
        """Check if it's a human player's turn"""
        if self.game_mode == 'human_vs_ai':
            return self.engine.turn == 1  # Human plays white
        elif self.game_mode == 'ai_vs_human':
            return self.engine.turn == -1  # Human plays black
        else:  # ai_vs_ai
            return False

    def is_ai_turn(self):
        """Check if it's AI's turn (matches ws_server.py)"""
        if self.game_mode == "human_vs_ai":
            return self.engine.turn == -1
        elif self.game_mode == "ai_vs_human":
            return self.engine.turn == 1
        elif self.game_mode == "ai_vs_ai":
            return True
        return False

    def get_point_geometry(self, index):
        """Calculates position for points 0-23."""
        w, h = self.size
        unit_w = w / 15
        point_h = h * 0.38
        
        # Standard Backgammon layout
        if 0 <= index <= 5:
            col = 13 - index
            x = col * unit_w
            y = 0
            up = True
        elif 6 <= index <= 11:
            col = 6 - (index - 6)
            x = col * unit_w
            y = 0
            up = True
        elif 12 <= index <= 17:
            col = 1 + (index - 12)
            x = col * unit_w
            y = h
            up = False
        else:  # 18-23
            col = 8 + (index - 18)
            x = col * unit_w
            y = h
            up = False
        return x, y, up, unit_w, point_h

    def get_bar_geometry(self, player):
        w, h = self.size
        unit_w = w / 15
        x = 7 * unit_w
        y = h * 0.7 if player == 1 else h * 0.2
        return x, y, unit_w

    def redraw(self, *args):
        self.canvas.clear()
        
        with self.canvas:
            # Board background (green felt)
            Color(*COLOR_FELT)
            Rectangle(pos=self.pos, size=self.size)
            
            # The Bar (wood)
            Color(*COLOR_WOOD_MID)
            Rectangle(pos=(self.width/15 * 7, 0), size=(self.width/15, self.height))
            
            # Border (wood frame)
            Color(*COLOR_WOOD_FRAME)
            Line(rectangle=(self.x, self.y, self.width, self.height), width=6)
            
            Color(*COLOR_WOOD_DARK)
            Line(rectangle=(self.x+3, self.y+3, self.width-6, self.height-6), width=2)

        # Draw points
        for i in range(24):
            x, y, up, w, h = self.get_point_geometry(i)
            
            if i in self.legal_moves and self.selected_index is not None:
                color = COLOR_HIGHLIGHT
            elif self.selected_index == i:
                color = COLOR_SELECTED
            else:
                color = COLOR_POINT_LIGHT if i % 2 == 0 else COLOR_POINT_DARK
                
            with self.canvas:
                Color(*color)
                tip_y = y + h if up else y - h
                Triangle(points=[x, y, x+w, y, x + w/2, tip_y])

        # Draw checkers
        radius = min(self.width/15, self.height/26) * 1.8
        
        for i, count in enumerate(self.engine.board):
            if count == 0:
                continue
            
            player = 1 if count > 0 else -1
            col = COLOR_WHITE_CHECKER if player == 1 else COLOR_BLACK_CHECKER
            
            x, y, up, w, h = self.get_point_geometry(i)
            center_x = x + w/2
            
            for c in range(abs(count)):
                y_offset = c * radius
                if abs(count) > 5:
                    y_offset = c * (radius * 0.8)
                
                cy = (y + radius/2 + y_offset) if up else (y - radius/2 - y_offset)
                
                with self.canvas:
                    Color(*col)
                    Ellipse(pos=(center_x - radius/2, cy - radius/2), size=(radius, radius))
                    # Border
                    Color(0.5, 0.5, 0.5, 0.5)
                    Line(circle=(center_x, cy, radius/2), width=1.2)

        # Draw bar checkers
        for p_idx, count in enumerate(self.engine.bar):
            player = 1 if p_idx == 0 else -1
            if count > 0:
                bx, by, bw = self.get_bar_geometry(player)
                
                if self.selected_index == 'bar' and self.engine.turn == player:
                    with self.canvas:
                        Color(*COLOR_SELECTED)
                        Ellipse(pos=(bx + bw/2 - radius/1.5, by - radius/1.5), 
                               size=(radius*1.5, radius*1.5))

                col = COLOR_WHITE_CHECKER if player == 1 else COLOR_BLACK_CHECKER
                for c in range(count):
                    with self.canvas:
                        Color(*col)
                        Ellipse(pos=(bx + bw/2 - radius/2, by + c*(radius+2)), size=(radius, radius))

        self._draw_cube()
        self._draw_bearoff()

    def _draw_cube(self):
        """Draw the doubling cube on the board."""
        if self.engine.crawford:
            return  # No cube in Crawford game

        cube_size = dp(36)
        w, h = self.size
        x = w - cube_size - dp(10)
        
        if self.engine.cube_owner == 1:
            y = dp(10)
        elif self.engine.cube_owner == -1:
            y = h - cube_size - dp(10)
        else:
            y = h/2 - cube_size/2
        
        with self.canvas:
            Color(*COLOR_CUBE)
            RoundedRectangle(pos=(x, y), size=(cube_size, cube_size), radius=[dp(4)])
            
            Color(*COLOR_WOOD_LIGHT)
            Line(rounded_rectangle=(x, y, cube_size, cube_size, dp(4)), width=2)

    def _draw_bearoff(self):
        """Draw bear-off area with checker counts"""
        w, h = self.size
        unit_w = w / 15
        
        # Bear-off area on the right
        with self.canvas:
            Color(*COLOR_WOOD_MID)
            Rectangle(pos=(14 * unit_w, 0), size=(unit_w, h))
        
        # Draw borne-off checkers count
        radius = dp(8)
        
        # White bear-off (bottom)
        white_off = self.engine.off[0]
        if white_off > 0:
            with self.canvas:
                Color(*COLOR_WHITE_CHECKER)
                for i in range(min(white_off, 15)):
                    Ellipse(pos=(14.2 * unit_w, dp(10) + i * dp(6)), size=(radius, radius))
        
        # Black bear-off (top)
        black_off = self.engine.off[1]
        if black_off > 0:
            with self.canvas:
                Color(*COLOR_BLACK_CHECKER)
                for i in range(min(black_off, 15)):
                    Ellipse(pos=(14.2 * unit_w, h - dp(20) - i * dp(6)), size=(radius, radius))

    def on_touch_down(self, touch):
        if not self.is_human_turn() or self.is_ai_thinking or not self.engine.dice:
            return False
        if not self.collide_point(*touch.pos):
            return False

        tx, ty = touch.pos
        w, h = self.size
        unit_w = w / 15
        
        clicked = None
        
        if 7 * unit_w <= tx <= 8 * unit_w:
            clicked = 'bar'
        elif tx > 14 * unit_w:
            clicked = 'off'
        else:
            col = int(tx / unit_w)
            if col == 7:
                return False
            
            if ty < h/2:
                if col > 7:
                    clicked = 13 - col
                elif col < 7:
                    clicked = 6 + (6 - col)
            else:
                if col < 7:
                    clicked = 12 + (col - 1)
                elif col > 7:
                    clicked = 18 + (col - 8)

        if clicked is not None:
            self.handle_click(clicked)
            return True
        return False

    def handle_click(self, index):
        # 1. Execute move if clicking on a legal destination
        if index in self.legal_moves and self.selected_index is not None:
            winner, mult = self.engine.step_atomic((self.selected_index, index))
            self.selected_index = None
            self.legal_moves = []
            
            if winner != 0:
                status = self._handle_game_win(winner, mult)
                if hasattr(self, 'game_screen'):
                    self.game_screen.update_status(status)
                    self.game_screen.update_score_display()
            
            self.redraw()
            if hasattr(self, 'game_screen'):
                self.game_screen.dice_widget.set_dice(self.engine.dice)
                self.game_screen.update_buttons()
            return

        # 2. Select source
        can_select = False
        if index == 'bar':
            p_idx = 0 if self.engine.turn == 1 else 1
            if self.engine.bar[p_idx] > 0:
                can_select = True
        elif isinstance(index, int) and 0 <= index <= 23:
            count = self.engine.board[index]
            if (self.engine.turn == 1 and count > 0) or (self.engine.turn == -1 and count < 0):
                p_idx = 0 if self.engine.turn == 1 else 1
                if self.engine.bar[p_idx] == 0:
                    can_select = True

        if can_select:
            self.selected_index = index
            all_moves = self.engine.get_legal_moves()
            self.legal_moves = [m[1] for m in all_moves if m[0] == index]
            self.redraw()
        else:
            self.selected_index = None
            self.legal_moves = []
            self.redraw()

    def load_model_thread(self, filepath, callback):
        """Load a model in background thread"""
        def _load():
            try:
                self._load_model_sync(filepath)
                Clock.schedule_once(lambda dt: callback(True, filepath))
            except Exception as e:
                Clock.schedule_once(lambda dt: callback(False, str(e)))
        threading.Thread(target=_load, daemon=True).start()

    def run_ai(self, finished_callback):
        """Run AI for the current player (matches ws_server.py ai_move)"""
        if not self.model or not self.mcts:
            if hasattr(self, 'game_screen'):
                self.game_screen.update_status("⚠️ No model loaded!")
            finished_callback()
            return

        self.is_ai_thinking = True
        
        def _think():
            try:
                # Roll dice if needed
                if not self.engine.dice:
                    self.engine.roll_dice()
                    self.has_rolled = True
                    Clock.schedule_once(lambda dt: self._update_ui_dice())
                    time.sleep(0.3)
                
                # Make moves until dice exhausted
                while self.engine.dice and not self.game_over:
                    legal = self.engine.get_legal_moves()
                    if not legal:
                        break
                    
                    # Use MCTS to find best move
                    root = self.mcts.search(
                        self.engine.copy(),
                        self.engine.match_scores.get(self.engine.turn, 0),
                        self.engine.match_scores.get(-self.engine.turn, 0)
                    )
                    
                    if root.children:
                        best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
                    else:
                        best_move = legal[0]
                    
                    # Apply move
                    winner, mult = self.engine.step_atomic(best_move)
                    
                    Clock.schedule_once(lambda dt: self.redraw())
                    Clock.schedule_once(lambda dt: self._update_ui_dice())
                    
                    if winner != 0:
                        status = self._handle_game_win(winner, mult)
                        # Call finished_callback even on win so autoplay continues
                        Clock.schedule_once(lambda dt, s=status: self._on_ai_win(s, finished_callback))
                        return
                    
                    time.sleep(0.2)
                
                # End AI turn
                self.engine.dice = []
                self.engine.switch_turn()
                self.has_rolled = False
                
                Clock.schedule_once(lambda dt: finished_callback())
            finally:
                self.is_ai_thinking = False

        threading.Thread(target=_think, daemon=True).start()

    def _update_ui_dice(self):
        if hasattr(self, 'game_screen'):
            self.game_screen.dice_widget.set_dice(self.engine.dice)

    def _on_ai_win(self, status, finished_callback):
        if hasattr(self, 'game_screen'):
            self.game_screen.update_status(status)
            self.game_screen.update_score_display()
            self.game_screen.update_buttons()
        # Call the callback so autoplay can continue
        finished_callback()


class GameScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(5)
        self.padding = dp(5)

        # --- TOP CONTROL BAR ---
        controls = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(5))
        
        btn_new_match = Button(text="New Match", size_hint_x=None, width=dp(100),
                               background_color=COLOR_WOOD_DARK[:3] + (1,))
        btn_new_match.bind(on_press=self.show_new_match_dialog)
        
        btn_new_game = Button(text="New Game", size_hint_x=None, width=dp(90),
                              background_color=COLOR_WOOD_MID[:3] + (1,))
        btn_new_game.bind(on_press=lambda x: self.start_new_game())
        
        btn_load = Button(text="Load Model", size_hint_x=None, width=dp(100),
                         background_color=(0.2, 0.4, 0.4, 1))
        btn_load.bind(on_press=self.show_load_dialog)
        
        # Game mode selector (3 options like ws_server.py)
        self.mode_spinner = Spinner(
            text='human_vs_ai',
            values=('human_vs_ai', 'ai_vs_human', 'ai_vs_ai'),
            size_hint_x=None, width=dp(130)
        )
        self.mode_spinner.bind(text=self.on_mode_change)
        
        # Auto-play controls (for ai_vs_ai)
        self.btn_auto = Button(text="▶ Auto", size_hint_x=None, width=dp(80),
                               background_color=(0.3, 0.3, 0.3, 1))
        self.btn_auto.bind(on_press=self.toggle_auto_play)
        
        self.speed_spinner = Spinner(text='1.0s', values=('0.2s', '0.5s', '1.0s', '2.0s'),
                                     size_hint_x=None, width=dp(70))
        self.speed_spinner.bind(text=self.on_speed_change)
        
        self.btn_step = Button(text="Step", size_hint_x=None, width=dp(60))
        self.btn_step.bind(on_press=lambda x: self.step_ai())
        
        controls.add_widget(btn_new_match)
        controls.add_widget(btn_new_game)
        controls.add_widget(btn_load)
        controls.add_widget(self.mode_spinner)
        controls.add_widget(self.btn_auto)
        controls.add_widget(self.speed_spinner)
        controls.add_widget(self.btn_step)
        self.add_widget(controls)

        # --- SCORE & INFO BAR ---
        info_bar = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        
        # Match score
        self.lbl_score = Label(text="White: 0 | Black: 0 | Match to 7",
                               font_size='14sp', bold=True, size_hint_x=0.4)
        info_bar.add_widget(self.lbl_score)
        
        # Cube display
        cube_box = BoxLayout(size_hint_x=None, width=dp(100), orientation='vertical')
        self.lbl_cube = Label(text="Cube: 1", font_size='12sp', bold=True,
                              color=COLOR_WOOD_FRAME[:3] + (1,))
        self.lbl_cube_owner = Label(text="(Center)", font_size='10sp',
                                    color=(0.7, 0.7, 0.7, 1))
        cube_box.add_widget(self.lbl_cube)
        cube_box.add_widget(self.lbl_cube_owner)
        info_bar.add_widget(cube_box)
        
        # Dice
        self.dice_widget = DiceWidget(size_hint=(0.4, 1))
        info_bar.add_widget(self.dice_widget)
        
        # Model status
        self.lbl_model = Label(text="No model", font_size='10sp',
                               color=(0.8, 0.4, 0.4, 1), size_hint_x=0.2)
        info_bar.add_widget(self.lbl_model)
        
        self.add_widget(info_bar)

        # --- STATUS LABEL ---
        self.lbl_status = Label(text="Welcome! Start a new match.",
                                size_hint_y=None, height=dp(30),
                                color=COLOR_POINT_LIGHT[:3] + (1,))
        self.add_widget(self.lbl_status)

        # --- ACTION BUTTONS ---
        action_bar = BoxLayout(size_hint_y=None, height=dp(45), spacing=dp(5))
        
        self.btn_double = Button(text="Double", background_color=(0.6, 0.4, 0.2, 1))
        self.btn_double.bind(on_press=self.human_offers_double)
        
        self.btn_roll = Button(text="Roll Dice", background_color=(0.2, 0.5, 0.3, 1))
        self.btn_roll.bind(on_press=self.roll_dice)
        
        self.btn_end = Button(text="End Turn", background_color=(0.2, 0.4, 0.6, 1),
                              disabled=True)
        self.btn_end.bind(on_press=self.end_turn)

        action_bar.add_widget(self.btn_double)
        action_bar.add_widget(self.btn_roll)
        action_bar.add_widget(self.btn_end)
        self.add_widget(action_bar)

        # --- BOARD ---
        board_container = FloatLayout()
        self.board = BackgammonBoard()
        self.board.game_screen = self
        board_container.add_widget(self.board)
        self.add_widget(board_container)
        
        self.update_buttons()
        self.update_model_status()

    def on_mode_change(self, spinner, text):
        self.board.game_mode = text
        self.update_status(f"Mode: {text}")
        self.update_buttons()
        
        # If AI starts, trigger
        if self.board.is_ai_turn() and not self.board.game_over and not self.board.engine.dice:
            self.trigger_ai()

    def toggle_auto_play(self, instance):
        if self.board.game_mode != 'ai_vs_ai':
            self.update_status("Auto-play only in AI vs AI mode")
            return
        
        self.board.ai_auto_play = not self.board.ai_auto_play
        self.btn_auto.text = "⏸ Stop" if self.board.ai_auto_play else "▶ Auto"
        
        if self.board.ai_auto_play:
            self.update_status("Autoplay started")
            # Trigger first AI move
            if not self.board.is_ai_thinking and not self.board.game_over:
                self.trigger_ai()
        else:
            self.update_status("Autoplay stopped")

    def on_speed_change(self, spinner, text):
        try:
            self.board.ai_speed = float(text.replace('s', ''))
        except:
            pass

    def step_ai(self):
        if self.board.game_mode != 'ai_vs_ai':
            self.update_status("Step only in AI vs AI mode")
            return
        if self.board.is_ai_thinking:
            return
        self.trigger_ai()

    def trigger_ai(self):
        if self.board.game_over or not self.board.is_ai_turn():
            return
        
        self.update_status("AI thinking...")
        self.update_buttons()
        self.board.run_ai(self.on_ai_finished)

    def on_ai_finished(self):
        self.board.is_ai_thinking = False
        
        # Check if game ended during AI turn
        if self.board.game_over:
            self.update_score_display()
            self.update_buttons()
            self.dice_widget.set_dice([])
            self.board.redraw()
            
            # If autoplay is on and match not won, schedule next game
            if self.board.ai_auto_play:
                match_won = (
                    self.board.engine.match_scores.get(1, 0) >= self.board.match_target or
                    self.board.engine.match_scores.get(-1, 0) >= self.board.match_target
                )
                if not match_won:
                    Clock.schedule_once(lambda dt: self.start_new_game(), 0.5)
                else:
                    self.board.ai_auto_play = False
                    self.btn_auto.text = "▶ Auto"
            return
        
        turn_name = "White" if self.board.engine.turn == 1 else "Black"
        self.update_status(f"{turn_name}'s turn")
        self.dice_widget.set_dice([])
        self.board.redraw()
        self.update_buttons()
        
        # If AI vs AI and autoplay, trigger next AI immediately
        if self.board.game_mode == 'ai_vs_ai' and self.board.ai_auto_play:
            Clock.schedule_once(lambda dt: self.trigger_ai(), 0.1)

    def show_new_match_dialog(self, instance):
        content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        content.add_widget(Label(text="Match Length (Points):", size_hint_y=None, height=dp(30)))
        
        spinner = Spinner(text='7', values=('1', '3', '5', '7', '9', '11', '15', '21'),
                         size_hint_y=None, height=dp(40))
        content.add_widget(spinner)
        
        btn_start = Button(text="Start Match", size_hint_y=None, height=dp(50))
        content.add_widget(btn_start)
        
        popup = Popup(title="New Match", content=content, size_hint=(0.5, 0.4))
        
        def start(x):
            target = int(spinner.text)
            self.board.new_match(target)
            self.update_score_display()
            self.update_cube_display()
            self.dice_widget.set_dice([])
            self.update_status(f"New match! First to {target} points.")
            self.update_buttons()
            popup.dismiss()
            
            if self.board.is_ai_turn():
                Clock.schedule_once(lambda dt: self.trigger_ai(), 0.3)
        
        btn_start.bind(on_press=start)
        popup.open()

    def start_new_game(self):
        self.board.new_game()
        self.update_cube_display()
        self.dice_widget.set_dice([])
        
        crawford_msg = " (Crawford!)" if self.board.engine.crawford else ""
        self.update_status(f"New game started!{crawford_msg}")
        self.update_buttons()
        
        if self.board.is_ai_turn():
            Clock.schedule_once(lambda dt: self.trigger_ai(), 0.3)

    def update_score_display(self):
        w = self.board.engine.match_scores.get(1, 0)
        b = self.board.engine.match_scores.get(-1, 0)
        t = self.board.match_target
        self.lbl_score.text = f"White: {w} | Black: {b} | Match to {t}"

    def update_cube_display(self):
        if self.board.engine.crawford:
            self.lbl_cube.text = "CRAWFORD"
            self.lbl_cube_owner.text = "No Cube"
            return
        
        self.lbl_cube.text = f"Cube: {self.board.engine.cube}"
        if self.board.engine.cube_owner == 0:
            self.lbl_cube_owner.text = "(Center)"
        elif self.board.engine.cube_owner == 1:
            self.lbl_cube_owner.text = "(White)"
        else:
            self.lbl_cube_owner.text = "(Black)"

    def update_model_status(self):
        if self.board.model_filename:
            self.lbl_model.text = self.board.model_filename
            self.lbl_model.color = (0.4, 0.7, 0.5, 1)
        else:
            self.lbl_model.text = "No model"
            self.lbl_model.color = (0.8, 0.4, 0.4, 1)

    def update_buttons(self):
        if self.board.game_over or self.board.is_ai_thinking:
            self.btn_double.disabled = True
            self.btn_roll.disabled = True
            self.btn_end.disabled = True
            return
        
        is_human = self.board.is_human_turn()
        has_dice = bool(self.board.engine.dice)
        has_moves = bool(self.board.engine.get_legal_moves()) if has_dice else False
        
        # Double: can double before rolling, if allowed
        can_double = (
            is_human and
            not has_dice and
            not self.board.has_rolled and
            self.board.engine.can_double()
        )
        self.btn_double.disabled = not can_double
        
        # Roll: can roll if human turn and no dice yet
        self.btn_roll.disabled = not is_human or has_dice
        
        # End turn: can end if rolled and (no dice left or no legal moves)
        can_end = (
            is_human and
            self.board.has_rolled and
            (not has_dice or not has_moves)
        )
        self.btn_end.disabled = not can_end

    def roll_dice(self, instance):
        if self.board.game_over or self.board.is_ai_thinking:
            return
        if not self.board.is_human_turn():
            return
        if self.board.engine.dice:
            return
        
        self.board.engine.roll_dice()
        self.board.has_rolled = True
        
        self.dice_widget.set_dice(self.board.engine.dice)
        
        legal = self.board.engine.get_legal_moves()
        if not legal:
            self.update_status(f"Rolled {self.board.engine.dice} - No legal moves!")
        else:
            self.update_status(f"Rolled {self.board.engine.dice}")
        
        self.board.redraw()
        self.update_buttons()

    def end_turn(self, instance):
        if self.board.game_over:
            return
        if not self.board.has_rolled:
            return
        
        self.board.engine.dice = []
        self.board.engine.switch_turn()
        self.board.has_rolled = False
        self.board.selected_index = None
        self.board.legal_moves = []
        
        self.dice_widget.set_dice([])
        turn_name = "White" if self.board.engine.turn == 1 else "Black"
        self.update_status(f"{turn_name}'s turn")
        self.board.redraw()
        self.update_buttons()
        
        # Trigger AI if needed
        if self.board.is_ai_turn():
            Clock.schedule_once(lambda dt: self.trigger_ai(), 0.3)

    def human_offers_double(self, instance):
        if self.board.engine.crawford:
            return
        if not self.board.engine.can_double():
            return
        
        # For simplicity, AI always accepts
        self.board.engine.apply_double()
        self.update_cube_display()
        self.update_status("AI accepted double!")
        self.update_buttons()

    def show_load_dialog(self, instance):
        content = BoxLayout(orientation='vertical')
        
        if os.path.exists(CHECKPOINTS_DIR):
            path = CHECKPOINTS_DIR
        else:
            path = os.getcwd()
        
        file_chooser = FileChooserListView(path=path, filters=['*.pt', '*.pth'])
        content.add_widget(file_chooser)
        
        btns = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        btn_cancel = Button(text="Cancel")
        btn_load = Button(text="Load")
        btns.add_widget(btn_cancel)
        btns.add_widget(btn_load)
        content.add_widget(btns)
        
        popup = Popup(title="Load Model", content=content, size_hint=(0.9, 0.9))
        btn_cancel.bind(on_press=popup.dismiss)
        
        def load_selection(x):
            if file_chooser.selection:
                filepath = file_chooser.selection[0]
                self.update_status("Loading model...")
                self.board.load_model_thread(filepath, self.on_model_loaded)
                popup.dismiss()
        
        btn_load.bind(on_press=load_selection)
        popup.open()

    def on_model_loaded(self, success, msg):
        if success:
            self.update_status(f"Model loaded: {os.path.basename(msg)}")
            self.update_model_status()
        else:
            self.update_status(f"Load failed: {msg}")

    def update_status(self, text=None):
        if text:
            self.lbl_status.text = text
        else:
            p = "White" if self.board.engine.turn == 1 else "Black"
            self.lbl_status.text = f"{p}'s Turn"


class BackgammonApp(App):
    def build(self):
        Window.clearcolor = COLOR_FELT_DARK
        Window.size = (1100, 750)
        return GameScreen()


if __name__ == '__main__':
    BackgammonApp().run()


import threading
import os
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.graphics import Color, Line, Triangle, Ellipse, Rectangle, RoundedRectangle
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp

import numpy as np
import torch
from bg_engine import BackgammonGame
from model import BackgammonTransformer
from mcts import MCTS
from config import Config
from checkpoint import load_checkpoint

# --- Visual Constants ---
COLOR_BG = (0.15, 0.15, 0.15, 1)
COLOR_BOARD = (0.35, 0.25, 0.15, 1)
COLOR_POINT_1 = (0.6, 0.4, 0.2, 1)      # Light Brown
COLOR_POINT_2 = (0.25, 0.15, 0.1, 1)    # Dark Brown
COLOR_P1 = (0.95, 0.95, 0.95, 1)        # White Checkers
COLOR_P2 = (0.15, 0.15, 0.15, 1)        # Black Checkers
COLOR_HIGHLIGHT = (0.2, 0.9, 0.2, 0.5)  # Green selection
COLOR_DICE_BG = (0.9, 0.9, 0.9, 1)
COLOR_DICE_DOT = (0.1, 0.1, 0.1, 1)

class DiceWidget(Widget):
    """Draws graphical dice faces based on a list of values."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dice_values = []
        self.bind(pos=self.redraw, size=self.redraw)

    def set_dice(self, values):
        self.dice_values = values
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
                # Die Background
                Color(*COLOR_DICE_BG)
                RoundedRectangle(pos=(x, y), size=(die_size, die_size), radius=[dp(5)])
                
                # Dots logic
                Color(*COLOR_DICE_DOT)
                dot_size = die_size / 5
                
                # Helper positions (0..1 scale)
                l, c, r = 0.25, 0.5, 0.75
                
                # Coordinates mapping
                coords = []
                if val == 1: coords = [(c, c)]
                elif val == 2: coords = [(l, r), (r, l)]
                elif val == 3: coords = [(l, r), (c, c), (r, l)]
                elif val == 4: coords = [(l, l), (l, r), (r, l), (r, r)]
                elif val == 5: coords = [(l, l), (l, r), (c, c), (r, l), (r, r)]
                elif val == 6: coords = [(l, l), (l, c), (l, r), (r, l), (r, c), (r, r)]
                
                for (cx, cy) in coords:
                    # Invert Y because 0 is bottom
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
        self.ai_model = None
        self.ai_mcts = None
        self.is_ai_thinking = False
        
        self.bind(pos=self.redraw, size=self.redraw)
        self.engine.reset()

    def get_point_geometry(self, index):
        """Calculates position for points 0-23."""
        w, h = self.size
        # Layout: 13 columns (6 left points + bar + 6 right points)
        # However, standard visual is Bar in middle.
        # We divide width by 15 units: 6 (home) + 1 (bar) + 6 (outer) + 2 (off/borders)
        unit_w = w / 15
        point_h = h * 0.38
        
        # Visual Mapping (0 is Bottom Right P1 Home)
        # 0-5: Bottom Right (Right to Left)
        # 6-11: Bottom Left (Right to Left)
        # 12-17: Top Left (Left to Right)
        # 18-23: Top Right (Left to Right)
        
        if 0 <= index <= 5:   # Bottom Right
            col = 13 - index  # 13 down to 8
            x = col * unit_w
            y = 0
            up = True
        elif 6 <= index <= 11: # Bottom Left
            col = 6 - (index - 6) # 6 down to 1
            x = col * unit_w
            y = 0
            up = True
        elif 12 <= index <= 17: # Top Left
            col = 1 + (index - 12) # 1 up to 6
            x = col * unit_w
            y = h
            up = False
        else: # 18-23 Top Right
            col = 8 + (index - 18) # 8 up to 13
            x = col * unit_w
            y = h
            up = False
            
        return x, y, up, unit_w, point_h

    def get_bar_geometry(self, player):
        w, h = self.size
        unit_w = w / 15
        x = 7 * unit_w # Center
        y = h * 0.7 if player == 1 else h * 0.2
        return x, y, unit_w

    def redraw(self, *args):
        self.canvas.clear()
        
        # 1. Background & Frame
        with self.canvas:
            Color(*COLOR_BOARD)
            Rectangle(pos=self.pos, size=self.size)
            
            # Bar (Center Strip)
            Color(0.2, 0.15, 0.1, 1)
            Rectangle(pos=(self.width/15 * 7, 0), size=(self.width/15, self.height))
            
            # Borders
            Color(0.2, 0.15, 0.1, 1)
            Line(rectangle=(self.x, self.y, self.width, self.height), width=2)

        # 2. Points
        for i in range(24):
            x, y, up, w, h = self.get_point_geometry(i)
            
            # Selection Highlight
            if i in self.legal_moves and self.selected_index is not None:
                color = COLOR_HIGHLIGHT
            else:
                color = COLOR_POINT_1 if i % 2 == 0 else COLOR_POINT_2
                
            with self.canvas:
                Color(*color)
                tip_y = y + h if up else y - h
                Triangle(points=[x, y, x+w, y, x + w/2, tip_y])

        # 3. Checkers (Board)
        radius = min(self.width/15, self.height/26) * 0.85
        
        for i, count in enumerate(self.engine.board):
            if count == 0: continue
            
            player = 1 if count > 0 else -1
            col = COLOR_P1 if player == 1 else COLOR_P2
            
            x, y, up, w, h = self.get_point_geometry(i)
            center_x = x + w/2
            
            # Stack Logic
            for c in range(abs(count)):
                # Compress stack if too tall
                y_offset = c * radius
                if abs(count) > 5:
                    y_offset = c * (radius * 0.8) # Overlap
                
                cy = (y + radius/2 + y_offset) if up else (y - radius/2 - y_offset)
                
                with self.canvas:
                    Color(*col)
                    Ellipse(pos=(center_x - radius/2, cy - radius/2), size=(radius, radius))
                    # Outline
                    Color(0,0,0,0.5)
                    Line(circle=(center_x, cy, radius/2), width=1.1)

        # 4. Checkers (Bar)
        for p_idx, count in enumerate(self.engine.bar):
            player = 1 if p_idx == 0 else -1
            if count > 0:
                bx, by, bw = self.get_bar_geometry(player)
                
                # Highlight bar if selected
                if self.selected_index == 'bar' and self.engine.turn == player:
                    with self.canvas:
                        Color(*COLOR_HIGHLIGHT)
                        Ellipse(pos=(bx + bw/2 - radius/1.5, by - radius/1.5), size=(radius*1.5, radius*1.5))

                col = COLOR_P1 if player == 1 else COLOR_P2
                for c in range(count):
                    with self.canvas:
                        Color(*col)
                        Ellipse(pos=(bx + bw/2 - radius/2, by + c*5), size=(radius, radius))

    def on_touch_down(self, touch):
        if self.is_ai_thinking or not self.engine.dice: return False
        if not self.collide_point(*touch.pos): return False

        tx, ty = touch.pos
        w, h = self.size
        unit_w = w / 15
        
        # Click Detection Logic
        clicked = None
        
        # Check Bar (Middle Column 7)
        if 7 * unit_w <= tx <= 8 * unit_w:
            clicked = 'bar'
        elif tx > 14 * unit_w:
             clicked = 'off' # Right edge
        else:
            # Points
            col = int(tx / unit_w)
            if col == 7: return False
            
            # Map visual column to logical index
            if ty < h/2: # Bottom
                # Right side (8-13) -> 5-0
                # Left side (1-6) -> 11-6
                if col > 7: clicked = 13 - (col - 7 + 7) + 5 # Simplified: 13-col
                elif col < 7: clicked = 6 + (6 - col)
            else: # Top
                # Left side (1-6) -> 12-17
                # Right side (8-13) -> 18-23
                if col < 7: clicked = 12 + (col - 1)
                elif col > 7: clicked = 18 + (col - 8)

        if clicked is not None:
            self.handle_click(clicked)
            return True
        return False

    def handle_click(self, index):
        # 1. Execute Move
        if index in self.legal_moves and self.selected_index is not None:
            self.engine.step_atomic((self.selected_index, index))
            self.selected_index = None
            self.legal_moves = []
            
            # Check Win
            w, _ = self.engine.check_win()
            if w != 0:
                self.parent.parent.update_status(f"WINNER: {'White' if w==1 else 'Black'}!")
            
            # Update UI
            self.redraw()
            self.parent.parent.dice_widget.set_dice(self.engine.dice)
            return

        # 2. Select Source
        # Can select if: Own checker on Bar OR Own checker on Point
        can_select = False
        if index == 'bar':
            p_idx = 0 if self.engine.turn == 1 else 1
            if self.engine.bar[p_idx] > 0: can_select = True
        elif isinstance(index, int) and 0 <= index <= 23:
            count = self.engine.board[index]
            if (self.engine.turn == 1 and count > 0) or (self.engine.turn == -1 and count < 0):
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
        def _load():
            try:
                model = BackgammonTransformer().to(Config.DEVICE)
                load_checkpoint(filepath, model)
                model.eval()
                self.ai_model = model
                self.ai_mcts = MCTS(model)
                Clock.schedule_once(lambda dt: callback(True, filepath))
            except Exception as e:
                print(e)
                Clock.schedule_once(lambda dt: callback(False, str(e)))
        threading.Thread(target=_load, daemon=True).start()

    def run_ai(self, finished_callback):
        if not self.ai_model: return
        self.is_ai_thinking = True
        
        def _think():
            if not self.engine.dice:
                self.engine.roll_dice()
                Clock.schedule_once(lambda dt: self.parent.parent.dice_widget.set_dice(self.engine.dice))

            while self.engine.dice:
                moves = self.engine.get_legal_moves()
                if not moves: break
                
                # MCTS
                root = self.ai_mcts.search(None, self.engine, 0, 0)
                visits = {act: child.visits for act, child in root.children.items()}
                
                if visits:
                    action = max(visits, key=visits.get)
                else:
                    action = moves[0]
                    
                self.engine.step_atomic(action)
                
                # Update UI safely
                Clock.schedule_once(lambda dt: self.redraw())
                Clock.schedule_once(lambda dt: self.parent.parent.dice_widget.set_dice(self.engine.dice))
                
                if self.engine.check_win()[0] != 0: break
                import time; time.sleep(0.4) 

            # End turn
            if self.engine.check_win()[0] == 0:
                self.engine.switch_turn()
            
            Clock.schedule_once(lambda dt: finished_callback())

        threading.Thread(target=_think, daemon=True).start()


class GameScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(5)
        
        # --- 1. TOP CONTROL BAR ---
        controls = BoxLayout(size_hint_y=None, height=dp(50), padding=dp(5), spacing=dp(10))
        
        # Load Model Button
        btn_load = Button(text="Load AI", size_hint_x=None, width=dp(100), background_color=(0.2, 0.5, 0.5, 1))
        btn_load.bind(on_press=self.show_load_dialog)
        
        # Reset
        btn_reset = Button(text="New Game", size_hint_x=None, width=dp(100), background_color=(0.5, 0.2, 0.2, 1))
        btn_reset.bind(on_press=self.reset_game)

        # Status Label
        self.lbl_status = Label(text="P1 (White) Turn", font_size='18sp', bold=True, color=(1,1,1,1))
        
        # Actions
        self.btn_roll = Button(text="Roll Dice", background_color=(0.2, 0.6, 0.2, 1))
        self.btn_roll.bind(on_press=self.roll_dice)
        
        self.btn_ai = Button(text="AI Move", background_color=(0.7, 0.3, 0.7, 1), disabled=True)
        self.btn_ai.bind(on_press=self.trigger_ai)

        controls.add_widget(btn_load)
        controls.add_widget(btn_reset)
        controls.add_widget(self.lbl_status)
        controls.add_widget(self.btn_ai)
        controls.add_widget(self.btn_roll)
        
        self.add_widget(controls)

        # --- 2. DICE DISPLAY AREA ---
        dice_area = BoxLayout(size_hint_y=None, height=dp(60))
        self.dice_widget = DiceWidget(size_hint=(1, 1))
        dice_area.add_widget(self.dice_widget)
        self.add_widget(dice_area)

        # --- 3. BOARD AREA ---
        board_container = FloatLayout()
        self.board = BackgammonBoard()
        board_container.add_widget(self.board)
        self.add_widget(board_container)

    def roll_dice(self, instance):
        if self.board.is_ai_thinking: return
        if self.board.engine.dice: return # Already rolled
        
        d = self.board.engine.roll_dice()
        self.dice_widget.set_dice(d)
        self.update_status()
        self.board.redraw()
        
        # Auto-pass if blocked
        if not self.board.engine.get_legal_moves():
            Clock.schedule_once(lambda dt: self.pass_turn(), 1.5)

    def pass_turn(self):
        self.update_status("No moves! Switching...")
        Clock.schedule_once(lambda dt: self._switch(), 1.0)

    def _switch(self):
        self.board.engine.switch_turn()
        self.board.engine.dice = []
        self.dice_widget.set_dice([])
        self.update_status()

    def trigger_ai(self, instance):
        self.btn_roll.disabled = True
        self.btn_ai.disabled = True
        self.update_status("AI Thinking...")
        self.board.run_ai(self.ai_done)

    def ai_done(self):
        self.board.is_ai_thinking = False
        self.dice_widget.set_dice([])
        self.update_status()
        self.btn_roll.disabled = False
        self.btn_ai.disabled = False

    def reset_game(self, instance):
        self.board.engine.reset()
        self.board.selected_index = None
        self.board.legal_moves = []
        self.dice_widget.set_dice([])
        self.update_status("New Game Started")
        self.board.redraw()

    def update_status(self, text=None):
        if text:
            self.lbl_status.text = text
            return
        
        p = "White (P1)" if self.board.engine.turn == 1 else "Black (P-1)"
        self.lbl_status.text = f"{p} Turn"

    # --- FILE LOADING LOGIC ---
    def show_load_dialog(self, instance):
        content = BoxLayout(orientation='vertical')
        
        # File Chooser
        path = os.getcwd()
        file_chooser = FileChooserListView(path=path, filters=['*.pt'])
        content.add_widget(file_chooser)
        
        # Buttons
        btns = BoxLayout(size_hint_y=None, height=dp(50), spacing=dp(10))
        btn_cancel = Button(text="Cancel")
        btn_load = Button(text="Load")
        
        btns.add_widget(btn_cancel)
        btns.add_widget(btn_load)
        content.add_widget(btns)
        
        popup = Popup(title="Load Model Checkpoint", content=content, size_hint=(0.9, 0.9))
        
        btn_cancel.bind(on_press=popup.dismiss)
        
        def load_selection(x):
            if file_chooser.selection:
                f = file_chooser.selection[0]
                self.update_status("Loading Model...")
                self.board.load_model_thread(f, self.on_model_loaded)
                popup.dismiss()
        
        btn_load.bind(on_press=load_selection)
        popup.open()

    def on_model_loaded(self, success, msg):
        if success:
            filename = os.path.basename(msg)
            self.update_status(f"Loaded: {filename}")
            self.btn_ai.disabled = False
        else:
            self.update_status("Load Failed")
            print(msg)


class BackgammonApp(App):
    def build(self):
        Window.clearcolor = COLOR_BG
        Window.size = (1024, 768)
        return GameScreen()

if __name__ == '__main__':
    BackgammonApp().run()
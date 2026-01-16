import threading
import os
import copy
import time
import random
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

# --- ASSUMED EXTERNAL IMPORTS ---
try:
    from bg_engine import BackgammonGame
    from model import get_model
    from mcts import MCTS
    from config import Config
    from checkpoint import load_checkpoint
except ImportError:
    print("Notice: External AI modules not found. Using internal Mock Engine for UI testing.")
    
    # Mock Engine for UI testing purposes
    class BackgammonGame:
        def __init__(self): 
            self.board = [0]*24
            # Setup a board (Standard setup)
            self.board[0] = 2; self.board[11] = 5; self.board[16] = 3; self.board[18] = 5
            self.board[23] = -2; self.board[12] = -5; self.board[7] = -3; self.board[5] = -5
            self.bar = [0,0]
            self.off = [0,0]
            self.dice = []
            self.turn = 1 # 1 = White, -1 = Black
            
        def reset(self): 
            self.__init__()

        def roll_dice(self): 
            self.dice = [random.randint(1,6), random.randint(1,6)]
            if self.dice[0] == self.dice[1]: self.dice *= 2
            return self.dice

        def get_legal_moves(self):
            # Mock: return a dummy move if dice exist
            if not self.dice: return []
            # Find a piece to move
            moves = []
            direction = -1 if self.turn == 1 else 1 # Visual mapping is tricky, simplifying mock
            for i, c in enumerate(self.board):
                if (self.turn == 1 and c > 0) or (self.turn == -1 and c < 0):
                    target = i + self.dice[0] * (-1 if self.turn==1 else 1) # Simple logic
                    if 0 <= target <= 23:
                        moves.append((i, target))
            return moves

        def step_atomic(self, action):
            # Very basic move execution for mock
            src, dst = action
            self.board[src] -= (1 if self.turn == 1 else -1)
            self.board[dst] += (1 if self.turn == 1 else -1)
            if self.dice: self.dice.pop(0)

        def check_win(self): 
            # Mock win condition: random for testing UI flow if checkers are low
            # Returns: (winner_id, multiplier)
            # 0 = No win, 1 = P1, -1 = P2
            # Multiplier: 1 (Normal), 2 (Gammon), 3 (Backgammon)
            
            # Simple check: if someone bore off 15 (mock logic: just check if board empty-ish)
            # For testing, let's just say if off count > 0 (this is fake logic)
            if self.off[0] >= 15: return 1, 1
            if self.off[1] >= 15: return -1, 1
            return 0, 1
            
        def copy(self): return copy.deepcopy(self)
    
    class Config:
        DEVICE = 'cpu'
    
    def get_model(): return None
    def load_checkpoint(*args, **kwargs): pass
    class MCTS:
        def __init__(self, *args): pass
        def search(self, *args): return None


# --- VISUAL CONSTANTS ---
COLOR_BG = (0.15, 0.15, 0.15, 1)
COLOR_BOARD = (0.35, 0.25, 0.15, 1)
COLOR_POINT_1 = (0.6, 0.4, 0.2, 1)
COLOR_POINT_2 = (0.25, 0.15, 0.1, 1)
COLOR_P1 = (0.95, 0.95, 0.95, 1)
COLOR_P2 = (0.15, 0.15, 0.15, 1)
COLOR_HIGHLIGHT = (0.2, 0.9, 0.2, 0.5)
COLOR_DICE_BG = (0.9, 0.9, 0.9, 1)
COLOR_DICE_DOT = (0.1, 0.1, 0.1, 1)
COLOR_CUBE = (0.8, 0.2, 0.2, 1)


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
        self.ai_model = None
        self.ai_device = None
        self.ai_mcts = None
        self.is_ai_thinking = False
        
        # Game Settings
        self.game_mode = 'human_vs_ai'
        
        # Match State
        self.match_target = 3  # Default match length
        self.match_score = {1: 0, -1: 0} # 1: White, -1: Black
        self.crawford_active = False # Is this specific game the Crawford game?
        self.crawford_triggered = False # Has the Crawford game already happened in this match?

        # Doubling Cube
        self.cube_value = 1
        self.cube_owner = 0  # 0 = Center, 1 = P1, -1 = P2
        
        # Turn state for take-back
        self.turn_backup = None
        self.has_moved = False
        
        self.bind(pos=self.redraw, size=self.redraw)
        self.engine.reset()

    def reset_match(self, target=None):
        """Resets the entire match."""
        if target:
            self.match_target = target
        self.match_score = {1: 0, -1: 0}
        self.crawford_triggered = False
        self.reset_game()

    def reset_game(self):
        """Resets for the next game in the match."""
        self.engine.reset()
        self.cube_value = 1
        self.cube_owner = 0
        self.turn_backup = None
        self.has_moved = False
        self.selected_index = None
        self.legal_moves = []
        
        # Check Crawford Rule
        # If either player is n-1 points away from match, and we haven't played Crawford yet
        p1_needs = self.match_target - self.match_score[1]
        p2_needs = self.match_target - self.match_score[-1]
        
        if (p1_needs == 1 or p2_needs == 1) and not self.crawford_triggered:
            self.crawford_active = True
            self.crawford_triggered = True # It happens once per match
        else:
            self.crawford_active = False
            
        self.redraw()

    def save_turn_state(self):
        """Save current state for potential take-back."""
        self.turn_backup = {
            'board': list(self.engine.board),
            'bar': list(self.engine.bar),
            'off': list(self.engine.off),
            'dice': list(self.engine.dice) if self.engine.dice else [],
            'turn': self.engine.turn
        }
        self.has_moved = False

    def restore_turn_state(self):
        """Restore state from backup (take-back)."""
        if self.turn_backup:
            self.engine.board = list(self.turn_backup['board'])
            self.engine.bar = list(self.turn_backup['bar'])
            self.engine.off = list(self.turn_backup['off'])
            self.engine.dice = list(self.turn_backup['dice'])
            self.engine.turn = self.turn_backup['turn']
            self.has_moved = False
            self.selected_index = None
            self.legal_moves = []
            self.redraw()
            return True
        return False

    def is_human_turn(self):
        if self.game_mode == 'human_vs_ai':
            return self.engine.turn == 1
        elif self.game_mode == 'ai_vs_human':
            return self.engine.turn == -1
        else:  # ai_vs_ai
            return False

    def get_point_geometry(self, index):
        """Calculates position for points 0-23."""
        w, h = self.size
        unit_w = w / 15
        point_h = h * 0.38
        
        # Standard Backgammon layout (White Home is 0-5 bottom right)
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
        else: # 18-23
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
            Color(*COLOR_BOARD)
            Rectangle(pos=self.pos, size=self.size)
            
            # The Bar
            Color(0.2, 0.15, 0.1, 1)
            Rectangle(pos=(self.width/15 * 7, 0), size=(self.width/15, self.height))
            
            # Border
            Color(0.2, 0.15, 0.1, 1)
            Line(rectangle=(self.x, self.y, self.width, self.height), width=2)

        # Draw points
        for i in range(24):
            x, y, up, w, h = self.get_point_geometry(i)
            
            if i in self.legal_moves and self.selected_index is not None:
                color = COLOR_HIGHLIGHT
            else:
                color = COLOR_POINT_1 if i % 2 == 0 else COLOR_POINT_2
                
            with self.canvas:
                Color(*color)
                tip_y = y + h if up else y - h
                Triangle(points=[x, y, x+w, y, x + w/2, tip_y])

        # Draw checkers
        radius = min(self.width/15, self.height/26) * 0.9
        
        for i, count in enumerate(self.engine.board):
            if count == 0: continue
            
            player = 1 if count > 0 else -1
            col = COLOR_P1 if player == 1 else COLOR_P2
            
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
                    Color(0, 0, 0, 0.5)
                    Line(circle=(center_x, cy, radius/2), width=1.1)

        # Draw bar checkers
        for p_idx, count in enumerate(self.engine.bar):
            player = 1 if p_idx == 0 else -1
            if count > 0:
                bx, by, bw = self.get_bar_geometry(player)
                
                if self.selected_index == 'bar' and self.engine.turn == player:
                    with self.canvas:
                        Color(*COLOR_HIGHLIGHT)
                        Ellipse(pos=(bx + bw/2 - radius/1.5, by - radius/1.5), 
                               size=(radius*1.5, radius*1.5))

                col = COLOR_P1 if player == 1 else COLOR_P2
                for c in range(count):
                    with self.canvas:
                        Color(*col)
                        Ellipse(pos=(bx + bw/2 - radius/2, by + c*(radius+2)), size=(radius, radius))

        self._draw_cube()

    def _draw_cube(self):
        """Draw the doubling cube on the board."""
        if self.crawford_active:
            return # No cube in Crawford game

        cube_size = dp(40)
        w, h = self.size
        x = w - cube_size - dp(10)
        
        if self.cube_owner == 1:
            y = dp(10)
        elif self.cube_owner == -1:
            y = h - cube_size - dp(10)
        else:
            y = h/2 - cube_size/2
        
        with self.canvas:
            Color(*COLOR_CUBE)
            RoundedRectangle(pos=(x, y), size=(cube_size, cube_size), radius=[dp(5)])
            
            Color(0.4, 0.05, 0.05, 1)
            Line(rounded_rectangle=(x, y, cube_size, cube_size, dp(5)), width=2)

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
            if col == 7: return False
            
            if ty < h/2:
                if col > 7: clicked = 13 - col
                elif col < 7: clicked = 6 + (6 - col)
            else:
                if col < 7: clicked = 12 + (col - 1)
                elif col > 7: clicked = 18 + (col - 8)

        if clicked is not None:
            self.handle_click(clicked)
            return True
        return False

    def handle_click(self, index):
        # 1. Execute move
        if index in self.legal_moves and self.selected_index is not None:
            self.engine.step_atomic((self.selected_index, index))
            self.has_moved = True
            self.selected_index = None
            self.legal_moves = []
            
            # Check win immediately
            winner, mult = self.engine.check_win()
            if winner != 0:
                if hasattr(self, 'game_screen'):
                    self.game_screen.handle_game_end(winner, mult, dropped=False)
            
            self.redraw()
            if hasattr(self, 'game_screen'):
                self.game_screen.dice_widget.set_dice(self.engine.dice)
                self.game_screen.update_buttons()
            return

        # 2. Select source
        can_select = False
        if index == 'bar':
            p_idx = 0 if self.engine.turn == 1 else 1
            if self.engine.bar[p_idx] > 0: can_select = True
        elif isinstance(index, int) and 0 <= index <= 23:
            count = self.engine.board[index]
            if (self.engine.turn == 1 and count > 0) or (self.engine.turn == -1 and count < 0):
                p_idx = 0 if self.engine.turn == 1 else 1
                if self.engine.bar[p_idx] == 0: can_select = True
                else: print("Must move from bar first!")

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
                device = torch.device(Config.DEVICE)
                model = get_model().to(device)
                load_checkpoint(filepath, model, device=device)
                model.eval()
                self.ai_model = model
                self.ai_device = device
                self.ai_mcts = MCTS(model, device)
                Clock.schedule_once(lambda dt: callback(True, filepath))
            except Exception as e:
                Clock.schedule_once(lambda dt: callback(False, str(e)))
        threading.Thread(target=_load, daemon=True).start()

    def run_ai(self, finished_callback):
        if not self.ai_model and not isinstance(self.engine, BackgammonGame): 
            # If using mock engine without model, simulate random delay
            self.is_ai_thinking = True
            def _mock_ai():
                time.sleep(1)
                if not self.engine.dice: self.engine.roll_dice()
                Clock.schedule_once(lambda dt: self.game_screen.dice_widget.set_dice(self.engine.dice))
                
                while self.engine.dice:
                    time.sleep(0.5)
                    moves = self.engine.get_legal_moves()
                    if not moves: break
                    move = random.choice(moves)
                    self.engine.step_atomic(move)
                    Clock.schedule_once(lambda dt: self.redraw())
                    Clock.schedule_once(lambda dt: self.game_screen.dice_widget.set_dice(self.engine.dice))
                    if self.engine.check_win()[0] != 0: break
                
                Clock.schedule_once(lambda dt: finished_callback())
            threading.Thread(target=_mock_ai, daemon=True).start()
            return

        # Real AI Logic
        if not self.ai_model: return
        self.is_ai_thinking = True
        
        def _think():
            # 1. Decision: Double? (Placeholder for AI doubling logic)
            # For now, AI never doubles in this snippet, but you would check neural net value here.
            
            # 2. Roll
            if not self.engine.dice:
                self.engine.roll_dice()
                Clock.schedule_once(lambda dt: self.game_screen.dice_widget.set_dice(self.engine.dice))
                time.sleep(0.5)

            # 3. Move
            while self.engine.dice:
                legal = self.engine.get_legal_moves()
                if not legal: break
                
                root = self.ai_mcts.search(self.engine, 0, 0)
                if root and root.children:
                    action = max(root.children.items(), key=lambda x: x[1].visits)[0]
                else:
                    action = legal[0]
                
                self.engine.step_atomic(action)
                Clock.schedule_once(lambda dt: self.redraw())
                Clock.schedule_once(lambda dt: self.game_screen.dice_widget.set_dice(self.engine.dice))
                
                if self.engine.check_win()[0] != 0: break
                time.sleep(0.6)

            Clock.schedule_once(lambda dt: finished_callback())

        threading.Thread(target=_think, daemon=True).start()


class GameScreen(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(5)
        self.match_over = False

        # --- TOP CONTROL BAR ---
        controls = BoxLayout(size_hint_y=None, height=dp(50), padding=dp(5), spacing=dp(5))
        
        btn_new = Button(text="New Match", size_hint_x=None, width=dp(90), 
                          background_color=(0.5, 0.2, 0.2, 1))
        btn_new.bind(on_press=self.show_new_match_dialog)
        
        btn_load = Button(text="Load AI", size_hint_x=None, width=dp(80), 
                         background_color=(0.2, 0.5, 0.5, 1))
        btn_load.bind(on_press=self.show_load_dialog)
        
        self.lbl_score = Label(text="White: 0 | Black: 0 (Goal: 3)", font_size='16sp', bold=True)
        
        controls.add_widget(btn_new)
        controls.add_widget(btn_load)
        controls.add_widget(self.lbl_score)
        self.add_widget(controls)

        # --- INFO BAR (Cube & Dice) ---
        info_bar = BoxLayout(size_hint_y=None, height=dp(60), spacing=dp(10))
        
        cube_box = BoxLayout(size_hint_x=None, width=dp(120), orientation='vertical')
        self.lbl_cube = Label(text="Cube: 1", font_size='14sp', bold=True, color=(1, 0.3, 0.3, 1))
        self.lbl_cube_owner = Label(text="(Center)", font_size='12sp', color=(0.7, 0.7, 0.7, 1))
        cube_box.add_widget(self.lbl_cube)
        cube_box.add_widget(self.lbl_cube_owner)
        info_bar.add_widget(cube_box)
        
        self.dice_widget = DiceWidget(size_hint=(1, 1))
        info_bar.add_widget(self.dice_widget)
        self.add_widget(info_bar)

        # --- STATUS LABEL ---
        self.lbl_status = Label(text="Welcome! Start a new match.", size_hint_y=None, height=dp(30), color=(0.9, 0.9, 0.5, 1))
        self.add_widget(self.lbl_status)

        # --- ACTION BUTTONS ---
        action_bar = BoxLayout(size_hint_y=None, height=dp(50), padding=dp(5), spacing=dp(5))
        
        self.btn_double = Button(text="Double", background_color=(0.8, 0.4, 0.1, 1))
        self.btn_double.bind(on_press=self.human_offers_double)
        
        self.btn_roll = Button(text="Roll Dice", background_color=(0.2, 0.6, 0.2, 1))
        self.btn_roll.bind(on_press=self.roll_dice)
        
        self.btn_takeback = Button(text="Take Back", background_color=(0.6, 0.3, 0.1, 1), disabled=True)
        self.btn_takeback.bind(on_press=self.take_back)
        
        self.btn_approve = Button(text="End Turn", background_color=(0.1, 0.5, 0.8, 1), disabled=True)
        self.btn_approve.bind(on_press=self.approve_turn)

        action_bar.add_widget(self.btn_double)
        action_bar.add_widget(self.btn_roll)
        action_bar.add_widget(self.btn_takeback)
        action_bar.add_widget(self.btn_approve)
        self.add_widget(action_bar)

        # --- BOARD ---
        board_container = FloatLayout()
        self.board = BackgammonBoard()
        self.board.game_screen = self 
        board_container.add_widget(self.board)
        self.add_widget(board_container)
        
        self.update_buttons()

    def show_new_match_dialog(self, instance):
        """Dialog to set match length."""
        content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        content.add_widget(Label(text="Match Length (Points):", size_hint_y=None, height=dp(30)))
        
        spinner = Spinner(text='3', values=('1', '3', '5', '7', '9', '11'), size_hint_y=None, height=dp(40))
        content.add_widget(spinner)
        
        btn_start = Button(text="Start Match", size_hint_y=None, height=dp(50))
        content.add_widget(btn_start)
        
        popup = Popup(title="New Match Settings", content=content, size_hint=(0.5, 0.4))
        
        def start(x):
            target = int(spinner.text)
            self.start_new_match(target)
            popup.dismiss()
        
        btn_start.bind(on_press=start)
        popup.open()

    def start_new_match(self, target):
        self.match_over = False
        self.board.reset_match(target)
        self.update_score_display()
        self.update_cube_display()
        self.update_status("New Match Started!")
        self.dice_widget.set_dice([])
        self.update_buttons()
        # If AI starts (randomize start player logic could be added here)
        if self.board.game_mode in ['ai_vs_human', 'ai_vs_ai'] and self.board.engine.turn == -1:
            self.trigger_ai_auto()

    def update_score_display(self):
        w = self.board.match_score[1]
        b = self.board.match_score[-1]
        t = self.board.match_target
        self.lbl_score.text = f"White: {w} | Black: {b} (Goal: {t})"

    def update_cube_display(self):
        if self.board.crawford_active:
            self.lbl_cube.text = "CRAWFORD"
            self.lbl_cube_owner.text = "No Cube"
            return

        self.lbl_cube.text = f"Cube: {self.board.cube_value}"
        if self.board.cube_owner == 0:
            self.lbl_cube_owner.text = "(Center)"
        elif self.board.cube_owner == 1:
            self.lbl_cube_owner.text = "(White owns)"
        else:
            self.lbl_cube_owner.text = "(Black owns)"

    def update_buttons(self):
        if self.match_over or self.board.is_ai_thinking:
            self.btn_double.disabled = True
            self.btn_roll.disabled = True
            self.btn_takeback.disabled = True
            self.btn_approve.disabled = True
            return

        is_human = self.board.is_human_turn()

        # Safely compute how many dice are currently present (0 if none)
        dice_attr = getattr(self.board.engine, 'dice', None)
        dice_count = len(dice_attr) if dice_attr is not None else 0
        has_dice = dice_count > 0

        # If dice exist, ask engine for legal moves; otherwise no moves to consider
        has_moves = bool(self.board.engine.get_legal_moves()) if has_dice else False
        has_moved = self.board.has_moved

        # Cube Rules (unchanged logic)
        can_double = False
        if is_human and not has_dice and not self.board.crawford_active:
            if self.board.cube_owner == 0 or self.board.cube_owner == self.board.engine.turn:
                can_double = True

        self.btn_double.disabled = not can_double
        self.btn_roll.disabled = not is_human or has_dice
        self.btn_takeback.disabled = not is_human or not has_moved

        # Approve (End Turn) logic:
        # - dice_exhausted: we had a roll this turn (turn_backup set) and dice_count is now 0
        # - no_moves_left: dice exist but there are no legal moves for them
        # We use turn_backup to ensure that approve is only possible after a roll (not at start of turn).
        had_roll_this_turn = self.board.turn_backup is not None
        dice_exhausted = had_roll_this_turn and dice_count == 0
        no_moves_left = has_dice and not has_moves and had_roll_this_turn

        # enable approve only for human when (dice exhausted OR no moves left)
        self.btn_approve.disabled = not is_human or not (dice_exhausted or no_moves_left)

    def roll_dice(self, instance):
        # Safety checks
        if self.match_over:
            return
        if self.board.is_ai_thinking:
            return
        if not self.board.is_human_turn():
            return
        if self.board.engine.dice:
            return  # already rolled

        # Save state for take-back
        self.board.save_turn_state()

        # Roll
        self.board.engine.roll_dice()

        # Update UI
        self.dice_widget.set_dice(self.board.engine.dice)
        self.update_status()
        self.board.redraw()
        self.update_buttons()


    # --- DOUBLING LOGIC ---
    def human_offers_double(self, instance):
        """Human clicks Double button."""
        # Check logic again just in case
        if self.board.crawford_active: return
        
        # AI Decision (Simulated)
        # In a real app, self.board.ai_model would evaluate position
        ai_accepts = True # Placeholder: AI always accepts for now
        
        if ai_accepts:
            self.perform_double_accept()
            self.update_status("AI Accepted Double!")
        else:
            # AI Drops
            self.handle_game_end(winner=1, multiplier=1, dropped=True)

    def perform_double_accept(self):
        self.board.cube_value *= 2
        self.board.cube_owner = -1 * self.board.engine.turn 
        self.update_cube_display()

    def offer_double_to_human(self):
        """Called when AI decides to double."""
        if self.board.crawford_active: return
        
        content = BoxLayout(orientation='vertical', padding=dp(10), spacing=dp(10))
        lbl = Label(text=f"AI offers double to {self.board.cube_value * 2}.\nTake or Drop?")
        btns = BoxLayout()
        btn_take = Button(text="Take", background_color=(0.2, 0.8, 0.2, 1))
        btn_drop = Button(text="Drop", background_color=(0.8, 0.2, 0.2, 1))
        btns.add_widget(btn_take)
        btns.add_widget(btn_drop)
        content.add_widget(lbl)
        content.add_widget(btns)
        
        popup = Popup(title="Double Offered", content=content, size_hint=(0.6, 0.4), auto_dismiss=False)
        
        def take(x):
            self.perform_double_accept()
            popup.dismiss()
            # Continue AI turn
            self.trigger_ai_auto(continue_turn=True)
            
        def drop(x):
            popup.dismiss()
            # AI wins immediately
            self.handle_game_end(winner=-1, multiplier=1, dropped=True)
            
        btn_take.bind(on_press=take)
        btn_drop.bind(on_press=drop)
        popup.open()

    # --- END GAME LOGIC ---
    def handle_game_end(self, winner, multiplier, dropped=False):
        """
        winner: 1 or -1
        multiplier: 1 (Normal), 2 (Gammon), 3 (Backgammon)
        dropped: True if game ended via cube drop
        """
        # Calculate Points
        points = self.board.cube_value * multiplier
        
        # If dropped, multiplier is always 1, but cube value is current
        if dropped:
            points = self.board.cube_value 
            type_str = "Drop"
        else:
            if multiplier == 3: type_str = "BACKGAMMON (x3)"
            elif multiplier == 2: type_str = "GAMMON (x2)"
            else: type_str = "Normal Win"

        winner_name = "White" if winner == 1 else "Black"
        
        # Update Match Score
        self.board.match_score[winner] += points
        self.update_score_display()
        
        # Check Match Win
        target = self.board.match_target
        if self.board.match_score[winner] >= target:
            self.match_over = True
            self.lbl_status.text = f"MATCH OVER! {winner_name} wins {self.board.match_score[winner]} to {self.board.match_score[winner*-1]}"
            self.update_buttons()
            
            # Show Popup
            popup = Popup(title="Match Over", 
                          content=Label(text=f"{winner_name} Wins the Match!"),
                          size_hint=(0.6, 0.4))
            popup.open()
        else:
            # Prepare next game
            msg = f"{winner_name} wins {points} pt(s) ({type_str}). Starting next game..."
            self.update_status(msg)
            Clock.schedule_once(lambda dt: self.start_next_game(), 3.0)

    def start_next_game(self):
        self.board.reset_game()
        self.update_cube_display()
        self.dice_widget.set_dice([])
        self.update_buttons()
        self.update_status("New Game Started")
        
        # Winner of previous game usually goes first, or roll for it. 
        # For simplicity, let's alternate or keep engine default.
        # Assuming engine.reset() sets turn to 1.
        if self.board.game_mode in ['ai_vs_human', 'ai_vs_ai'] and self.board.engine.turn == -1:
            self.trigger_ai_auto()

    def take_back(self, instance):
        if self.board.restore_turn_state():
            self.dice_widget.set_dice(self.board.engine.dice)
            self.update_buttons()
            self.update_status("Move Taken Back")

    def approve_turn(self, instance):
        self.board.engine.turn *= -1
        self.board.engine.dice = []
        self.dice_widget.set_dice([])
        self.board.has_moved = False
        self.board.turn_backup = None
        
        self.update_status()
        self.update_buttons()
        self.board.redraw()
        self.trigger_ai_auto()

    def trigger_ai_auto(self, continue_turn=False):
        if self.match_over: return
        
        is_ai_turn = False
        if self.board.game_mode == 'human_vs_ai' and self.board.engine.turn == -1: is_ai_turn = True
        elif self.board.game_mode == 'ai_vs_human' and self.board.engine.turn == 1: is_ai_turn = True
        elif self.board.game_mode == 'ai_vs_ai': is_ai_turn = True
            
        if is_ai_turn:
            self.update_status("AI Thinking...")
            self.update_buttons()
            
            # Hook for AI Double (Simple random check for demo if not continuing)
            # In real code, check NN value here.
            if not continue_turn and not self.board.crawford_active and self.board.cube_value < 4:
                # Randomly offer double for testing UI
                if random.random() < 0.05: 
                    self.offer_double_to_human()
                    return

            self.board.run_ai(self.on_ai_finished)

    def on_ai_finished(self):
        self.board.is_ai_thinking = False
        
        winner, mult = self.board.engine.check_win()
        if winner != 0:
            self.handle_game_end(winner, mult)
            return

        self.board.engine.turn *= -1
        self.board.engine.dice = []
        self.dice_widget.set_dice([])
        self.update_status()
        self.board.redraw()
        self.update_buttons()
        
        if self.board.game_mode == 'ai_vs_ai':
            Clock.schedule_once(lambda dt: self.trigger_ai_auto(), 1.0)

    def update_status(self, text=None):
        if text:
            self.lbl_status.text = text
            return
        p = "White" if self.board.engine.turn == 1 else "Black"
        self.lbl_status.text = f"{p}'s Turn"

    def show_load_dialog(self, instance):
        content = BoxLayout(orientation='vertical')
        path = os.getcwd()
        file_chooser = FileChooserListView(path=path, filters=['*.pt', '*.pth'])
        content.add_widget(file_chooser)
        
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
"""Entry point for the WebSocket server (HTML UI)."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import asyncio
import base64
import io
import json
import torch
import websockets
import random
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

from backgammon.engine import BackgammonGame
from backgammon.mcts import MCTS
from backgammon.model import get_model
from backgammon.config import Config
from backgammon.utils.cube import get_learned_cube_decision


# =========================
# CONFIG
# =========================
HOST = "0.0.0.0"
PORT = 8765
HTTP_PORT = 8080
DEVICE = Config.DEVICE
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pt")
UI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ui")


# =========================
# GAME SERVER
# =========================
class BackgammonServer:
    def __init__(self):
        # Use train_mode=False for real backgammon scoring (1, 2, 3)
        # Set train_mode=True if you want to use Config.R_WIN/R_GAMMON/R_BACKGAMMON values
        self.game = BackgammonGame(train_mode=False)
        self.model = None
        self.mcts = None
        self.model_filename = None
        self.game_mode = "human_vs_ai"  # human_vs_ai, ai_vs_human, ai_vs_ai
        self.game_over = False
        self.winner = 0
        self.waiting_for_cube_decision = False
        self.has_rolled = False  # Track if current player has rolled
        self.match_target = Config.MATCH_TARGET
        
        # Ensure engine knows the config target
        self.game.match_target = self.match_target
        
        # Try to load default model
        self._try_load_default_model()

    def _try_load_default_model(self):
        """Try to load default model from checkpoints folder"""
        paths_to_try = [MODEL_PATH, MODEL_PATH.replace("best_model.pt", "latest_model.pt")]
        
        for path in paths_to_try:
            if not os.path.exists(path):
                continue
                
            try:
                checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
                
                # Check model type from checkpoint config
                checkpoint_model_type = None
                if isinstance(checkpoint, dict) and 'config' in checkpoint:
                    checkpoint_model_type = checkpoint['config'].get('model_type', None)
                
                # Warn if types don't match
                if checkpoint_model_type and checkpoint_model_type != Config.MODEL_TYPE:
                    print(f"⚠️ Skipping {path}: model type mismatch (checkpoint: {checkpoint_model_type}, config: {Config.MODEL_TYPE})")
                    continue
                
                model = get_model().to(DEVICE)
                
                # Load state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.eval()
                self.model = model
                self.mcts = MCTS(self.model, device=DEVICE)
                self.model_filename = os.path.basename(path)
                
                elo = checkpoint.get('elo', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
                step = checkpoint.get('step', 'N/A') if isinstance(checkpoint, dict) else 'N/A'
                model_info = checkpoint_model_type or Config.MODEL_TYPE
                
                print(f"✅ Model loaded: {path}")
                print(f"   Type: {model_info}, ELO: {elo}, Step: {step}")
                return
                
            except Exception as e:
                print(f"⚠️ Error loading {path}: {e}")
        
        print(f"⚠️ No compatible model found - AI will not work until model is loaded")

    def load_model_from_data(self, filename, data_base64):
        """Load model from base64 encoded data sent from client"""
        try:
            model_bytes = base64.b64decode(data_base64)
            buffer = io.BytesIO(model_bytes)
            checkpoint = torch.load(buffer, map_location=DEVICE, weights_only=False)
            
            # Check model type
            checkpoint_model_type = None
            if isinstance(checkpoint, dict) and 'config' in checkpoint:
                checkpoint_model_type = checkpoint['config'].get('model_type', None)
            
            if checkpoint_model_type and checkpoint_model_type != Config.MODEL_TYPE:
                return {"type": "model_error", "error": f"Type mismatch: {checkpoint_model_type} vs {Config.MODEL_TYPE}"}
            
            model = get_model().to(DEVICE)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            self.model = model
            self.mcts = MCTS(self.model, device=DEVICE)
            self.model_filename = filename
            
            return {
                "type": "model_loaded", 
                "filename": filename, 
                "model_type": checkpoint_model_type or Config.MODEL_TYPE
            }
        except Exception as e:
            print(f"❌ Model load error: {e}")
            return {"type": "model_error", "error": str(e)}

    # =====================
    # SERIALIZATION
    # =====================
    def serialize(self, status=""):
        # Get legal moves for current player
        legal_moves = []
        if self.game.dice and not self.game_over:
            moves = self.game.get_legal_moves()
            for m in moves:
                # Engine returns atomic moves as ((src, dst), die_used)
                (src, dst), die_used = m
                legal_moves.append([src, dst])

        return {
            "type": "state",
            "payload": {
                "board": list(self.game.board),
                "bar": list(self.game.bar),
                "off": list(self.game.off),
                "turn": self.game.turn,
                "dice": list(self.game.dice) if self.game.dice else [],
                "cube_value": self.game.cube,
                "cube_owner": self.game.cube_owner,
                "waiting_for_cube": self.waiting_for_cube_decision,
                "can_double": self.can_offer_double() and not self.waiting_for_cube_decision,
                "legal_moves": legal_moves,
                "status": status,
                "game_over": self.game_over,
                "winner": self.winner,
                "mode": self.game_mode,
                "model_loaded": self.model is not None,
                "model_filename": self.model_filename,
                "model_type": Config.MODEL_TYPE,
                "has_rolled": self.has_rolled,
                "match_target": self.match_target,
                "match_scores": {
                    "white": self.game.match_scores.get(1, 0),
                    "black": self.game.match_scores.get(-1, 0)
                },
                "crawford": self.game.crawford_active,
                "crawford_used": self.game.crawford_used
            }
        }

    # =====================
    # GAME ACTIONS
    # =====================
    def new_game(self):
        self.game.reset()
        self.game_over = False
        self.winner = 0
        self.has_rolled = False
        crawford_msg = " (Crawford Game!)" if self.game.crawford_active else ""
        return self.serialize(f"New game started!{crawford_msg} Roll dice to begin.")
    
    def new_match(self, target=None):
        if target is not None:
            self.match_target = max(1, min(21, int(target)))
            self.game.match_target = self.match_target
            
        self.game.match_scores = {1: 0, -1: 0}
        self.game.crawford_used = False
        self.game.reset() 
        
        self.game_over = False
        self.winner = 0
        self.has_rolled = False
        return self.serialize(f"New match started! First to {self.match_target} points.")
    
    def _calculate_multiplier(self, winner):
        """
        Determines if the win is a Single, Gammon, or Backgammon.
        Real backgammon scoring: Single=1, Gammon=2, Backgammon=3
        winner: 1 (White) or -1 (Black)
        """
        loser = -winner
        loser_off = self.game.off[0 if loser == 1 else 1]
        
        # 1. If loser has borne off at least one checker, it's a Single Game
        if loser_off > 0:
            return 1
        
        # 2. Check for Backgammon: Loser has 0 off AND 
        #    has checkers in winner's home board or on the bar.
        
        # Winner White (1) home board is points 19-24
        # Winner Black (-1) home board is points 1-6
        loser_bar = self.game.bar[0 if loser == 1 else 1]
        
        if loser_bar > 0:
            return 3  # Backgammon (stuck on bar)
            
        # Check winner's home board for loser's pieces
        board = self.game.board
        if winner == 1: # White home is 19-24 (indices 18-23)
            for i in range(18, 24):
                if (board[i] > 0 and loser == 1) or (board[i] < 0 and loser == -1):
                    return 3
        else: # Black home is 1-6 (indices 0-5)
            for i in range(0, 6):
                if (board[i] > 0 and loser == 1) or (board[i] < 0 and loser == -1):
                    return 3
                    
        # 3. If loser has 0 off but escaped the winner's home, it's a Gammon
        return 2

    def _handle_game_win(self, winner, points_from_engine):
        """
        Handle game win. The engine has already:
        1. Calculated points = cube × multiplier (using Config.R_WIN/R_GAMMON/R_BACKGAMMON)
        2. Updated self.game.match_scores[winner] with those points in _finalize_win()
        
        We just need to display the result correctly.
        """
        # Engine already added points to match_scores in _finalize_win(), 
        # so DON'T add them again!
        total_points = points_from_engine
        
        # Calculate multiplier for display (1=Single, 2=Gammon, 3=Backgammon)
        multiplier = self._calculate_multiplier(winner)
        
        score_w = self.game.match_scores.get(1, 0)
        score_b = self.game.match_scores.get(-1, 0)
        
        match_winner = None
        if score_w >= self.match_target: match_winner = 1
        elif score_b >= self.match_target: match_winner = -1
        
        self.game_over = True
        self.winner = winner
        
        win_type = {1: "Single", 2: "a GAMMON", 3: "a BACKGAMMON"}[multiplier]
        winner_name = "White" if winner == 1 else "Black"
        
        if match_winner:
            return f"🏆 {winner_name} wins {win_type}! Final Match Score: {score_w}-{score_b}"
        else:
            return f"{winner_name} wins {win_type} (+{total_points}pts)! Score: {score_w}-{score_b}."

    def roll(self):
        if self.game_over:
            return self.serialize("Game is over. Start a new game.")
        if self.game.dice:
            return self.serialize("Already rolled! Make your moves.")
        if self.has_rolled:
            return self.serialize("Already rolled this turn!")
        
        self.game.roll_dice()
        self.has_rolled = True
        
        legal = self.game.get_legal_moves()
        if not legal:
            status = f"Rolled {self.game.dice[0]}, {self.game.dice[1]} - No legal moves!"
        else:
            status = f"Rolled {self.game.dice[0]}, {self.game.dice[1]}"
        
        return self.serialize(status)

    def double(self):
        if self.game_over:
            return self.serialize("Game is over")

        if not self.game.can_double():
            return self.serialize("Cannot double now")

        if self.game.apply_double():
            return self.serialize(f"Cube doubled to {self.game.cube}")
        else:
            return self.serialize("Double rejected")

    def end_turn(self):
        if self.game_over:
            return self.serialize("Game is over")
        if not self.has_rolled:
            return self.serialize("Roll dice first!")
        
        self.game.switch_turn()
        self.has_rolled = False
        
        turn_name = "White" if self.game.turn == 1 else "Black"
        return self.serialize(f"{turn_name}'s turn")

    def make_move(self, src, dst):
        if self.game_over:
            return self.serialize("Game is over")
        if not self.game.dice:
            return self.serialize("Roll dice first!")
        
        if src == "bar": src = "bar"
        elif isinstance(src, str): src = int(src)
            
        if dst == "off": dst = "off"
        elif isinstance(dst, str): dst = int(dst)
        
        legal_moves = self.game.get_legal_moves()
        # Engine returns atomic moves as ((src, dst), die_used)
        # We need to check if any legal move has matching src and dst
        move_exists = False
        chosen_move = None
        for m in legal_moves:
            (m_src, m_dst), die_used = m
            if m_src == src and m_dst == dst:
                move_exists = True
                chosen_move = m
                break
        
        if not move_exists:
            return self.serialize(f"Illegal move: {src} -> {dst}")
        
        try:
            # Execute the atomic move with full format
            winner, points = self.game.step_atomic(chosen_move)
        except ValueError as e:
            return self.serialize(f"Engine Error: {str(e)}")
        
        if winner != 0:
            status = self._handle_game_win(winner, points)
            return self.serialize(status)
        
        if self.game.dice:
            legal = self.game.get_legal_moves()
            if not legal:
                return self.serialize("No more legal moves - end your turn")
            return self.serialize("Move made - continue or end turn")
        
        return self.serialize("All dice used - end your turn")

    def set_mode(self, mode):
        if mode in ["human_vs_ai", "ai_vs_human", "ai_vs_ai"]:
            self.game_mode = mode
            return self.serialize(f"Mode set to {mode}")
        return self.serialize("Invalid mode")
    
    def can_offer_double(self):
        g = self.game
        player = g.turn
        opponent = -player

        # No doubling after roll
        if self.has_rolled:
            return False

        # Crawford rule
        if g.crawford_active:
            return False

        # Ownership rule: 0 = center, 1 = white, -1 = black
        if g.cube_owner not in (0, player):
            return False

        # Max cube: cannot double beyond remaining points to win
        remaining_points = self.match_target - min(
            g.match_scores.get(player, 0),
            g.match_scores.get(opponent, 0)
        )

        if g.cube > remaining_points:
            return False

        return True

    async def offer_double(self, websocket):
        """Initiates the doubling process."""
        if not self.can_offer_double():
            return self.serialize("Cannot double right now.")
        
        self.waiting_for_cube_decision = True
        # We switch the 'turn' visually so the UI knows who must decide
        self.game.turn *= -1
        
        response = self.serialize("Double offered! Take or Pass?")
        await websocket.send(json.dumps(response))
        
        # If AI's turn to decide, let AI decide immediately
        if self.is_ai_turn():
            await asyncio.sleep(0.5)  # Small delay for drama
            await self.ai_cube_decision(websocket)
    
        return None  # Already sent response

    async def ai_cube_decision(self, websocket):
        """AI decides whether to take or pass a double."""
        if not self.waiting_for_cube_decision:
            return
        
        if not self.model or not self.mcts:
            # Without model, AI randomly decides
            import random
            choice = random.choice([True, False])
        else:
            # Use the learned cube decision from utils
            my_score = self.game.match_scores.get(self.game.turn, 0)
            opp_score = self.game.match_scores.get(-self.game.turn, 0)
            
            take_choice, _ = get_learned_cube_decision(
                self.model, self.game, DEVICE, my_score, opp_score, stochastic=False
            )
            choice = (take_choice == 1)
        
        if choice:
            await websocket.send(json.dumps(self.take_double()))
        else:
            await websocket.send(json.dumps(self.refuse_double()))

    def take_double(self):
        """Player accepts the double."""
        if not self.waiting_for_cube_decision:
            return self.serialize("No double offered.")

        # Revert turn to the offerer so apply_double() works correctly
        self.game.turn *= -1 
        self.game.apply_double()
        self.waiting_for_cube_decision = False
        
        return self.serialize(f"Double accepted! Cube is now {self.game.cube}")

    def refuse_double(self):
        """Player refuses (Passes). The game ends immediately."""
        if not self.waiting_for_cube_decision:
            return self.serialize("No double offered.")

        # In engine.py, handle_cube_refusal() gives points to the current turn.
        # Since we flipped the turn in offer_double, the 'offerer' is now -turn.
        # So we flip back first.
        self.game.turn *= -1
        winner, points = self.game.handle_cube_refusal()
        
        self.waiting_for_cube_decision = False
        status = self._handle_game_win(winner, points)
        return self.serialize(status)

    # =====================
    # AI
    # =====================
    def is_ai_turn(self):
        if self.game_mode == "human_vs_ai":
            return self.game.turn == -1
        elif self.game_mode == "ai_vs_human":
            return self.game.turn == 1
        elif self.game_mode == "ai_vs_ai":
            return True
        return False

    async def ai_move(self, websocket):
        """Execute AI move with doubling/refusal and send updates."""
        if self.game_over or not self.is_ai_turn():
            return

        if not self.model or not self.mcts:
            await websocket.send(json.dumps(
                self.serialize("⚠️ No model loaded! Please load a model first.")
            ))
            return

        # Cache scores for MCTS
        my_score = self.game.match_scores.get(self.game.turn, 0)
        opp_score = self.game.match_scores.get(-self.game.turn, 0)

        # --------------------
        # 1. AI Decision: Doubling (Pre-Roll)
        # --------------------
        if not self.has_rolled and self.can_offer_double():
            # Use the model's cube head to decide
            # take_choice == 1 means the AI would TAKE, but we use logic to decide to OFFER
            # For simplicity, we trigger the doubling logic if the model sees high value
            take_choice, _ = get_learned_cube_decision(
                self.model, self.game, DEVICE, my_score, opp_score, stochastic=False
            )
            
            # Logic: If the position is strong enough that the opponent might pass
            # or it's a "volatile" double, the AI should offer.
            # (Refining this logic is a next step; for now, let's focus on the move crash)
            pass 

        # --------------------
        # 2. Roll Dice
        # --------------------
        if not self.game.dice:
            self.game.roll_dice()
            self.has_rolled = True
            await websocket.send(json.dumps(
                self.serialize(f"AI rolled {self.game.dice[0]}, {self.game.dice[1]}")
            ))
            await asyncio.sleep(0.4)

        # --------------------
        # 3. Atomic Move Loop
        # --------------------
        while self.game.dice and not self.game_over:
            # Source of Truth: Get legal moves from the engine
            legal = self.game.get_legal_moves()
            if not legal:
                # No legal moves possible with remaining dice
                break

            # CRITICAL FIX: Reset the tree before searching to ensure the root
            # strictly matches the engine's current board and dice state.
            self.mcts.reset() 

            # Search on the current game state
            # Note: We pass self.game directly as search saves/restores state internally
            root = self.mcts.search(self.game, my_score, opp_score)

            if root.children:
                # Filter children to ensure selection is strictly from current legal engine moves
                legal_children = [c for c in root.children if c.action in legal]
                
                if legal_children:
                    # Choose the move with the most simulations (robustness)
                    best_action = max(legal_children, key=lambda n: n.visits).action
                else:
                    # Fallback if MCTS failed to find legal moves (should not happen)
                    best_action = random.choice(legal)
            else:
                best_action = random.choice(legal)

            # Unpack for the UI, but step_atomic needs the full wrapper ((s, e), die)
            (src, dst), die_used = best_action
            
            try:
                # Execute the atomic move
                winner, points = self.game.step_atomic(best_action)
                
                # Report specific move and die used
                await websocket.send(json.dumps(
                    self.serialize(f"AI moved {src} → {dst} using {die_used}")
                ))
            except ValueError as e:
                # Catch-all for engine desync
                print(f"❌ AI Move Error: {e}")
                await websocket.send(json.dumps(self.serialize(f"AI Error: {str(e)}")))
                break

            if winner != 0:
                status = self._handle_game_win(winner, points)
                await websocket.send(json.dumps(self.serialize(status)))
                return

            # Short delay for visual flow in UI
            await asyncio.sleep(0.3)

        # --------------------
        # 4. End AI Turn
        # --------------------
        if not self.game_over:
            self.game.switch_turn()
            self.has_rolled = False
            turn_name = "White" if self.game.turn == 1 else "Black"
            await websocket.send(json.dumps(
                self.serialize(f"AI finished. {turn_name}'s turn")
            ))


    # =====================
    # ROUTER
    # =====================
    async def handle(self, msg, websocket):
        t = msg.get("type")

        if t == "hello":
            return self.serialize("Connected to Backgammon AI")

        if t == "new_game":
            return self.new_game()
        
        if t == "new_match":
            target = msg.get("target", None)
            return self.new_match(target)

        if t == "roll":
            result = self.roll()
            await websocket.send(json.dumps(result))
            if self.is_ai_turn() and not self.game_over:
                await asyncio.sleep(0.3)
                await self.ai_move(websocket)
            return None

        if t == "double":
            await self.offer_double(websocket)
            return None  # Already sent

        if t == "take_double":
            return self.take_double()

        if t == "refuse_double":
            return self.refuse_double()

        if t == "end_turn":
            result = self.end_turn()
            await websocket.send(json.dumps(result))
            if self.is_ai_turn() and not self.game_over:
                await asyncio.sleep(0.3)
                await self.ai_move(websocket)
            return None

        if t == "move":
            src = msg.get("from")
            dst = msg.get("to")
            return self.make_move(src, dst)

        if t == "set_mode":
            mode = msg.get("mode")
            result = self.set_mode(mode)
            await websocket.send(json.dumps(result))
            if self.is_ai_turn() and not self.game_over and not self.game.dice:
                await asyncio.sleep(0.3)
                await self.ai_move(websocket)
            return None

        if t == "ai_play":
            if self.is_ai_turn() and not self.game_over:
                await self.ai_move(websocket)
            return None

        if t == "load_model":
            filename = msg.get("filename", "model.pt")
            data = msg.get("data", "")
            return self.load_model_from_data(filename, data)

        return self.serialize("Unknown command")


# =========================
# WEBSOCKET LOOP
# =========================
async def handler(websocket):
    server = BackgammonServer()
    print("🎮 Client connected")

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
            except json.JSONDecodeError as e:
                print(f"⚠️ Invalid JSON received: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Invalid JSON: {str(e)}"
                }))
                continue
            
            try:
                response = await server.handle(msg, websocket)
                if response is not None:
                    await websocket.send(json.dumps(response))
            except Exception as e:
                print(f"❌ Handler error: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "error": f"Server error: {str(e)}"
                }))
    except websockets.exceptions.ConnectionClosed:
        print("👋 Client disconnected")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def start_http_server():
    """Start HTTP server to serve the HTML UI"""
    os.chdir(UI_DIR)
    
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Suppress HTTP request logs
    
    httpd = HTTPServer((HOST, HTTP_PORT), QuietHandler)
    print(f"🌐 HTTP server: http://localhost:{HTTP_PORT}/html_ui.html")
    httpd.serve_forever()


async def main():
    print(f"🎲 Starting Backgammon AI WebSocket on ws://{HOST}:{PORT}")
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"🖥️  Device: {DEVICE}")
    
    # Start HTTP server in background thread
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    
    # Suppress websocket handshake errors (from non-WS HTTP requests)
    import logging
    logging.getLogger('websockets').setLevel(logging.ERROR)
    
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
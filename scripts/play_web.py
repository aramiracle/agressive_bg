#!/usr/bin/env python3
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

from backgammon.engine import BackgammonGame
from backgammon.mcts import MCTS
from backgammon.model import get_model
from backgammon.config import Config


# =========================
# CONFIG
# =========================
HOST = "0.0.0.0"
PORT = 8765
DEVICE = Config.DEVICE
CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
MODEL_PATH = os.path.join(CHECKPOINTS_DIR, "best_model.pt")


# =========================
# GAME SERVER
# =========================
class BackgammonServer:
    def __init__(self):
        self.game = BackgammonGame()
        self.model = None
        self.mcts = None
        self.model_filename = None
        self.game_mode = "human_vs_ai"  # human_vs_ai, ai_vs_human, ai_vs_ai
        self.game_over = False
        self.winner = 0
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
                src, dst = m
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
                "can_double": self.game.can_double() and not self.has_rolled,
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
    
    def _handle_game_win(self, winner, points):
        score_w = self.game.match_scores.get(1, 0)
        score_b = self.game.match_scores.get(-1, 0)
        
        match_winner = None
        if score_w >= self.match_target: match_winner = 1
        elif score_b >= self.match_target: match_winner = -1
        
        self.game_over = True
        self.winner = winner
        
        winner_name = "White" if winner == 1 else "Black"
        
        if match_winner:
            return f"🏆 {winner_name} WINS THE MATCH! Final: {score_w}-{score_b}"
        else:
            return f"{winner_name} wins (+{points}pts)! Score: {score_w}-{score_b}. Click 'New Game'."

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
        
        if self.game.double():
            return self.serialize(f"Cube doubled to {self.game.cube}")
        else:
            return self.serialize("Double rejected (Not implemented in UI yet)")

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
        move = (src, dst)
        
        if move not in legal_moves:
            return self.serialize(f"Illegal move: {src} -> {dst}")
        
        try:
            winner, points = self.game.step_atomic(move)
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
        """Execute AI move and send state updates"""
        if self.game_over or not self.is_ai_turn():
            return
        
        if not self.model or not self.mcts:
            await websocket.send(json.dumps(
                self.serialize("⚠️ No model loaded! Please load a model first.")
            ))
            return
        
        if not self.game.dice:
            self.game.roll_dice()
            self.has_rolled = True
            await websocket.send(json.dumps(
                self.serialize(f"AI rolled {self.game.dice[0]}, {self.game.dice[1]}")
            ))
            await asyncio.sleep(0.3)
        
        # Keep moving until dice are exhausted or no legal moves
        while self.game.dice and not self.game_over:
            legal = self.game.get_legal_moves()
            if not legal:
                break
            
            # --- MCTS Step ---
            # IMPORTANT: Pass a COPY of the game to MCTS.
            # MCTS simulations modify the board. If we pass self.game, 
            # the live board gets corrupted (dice consumed), causing crashes 
            # on subsequent loops or step_atomic calls.
            game_copy = self.game.copy()
            
            root = self.mcts.search(
                game_copy, 
                self.game.match_scores.get(self.game.turn, 0),
                self.game.match_scores.get(-self.game.turn, 0)
            )
            
            if root.children:
                best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
            else:
                best_move = legal[0]
            
            # --- Apply Move ---
            winner, points = self.game.step_atomic(best_move)
            
            src, dst = best_move
            await websocket.send(json.dumps(
                self.serialize(f"AI moved {src} → {dst}")
            ))
            
            if winner != 0:
                status = self._handle_game_win(winner, points)
                await websocket.send(json.dumps(self.serialize(status)))
                return
            
            await asyncio.sleep(0.2)
        
        # End AI turn
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
            return self.double()

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
            msg = json.loads(message)
            response = await server.handle(msg, websocket)
            if response is not None:
                await websocket.send(json.dumps(response))
    except websockets.exceptions.ConnectionClosed:
        print("👋 Client disconnected")


async def main():
    print(f"🎲 Starting Backgammon AI WebSocket on ws://{HOST}:{PORT}")
    print(f"📁 Model path: {MODEL_PATH}")
    print(f"🖥️  Device: {DEVICE}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
"""
Backgammon Neural WebSocket Server
---------------------------------
Bridges HTML client <-> bg_engine + MCTS + PyTorch model

Protocol (JSON):
Client -> Server:
  { "type": "hello" }
  { "type": "new_game" }
  { "type": "roll" }
  { "type": "double" }
  { "type": "end_turn" }

Server -> Client:
  {
    "type": "state",
    "payload": {
      "board": [24 ints],
      "bar": [white, black],
      "turn": 1 or -1,
      "dice": [d1, d2] or [],
      "cube_value": int,
      "cube_owner": 0/1/-1,
      "status": "string"
    }
  }
"""

import asyncio
import json
import torch
import random
import websockets

from bg_engine import BackgammonGame
from mcts import MCTS
from model import get_model
from config import Config


# =========================
# CONFIG
# =========================
HOST = "0.0.0.0"
PORT = 8765
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model.pt"  # change if needed

# =========================
# GAME SERVER
# =========================
class BackgammonServer:
    def __init__(self):
        self.game = BackgammonGame()
        self.model = self._load_model()
        self.mcts = MCTS(self.model, device=DEVICE)

    def _load_model(self):
        model = get_model().to(DEVICE)
        try:
            sd = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(sd)
            print("Model loaded:", MODEL_PATH)
        except Exception as e:
            print("Model not loaded, using untrained model:", e)
        model.eval()
        return model

    # =====================
    # SERIALIZATION
    # =====================
    def serialize(self, status=""):
        return {
            "type": "state",
            "payload": {
                "board": list(self.game.board),
                "bar": list(self.game.bar),
                "turn": self.game.turn,
                "dice": list(self.game.dice) if self.game.dice else [],
                "cube_value": self.game.cube_value,
                "cube_owner": self.game.cube_owner,
                "status": status
            }
        }

    # =====================
    # GAME ACTIONS
    # =====================
    def new_game(self):
        self.game.reset()
        return self.serialize("New match started")

    def roll(self):
        if self.game.dice:
            return self.serialize("Already rolled")
        self.game.roll_dice()
        return self.serialize("Dice rolled")

    def double(self):
        ok = self.game.offer_double()
        if ok:
            return self.serialize("Doubling cube offered")
        return self.serialize("Cannot double now")

    def end_turn(self):
        # AI MOVE IF NEEDED
        if self.is_ai_turn():
            self._ai_move()

        self.game.end_turn()
        return self.serialize("Turn ended")

    # =====================
    # AI
    # =====================
    def is_ai_turn(self):
        # White = human, Black = AI (change if needed)
        return self.game.turn == -1

    def _ai_move(self):
        # Ensure dice
        if not self.game.dice:
            self.game.roll_dice()

        # MCTS SEARCH
        move = self.mcts.search(self.game, temperature=0)

        if move:
            self.game.apply_move(move)

    # =====================
    # ROUTER
    # =====================
    def handle(self, msg):
        t = msg.get("type")

        if t == "hello":
            return self.serialize("Connected to Backgammon AI")

        if t == "new_game":
            return self.new_game()

        if t == "roll":
            return self.roll()

        if t == "double":
            return self.double()

        if t == "end_turn":
            return self.end_turn()

        return self.serialize("Unknown command")


# =========================
# WEBSOCKET LOOP
# =========================
async def handler(websocket):
    server = BackgammonServer()
    print("Client connected")

    try:
        async for message in websocket:
            msg = json.loads(message)
            response = server.handle(msg)
            await websocket.send(json.dumps(response))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():
    print(f"Starting Backgammon AI WebSocket on {HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())

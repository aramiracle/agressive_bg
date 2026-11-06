"""Test checkpoint utilities."""

import sys
import os
import glob

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import torch

# Pattern -> values to write
CHECKPOINT_TARGETS = {
    "checkpoints/*/latest_model.pt": {"elo": 400, "step": 1000},
    "checkpoints/*/best_model.pt": {"elo": 500, "step": 2000},
}


def checkpoint_files():
    """Return all matching checkpoints along with the values to apply."""
    for pattern, values in CHECKPOINT_TARGETS.items():
        for path in sorted(glob.glob(pattern)):
            yield path, values


def test_checkpoint_structure():
    """Test that checkpoints have expected structure."""
    files = list(checkpoint_files())
    if not files:
        print("Skipping test: no checkpoints found")
        return

    for path, _ in files:
        # Load checkpoint
        ckpt = torch.load(path, map_location="cpu")

        # Safety check
        assert isinstance(ckpt, dict), "Checkpoint should be a dictionary"

        # Print values
        print(f"Checkpoint contents ({path}):")
        print("step:", ckpt.get("step"))
        print("elo:", ckpt.get("elo"))
        print("loss:", ckpt.get("loss"))

        # Check required keys
        assert "model_state_dict" in ckpt, "Checkpoint should have model_state_dict"


def modify_checkpoint():
    """Utility to set step/elo on the configured checkpoints (not a test)."""
    files = list(checkpoint_files())
    if not files:
        print("No checkpoints found")
        return

    for path, values in files:
        # Load checkpoint
        ckpt = torch.load(path, map_location="cpu")

        # Safety check
        if not isinstance(ckpt, dict):
            raise ValueError(f"{path} is not a checkpoint dictionary")

        # Print old values
        print(f"\n{path} before:")
        print("step:", ckpt.get("step"))
        print("elo:", ckpt.get("elo"))
        print("loss:", ckpt.get("loss"))

        # Modify the configured values; keep loss intact
        ckpt.update(values)

        # Print new values
        print("After:")
        print("step:", ckpt.get("step"))
        print("elo:", ckpt.get("elo"))
        print("loss:", ckpt.get("loss"))

        # Save edited checkpoint in place
        torch.save(ckpt, path)

        print(f"Saved edited file: {path}")


if __name__ == "__main__":
    test_checkpoint_structure()
    modify_checkpoint()

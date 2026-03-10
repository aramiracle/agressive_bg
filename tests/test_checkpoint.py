"""Test checkpoint utilities."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

import torch

INPUT_FILE = "checkpoints_v1/latest_model.pt"
OUTPUT_FILE = "checkpoints_v1/latest_model.pt"


def test_checkpoint_structure():
    """Test that checkpoint has expected structure."""
    if not os.path.exists(INPUT_FILE):
        print(f"Skipping test: {INPUT_FILE} not found")
        return
    
    # Load checkpoint
    ckpt = torch.load(INPUT_FILE, map_location="cpu")

    # Safety check
    assert isinstance(ckpt, dict), "Checkpoint should be a dictionary"

    # Print values
    print("Checkpoint contents:")
    print("step:", ckpt.get("step"))
    print("elo:", ckpt.get("elo"))
    print("loss:", ckpt.get("loss"))
    
    # Check required keys
    assert "model_state_dict" in ckpt, "Checkpoint should have model_state_dict"


def modify_checkpoint():
    """Utility to modify checkpoint values (not a test)."""
    if not os.path.exists(INPUT_FILE):
        print(f"File not found: {INPUT_FILE}")
        return
    
    # Load checkpoint
    ckpt = torch.load(INPUT_FILE, map_location="cpu")

    # Safety check
    if not isinstance(ckpt, dict):
        raise ValueError("This .pt file is not a checkpoint dictionary")

    # Print old values
    print("Before:")
    print("step:", ckpt.get("step"))
    print("elo:", ckpt.get("elo"))
    print("loss:", ckpt.get("loss"))

    # Modify values
    ckpt["step"] = 0
    ckpt["elo"] = 400
    ckpt["loss"] = 2.0

    # Print new values
    print("\nAfter:")
    print("step:", ckpt["step"])
    print("elo:", ckpt["elo"])
    print("loss:", ckpt["loss"])

    # Save edited checkpoint
    torch.save(ckpt, OUTPUT_FILE)

    print(f"\nSaved edited file as: {OUTPUT_FILE}")


if __name__ == "__main__":
    test_checkpoint_structure()
    modify_checkpoint()


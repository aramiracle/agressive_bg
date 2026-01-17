import torch

INPUT_FILE = "checkpoints/latest_model.pt"
OUTPUT_FILE = "checkpoints/latest_model.pt"

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
ckpt["step"] = 10000
ckpt["elo"] = 1000
ckpt["loss"] = 2.0

# Print new values
print("\nAfter:")
print("step:", ckpt["step"])
print("elo:", ckpt["elo"])
print("loss:", ckpt["loss"])

# Save edited checkpoint
torch.save(ckpt, OUTPUT_FILE)

print(f"\nSaved edited file as: {OUTPUT_FILE}")

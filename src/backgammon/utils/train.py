import torch
import torch.nn as nn
from src.backgammon.config import Config
from src.backgammon.utils.distribution import smooth_distribution, jensen_shannon_loss
from src.backgammon.utils.checkpoint import load_checkpoint

def train_batch(model, optimizer, replay_buffer, batch_size, device, scaler):
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0

    # ------------------------------
    # Sample batch from replay buffer
    # ------------------------------
    batch, indices, weights = replay_buffer.sample(batch_size)

    # Stack tensors
    boards = torch.stack([x[0] for x in batch]).to(device).float()
    contexts = torch.stack([x[1] for x in batch]).to(device).float()
    rewards = torch.tensor([x[3] for x in batch], dtype=torch.float, device=device)
    weights_t = torch.tensor(weights, dtype=torch.float, device=device)

    # Normalize/clamp boards and contexts
    boards = boards / max(Config.CHECKERS_PER_PLAYER, 1)
    boards = torch.clamp(boards, -1.0, 1.0)
    contexts = torch.clamp(contexts, -1.0, 1.0)
    rewards = torch.clamp(rewards, -10.0, 10.0)

    # ------------------------------
    # Check batch for NaNs/Infs
    # ------------------------------
    for i, sample in enumerate(batch):
        for j, item in enumerate(sample):
            if isinstance(item, torch.Tensor) and (torch.isnan(item).any() or torch.isinf(item).any()):
                print(f"🚨 NaN/Inf in batch[{i}][{j}]: {item}")
            elif isinstance(item, (float, int)) and (item != item or item == float('inf') or item == -float('inf')):
                print(f"🚨 NaN/Inf in batch[{i}][{j}]: {item}")

    # ------------------------------
    # Forward pass (disable AMP for debugging)
    # ------------------------------
    with torch.amp.autocast(enabled=False, device_type='cuda'):
        p_from, p_to, v, cube_logits = model(boards, contexts)
        p_from, p_to, v, cube_logits = p_from.float(), p_to.float(), v.float(), cube_logits.float()

        # Check model outputs for NaNs/Infs
        if torch.isnan(v).any() or torch.isinf(v).any():
            print("🚨 NaN/Inf in value output v")
            print("Boards stats:", boards.min(), boards.max(), boards.mean())
            print("Contexts stats:", contexts.min(), contexts.max(), contexts.mean())
            print("v stats:", v.min(), v.max(), v.mean())
            raise RuntimeError("v is NaN/Inf! Check model forward and inputs")

        if torch.isnan(p_from).any() or torch.isinf(p_from).any():
            print("🚨 NaN/Inf in p_from output:", p_from)
        if torch.isnan(p_to).any() or torch.isinf(p_to).any():
            print("🚨 NaN/Inf in p_to output:", p_to)
        if torch.isnan(cube_logits).any() or torch.isinf(cube_logits).any():
            print("🚨 NaN/Inf in cube_logits output:", cube_logits)

        # ------------------------------
        # Value loss
        # ------------------------------
        td_errors = torch.abs(v.squeeze(-1) - rewards).detach()
        v_loss = (weights_t * (v.squeeze(-1) - rewards) ** 2).mean()

        # ------------------------------
        # Policy & cube loss
        # ------------------------------
        p_loss, c_loss = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0
        smoothing = Config.LABEL_SMOOTHING

        for i, (_, _, _, _, is_cube, visit_targets) in enumerate(batch):
            weight = weights_t[i]

            if is_cube:
                target = smooth_distribution(visit_targets.to(device).float(), smoothing, 2)
                logp = nn.functional.log_softmax(cube_logits[i], dim=0)
                loss_val = jensen_shannon_loss(logp, target)
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    print(f"🚨 NaN/Inf in cube JS loss at sample {i}: {loss_val}")
                c_loss += weight * loss_val
                c_count += 1
            else:
                target_f, target_t = visit_targets
                tf = smooth_distribution(target_f.to(device).float(), smoothing, 26)
                tt = smooth_distribution(target_t.to(device).float(), smoothing, 26)

                logp_f = nn.functional.log_softmax(p_from[i], dim=0)
                logp_t = nn.functional.log_softmax(p_to[i], dim=0)

                js_f = jensen_shannon_loss(logp_f, tf)
                js_t = jensen_shannon_loss(logp_t, tt)
                if torch.isnan(js_f) or torch.isnan(js_t):
                    print(f"🚨 NaN in JS loss at sample {i}: js_f={js_f}, js_t={js_t}")
                p_loss += weight * 0.5 * (js_f + js_t)
                p_count += 1

        # Total loss
        loss = v_loss
        if p_count > 0: loss += (p_loss / p_count)
        if c_count > 0: loss += (c_loss / c_count)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"🚨 NaN/Inf in total loss: v_loss={v_loss}, p_loss={p_loss}, c_loss={c_loss}")
            raise RuntimeError("Loss is NaN/Inf! Check forward pass.")

    # ------------------------------
    # Backward pass & gradient update
    # ------------------------------
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP)

    # Skip optimizer step if gradient exploded
    if torch.isfinite(grad_norm):
        scaler.step(optimizer)
        scaler.update()
        replay_buffer.update_priorities(indices, td_errors.cpu().numpy())
    else:
        print("🚨 Skipping step due to NaN/Inf gradient")
        scaler.update()
        return loss.item(), 0.0

    return loss.item(), grad_norm.item()

def load_model_with_config(config_path, model_path, device):
    import importlib.util
    from src.backgammon.model import BackgammonTransformer, BackgammonCNN

    spec = importlib.util.spec_from_file_location("model_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    ModelConfig = config_module.Config

    if ModelConfig.MODEL_TYPE == "transformer":
        model = BackgammonTransformer(config=ModelConfig).to(device)
    elif ModelConfig.MODEL_TYPE == "cnn":
        model = BackgammonCNN(config=ModelConfig).to(device)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {ModelConfig.MODEL_TYPE}")

    cp = load_checkpoint(model_path, model, None, device)
    elo = cp['elo'] if cp else ModelConfig.INITIAL_ELO
    model.eval()

    return model, elo
import torch
import torch.nn as nn
from src.backgammon.config import Config
from src.backgammon.utils.distribution import smooth_distribution, jensen_shannon_loss


def train_batch(model, optimizer, replay_buffer, batch_size, device, scaler):
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0

    # ------------------------------
    # Sample batch
    # ------------------------------
    batch, indices, weights = replay_buffer.sample(batch_size)

    if batch is None or len(batch) == 0:
        return 0.0, 0.0
    
    # transitions are usually (board, context, action, reward, is_cube, visit_targets)
    boards = torch.stack([x[0] for x in batch]).to(device)
    contexts = torch.stack([x[1] for x in batch]).to(device).float()
    rewards = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=device)
    weights_t = weights.clone().detach().to(device).float()

    # Hard Type/Range Safety
    boards = boards.long()
    contexts = contexts.float()
    vocab = model.embedding.num_embeddings
    if torch.min(boards) < 0 or torch.max(boards) >= vocab:
        raise RuntimeError("Board token out of embedding range")

    # ------------------------------
    # Forward pass
    # ------------------------------
    with torch.amp.autocast(enabled=False, device_type='cuda'):
        p_from, p_to, v, cube_logits = model(boards, contexts)

        p_from, p_to, v, cube_logits = p_from.float(), p_to.float(), v.float(), cube_logits.float()

        if torch.isnan(v).any():
            raise RuntimeError("Model produced NaN outputs")

        # Value loss
        td_errors = torch.abs(v.squeeze(-1) - rewards).detach()
        v_loss = (weights_t * (v.squeeze(-1) - rewards) ** 2).mean()

        # Policy & cube loss
        p_loss = torch.tensor(0.0, device=device)
        c_loss = torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0

        smoothing = Config.LABEL_SMOOTHING

        for i, transition in enumerate(batch):
            # Transition structure: (board, context, action, reward, is_cube, visit_targets)
            is_cube = transition[4]
            visit_targets = transition[5]
            weight = weights_t[i]

            if is_cube:
                # visit_targets is a [2] tensor for Pass/Take
                target = smooth_distribution(
                    visit_targets.to(device).float(),
                    smoothing,
                    2
                )
                logp = nn.functional.log_softmax(cube_logits[i], dim=0)
                loss_val = jensen_shannon_loss(logp, target)

                if torch.isfinite(loss_val):
                    c_loss += weight * loss_val
                    c_count += 1
            else:
                # visit_targets is (target_f, target_t) from MCTS
                target_f, target_t = visit_targets

                # IMPORTANT: target_f and target_t must be 26-dim tensors 
                # (0-23 points, 24 bar, 25 off)
                tf = smooth_distribution(target_f.to(device).float(), smoothing, Config.NUM_ACTIONS)
                tt = smooth_distribution(target_t.to(device).float(), smoothing, Config.NUM_ACTIONS)

                logp_f = nn.functional.log_softmax(p_from[i], dim=0)
                logp_t = nn.functional.log_softmax(p_to[i], dim=0)

                js_f = jensen_shannon_loss(logp_f, tf)
                js_t = jensen_shannon_loss(logp_t, tt)

                if torch.isfinite(js_f).all() and torch.isfinite(js_t).all():
                    p_loss += weight * 0.5 * (js_f + js_t)
                    p_count += 1

        # Total combined loss
        loss = v_loss
        if p_count > 0:
            loss += p_loss / p_count
        if c_count > 0:
            loss += c_loss / c_count

    # ------------------------------
    # Optimization Step
    # ------------------------------
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP)

    if torch.isfinite(grad_norm):
        scaler.step(optimizer)
        scaler.update()
        # Update priorities in buffer for PER
        replay_buffer.update_priorities(indices, td_errors)
    else:
        scaler.update()
        return loss.item(), 0.0

    return loss.item(), grad_norm.item()
import torch
import torch.nn as nn
from src.config import Config
from src.utils.distribution import smooth_distribution, jensen_shannon_loss


def train_batch(model, optimizer, replay_buffer, batch_size, device, scaler):
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0

    batch, indices, weights = replay_buffer.sample(batch_size)
    if batch is None or len(batch) == 0:
        return 0.0, 0.0

    boards    = torch.stack([x[0] for x in batch]).to(device).long()
    contexts  = torch.stack([x[1] for x in batch]).to(device).float()
    rewards   = torch.tensor([x[3] for x in batch], dtype=torch.float32, device=device)
    weights_t = weights.clone().detach().to(device).float()

    with torch.amp.autocast(enabled=False, device_type='cuda'):
        p_from, p_to, v, cube_logits = model(boards, contexts)

        # ----------------------------------------------------------------
        # Value loss — all transitions.
        # rewards are match equity changes, in [-1, 1] (no *2 scaling).
        # Value head is tanh → output in [-1, 1], perfectly aligned.
        # ----------------------------------------------------------------
        v_sq   = v.squeeze(-1)
        v_loss = (weights_t * (v_sq - rewards) ** 2).mean()

        p_loss = torch.tensor(0.0, device=device)
        c_loss = torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0
        smoothing = Config.LABEL_SMOOTHING

        # ----------------------------------------------------------------
        # Pre-compute batch-normalised advantage for cube REINFORCE.
        #
        # Why batch-normalise?
        #   Per-sample advantage = reward[i] - V(s[i]).  In a mixed batch
        #   (mostly move transitions, sparse cube transitions), the raw
        #   advantage values have high variance and inconsistent scale
        #   relative to the JS loss magnitude.  Batch-normalising across
        #   the cube transitions only (mean=0, std=1) stabilises the PG
        #   update and makes CUBE_LOSS_WEIGHT meaningful as a direct
        #   relative scale factor vs the policy and value losses.
        # ----------------------------------------------------------------
        cube_indices = [i for i, t in enumerate(batch) if t[4]]  # is_cube

        if cube_indices:
            cube_idx_t   = torch.tensor(cube_indices, device=device)
            raw_adv      = rewards[cube_idx_t] - v_sq[cube_idx_t].detach()
            adv_mean     = raw_adv.mean()
            adv_std      = raw_adv.std().clamp(min=1e-6)
            # Store normalised advantages keyed by position in batch
            adv_map = {
                cube_indices[k]: ((raw_adv[k] - adv_mean) / adv_std).item()
                for k in range(len(cube_indices))
            }
        else:
            adv_map = {}

        for i, transition in enumerate(batch):
            is_cube       = transition[4]
            visit_targets = transition[5]
            weight        = weights_t[i]

            if is_cube:
                # --------------------------------------------------------
                # REINFORCE for the cube head (actor-critic style).
                #
                # action_taken: recovered from one-hot probs via argmax.
                # advantage:    batch-normalised (computed above).
                # entropy:      regularisation to prevent premature collapse.
                #
                # Loss = -log π(a|s) * advantage - β * H(π)
                #
                # Positive advantage → reinforce the action (it led to a
                #   better-than-average match equity gain).
                # Negative advantage → suppress the action (it underperformed
                #   the value-head baseline).
                # --------------------------------------------------------
                action_taken = visit_targets.argmax().long()
                log_probs    = torch.log_softmax(cube_logits[i], dim=0)
                log_prob     = log_probs[action_taken]
                advantage    = adv_map.get(i, 0.0)
                entropy      = -(log_probs.exp() * log_probs).sum()

                pg_loss = -log_prob * advantage - Config.CUBE_ENTROPY_BETA * entropy

                if torch.isfinite(pg_loss):
                    c_loss  += weight * pg_loss
                    c_count += 1

            else:
                # --------------------------------------------------------
                # Move policy: MCTS visit-count targets via JS divergence.
                # --------------------------------------------------------
                target_f, target_t = visit_targets
                tf = smooth_distribution(target_f.to(device).float(), smoothing, Config.NUM_ACTIONS)
                tt = smooth_distribution(target_t.to(device).float(), smoothing, Config.NUM_ACTIONS)

                logp_f = nn.functional.log_softmax(p_from[i], dim=0)
                logp_t = nn.functional.log_softmax(p_to[i], dim=0)

                js_f = jensen_shannon_loss(logp_f, tf)
                js_t = jensen_shannon_loss(logp_t, tt)

                if torch.isfinite(js_f).all() and torch.isfinite(js_t).all():
                    p_loss  += weight * 0.5 * (js_f + js_t)
                    p_count += 1

        loss = v_loss
        if p_count > 0:
            loss = loss + p_loss / p_count
        if c_count > 0:
            loss = loss + (c_loss / c_count) * Config.CUBE_LOSS_WEIGHT

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP)

    if torch.isfinite(grad_norm):
        scaler.step(optimizer)
        scaler.update()
        replay_buffer.update_priorities(
            indices, torch.abs(v_sq - rewards).detach()
        )
    else:
        scaler.update()
        return loss.item(), 0.0

    return loss.item(), grad_norm.item()
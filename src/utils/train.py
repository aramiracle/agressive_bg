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
        # Rewards are match equity changes in [-1, 1].
        # Value head (tanh) outputs in [-1, 1] — perfectly aligned, no scaling.
        # ----------------------------------------------------------------
        v_sq   = v.squeeze(-1)
        v_loss = (weights_t * (v_sq - rewards) ** 2).mean()

        p_loss = torch.tensor(0.0, device=device)
        c_loss = torch.tensor(0.0, device=device)
        p_count, c_count = 0, 0
        smoothing = Config.LABEL_SMOOTHING

        for i, transition in enumerate(batch):
            is_cube       = transition[4]
            visit_targets = transition[5]
            weight        = weights_t[i]

            if is_cube:
                # --------------------------------------------------------
                # Cube policy: JS divergence against ME-derived soft target.
                #
                # This is the same loss family as the move policy (JS vs
                # MCTS visit counts), making all three losses commensurable:
                #   v_loss  : MSE,          always >= 0, scale ~[0, 1]
                #   p_loss  : JS/p_count,   always >= 0, scale ~[0, log2]
                #   c_loss  : JS/c_count,   always >= 0, scale ~[0, log2]
                #
                # The soft target (visit_targets, shape [2]) was computed at
                # collection time by compute_me_soft_target():
                #   target[1] = sigmoid(ev_double_vs_no * temperature)
                #   target[0] = 1 - target[1]
                #
                # ev_double_vs_no > 0 → target[1] > 0.5 → push toward double
                # ev_double_vs_no < 0 → target[1] < 0.5 → push toward no-double
                # ev_double_vs_no = 0 → target = [0.5, 0.5] → maximum uncertainty
                #
                # Label smoothing is applied identically to the move policy.
                # CUBE_LOSS_WEIGHT now has a clean, interpretable meaning:
                # it is the relative weight of cube JS loss vs move JS loss.
                # --------------------------------------------------------
                target = smooth_distribution(
                    visit_targets.to(device).float(), smoothing, 2
                )
                logp   = nn.functional.log_softmax(cube_logits[i], dim=0)
                c_loss_i = jensen_shannon_loss(logp, target)

                if torch.isfinite(c_loss_i):
                    c_loss  += weight * c_loss_i
                    c_count += 1

            else:
                # --------------------------------------------------------
                # Move policy: JS divergence against MCTS visit-count targets.
                # Unchanged.
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
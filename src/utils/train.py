import torch
import torch.nn.functional as F
from src.config import Config
from src.utils.distribution import smooth_distribution, jensen_shannon_loss_batch
from src.utils.outcome import money_weights


def cube_stats_text(stats):
    """One line of collection counts: games, doubles, take/drop, equity at the decision."""
    games = int(stats.get("games", 0))
    doubles = int(stats.get("doubles", 0))
    takes = int(stats.get("takes", 0))
    drops = int(stats.get("drops", 0))
    if games <= 0:
        return "no games"
    parts = [f"games {games}", f"doubles {doubles / games:.2f}/game"]
    if doubles > 0:
        parts.append(f"take {takes / doubles:.0%}")
        parts.append(f"drop {drops / doubles:.0%}")
        parts.append(f"eq@double {stats.get('sum_val_double', 0.0) / doubles:+.2f}")
        if drops:
            parts.append(f"eq@drop {stats.get('sum_val_drop', 0.0) / drops:+.2f}")
    return "  ".join(parts)


def train_batch(model, optimizer, replay_buffer, batch_size, device, scaler):
    """
    One optimisation step. Transitions are
        (board[28], ctx[CONTEXT_SIZE], outcome_target[6], is_cube, cube_target[2])

    Losses (all vectorised):
      outcome : soft-target cross-entropy between the predicted outcome
                distribution and the TD(lambda) target — every transition.
      cube    : Jensen-Shannon divergence against the ME-derived soft target —
                cube-decision transitions only.
    """
    if len(replay_buffer) < batch_size:
        return 0.0, 0.0

    batch, indices, weights = replay_buffer.sample(batch_size)
    if batch is None or len(batch) == 0:
        return 0.0, 0.0

    boards   = torch.stack([x[0] for x in batch]).to(device).long()
    contexts = torch.stack([x[1] for x in batch]).to(device).float()
    targets  = torch.stack([x[2] for x in batch]).to(device).float()
    is_cube  = torch.tensor([bool(x[3]) for x in batch], device=device)
    cube_tgt = torch.stack([x[4] for x in batch]).to(device).float()
    w        = weights.clone().detach().to(device).float()

    use_amp = device.type == 'cuda'
    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
        outcome_logits, cube_logits = model(boards, contexts)

    # Losses in fp32 regardless of autocast.
    logp   = F.log_softmax(outcome_logits.float(), dim=-1)
    ce     = -(targets * logp).sum(-1)                      # [B]
    v_loss = (w * ce).mean()

    loss = v_loss
    if is_cube.any():
        smoothing = Config.LABEL_SMOOTHING
        tgt   = smooth_distribution(cube_tgt[is_cube], smoothing, 2)
        logpc = F.log_softmax(cube_logits.float()[is_cube], dim=-1)
        js    = jensen_shannon_loss_batch(logpc, tgt)      # [Nc]
        finite = torch.isfinite(js)
        if finite.any():
            c_loss = (w[is_cube][finite] * js[finite]).mean()
            loss = loss + c_loss * Config.CUBE_LOSS_WEIGHT

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.GRAD_CLIP)

    if torch.isfinite(grad_norm):
        scaler.step(optimizer)
        scaler.update()
        # PER priority: equity error between predicted and target distributions.
        mw = money_weights().to(device)
        with torch.no_grad():
            eq_pred = (logp.exp() * mw).sum(-1)
            eq_tgt  = (targets * mw).sum(-1)
        replay_buffer.update_priorities(indices, (eq_pred - eq_tgt).abs())
    else:
        scaler.update()
        return loss.item(), 0.0

    return loss.item(), grad_norm.item()

import torch
import torch.nn as nn

def smooth_distribution(target, epsilon, num_classes):
    uniform = torch.ones_like(target) / num_classes
    return (1.0 - epsilon) * target + epsilon * uniform

def jensen_shannon_loss(log_p, q_target):
    """
    Calculates the JS Divergence as a scalar value.
    log_p: Log-probabilities from the model
    q_target: Probability distribution from MCTS
    """
    p = log_p.exp()
    m = 0.5 * (p + q_target)
    
    # We use reduction='sum' to get a single scalar value for the 
    # divergence between these two distributions.
    loss_pm = nn.functional.kl_div(log_p, m, reduction='sum')
    loss_qm = nn.functional.kl_div(q_target.log(), m, reduction='sum')
    
    return 0.5 * (loss_pm + loss_qm)


def jensen_shannon_loss_batch(log_p, q_target):
    """
    Row-wise JS divergence for a batch.
    log_p, q_target: [N, C]  ->  returns [N]
    """
    p = log_p.exp()
    m = 0.5 * (p + q_target)
    log_m = m.clamp_min(1e-12).log()
    kl_pm = (p * (log_p - log_m)).sum(-1)
    kl_qm = (q_target * (q_target.clamp_min(1e-12).log() - log_m)).sum(-1)
    return 0.5 * (kl_pm + kl_qm)

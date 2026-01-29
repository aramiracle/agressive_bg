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

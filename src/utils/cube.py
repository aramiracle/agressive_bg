import torch
import random

def get_learned_cube_decision(model, game, device, my_score, opp_score, stochastic=True, epsilon=0.0):
    """
    Get cube decision from the model with optional epsilon-greedy exploration.
    
    Returns:
        action: 0 (pass/drop), 1 (double/take)
        probs: Probability distribution
        value_est: The Value Head output (win probability from -1 to 1)
    """
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)
    
    with torch.no_grad():
        _, _, v, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        probs = torch.softmax(cube_logits.squeeze(0), dim=0)
        value_est = v.item()

    # FORCED EXPLORATION
    if epsilon > 0 and random.random() < epsilon:
        action = random.randint(0, 1)
    elif stochastic:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs).item()

    return action, probs, value_est
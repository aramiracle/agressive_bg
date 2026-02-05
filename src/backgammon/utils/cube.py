import torch
import random

def get_learned_cube_decision(model, game, device, my_score, opp_score, stochastic=True, epsilon=0.0):
    """
    Get cube decision from the model with optional epsilon-greedy exploration.
    
    Args:
        model: The neural network model
        game: Current game state
        device: torch device
        my_score: Current player's match score
        opp_score: Opponent's match score
        stochastic: If True, sample from distribution; if False, take argmax
        epsilon: Probability of taking random action (for exploration)
    
    Returns:
        action: 0 for pass/drop, 1 for double/take
        probs: Probability distribution from the model
    """
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)
    with torch.no_grad():
        _, _, _, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        probs = torch.softmax(cube_logits.squeeze(0), dim=0)

    # FORCED EXPLORATION: Critical for breaking the "never double" cycle
    if epsilon > 0 and random.random() < epsilon:
        action = random.randint(0, 1)  # Random cube decision
    elif stochastic:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs).item()

    return action, probs
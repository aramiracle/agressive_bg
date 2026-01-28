import torch

def get_learned_cube_decision(model, game, device, my_score, opp_score, stochastic=True):
    board_t, ctx_t = game.get_vector(my_score, opp_score, device=device, canonical=True)
    with torch.no_grad():
        _, _, _, cube_logits = model(board_t.unsqueeze(0), ctx_t.unsqueeze(0))
        probs = torch.softmax(cube_logits.squeeze(0), dim=0)

    if stochastic:
        action = torch.multinomial(probs, 1).item()
    else:
        action = torch.argmax(probs).item()

    return action, probs

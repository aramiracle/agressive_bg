from src.backgammon.config import Config

def finalize_history(history, current_won, total_points):
    data = []
    reward_magnitude = float(total_points) / (Config.MATCH_TARGET * (Config.R_BACKGAMMON + 1))
    final_reward = reward_magnitude if current_won else -reward_magnitude

    for board, ctx, act, turn, is_cube, probs in history:
        data.append((board, ctx, act, final_reward, is_cube, probs))
    return data

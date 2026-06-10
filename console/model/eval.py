import torch
import torch.nn as nn
import sys
import os
import random
import math
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from console.states.game_state import GameState
from console.model.quoridor_cnn import QuoridorModel, encodeState
from console.agents.given_agents.expectimax import expectimax
from console.agents.given_agents.minimax_ab_pruning import minimax_alpha_beta_pruning
from train import ACTION_SPACE, action_to_index

MODEL_PATH  = os.path.join(PROJECT_ROOT, "quoridor_model.pth")
NUM_GAMES   = 100
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = QuoridorModel()
    model.policy = nn.Linear(256, ACTION_SPACE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def recommend_move(model, game_state):
    game_state.player_two_walls_num = game_state.player_two_wall_num
    game_state.board = game_state.board.reshape(17, 17)
    with torch.no_grad():
        state = encodeState(game_state).unsqueeze(0).to(DEVICE)
        game_state.board = game_state.board.flatten()
        policy_logits, value = model(state)
        legal_moves = game_state.get_all_child_states(game_state.player_one)
        move_indices = {action_to_index(move): move for _, move in legal_moves}
        mask = torch.full((ACTION_SPACE,), float("-inf")).to(DEVICE)
        for idx in move_indices:
            mask[idx] = 0.0
        masked_logits = policy_logits.squeeze(0) + mask
        best_idx = torch.argmax(torch.softmax(masked_logits, dim=0)).item()
        return move_indices[best_idx], value.item()

def random_agent(game_state):
    moves = game_state.get_all_child_states(game_state.player_one)
    if not moves:
        return None
    return random.choice(moves)[1]

def expectimax_agent(game_state):
    player_one_maximizer = game_state.player_one
    d = {}
    for child in game_state.get_all_child_states(player_one_maximizer):
        value = expectimax(child[0], 2, False, player_one_maximizer)
        d[value] = child
    return d[max(d)][1] if d else None

def minimax_agent(game_state):
    player_one_maximizer = game_state.player_one
    d = {}
    for child in game_state.get_all_child_states(player_one_maximizer):
        value = minimax_alpha_beta_pruning(child[0], 3, -math.inf, math.inf, False, player_one_maximizer)
        d[value] = child
    return d[max(d)][1] if d else None

def run_game(model, opponent_fn, model_is_player_one):
    game = GameState()
    value_preds = []
    max_moves = 300

    while not game.is_end_state():
        if len(value_preds) >= max_moves:
            break
        if game.player_one == model_is_player_one:
            action, value = recommend_move(model, game.copy())
            value_preds.append(value)
        else:
            action = opponent_fn(game)
            if action is None:
                break
        game = game.execute_action(action)

    winner = game.get_winner()
    model_won = (winner == "P1") == model_is_player_one
    return model_won, value_preds

def compute_f1(predictions, actuals):
    tp = sum(p == 1 and a == 1 for p, a in zip(predictions, actuals))
    fp = sum(p == 1 and a == 0 for p, a in zip(predictions, actuals))
    fn = sum(p == 0 and a == 1 for p, a in zip(predictions, actuals))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def evaluate(model, opponent_fn, opponent_name):
    wins = 0
    all_value_preds = []
    all_actuals = []
    p1_wins = 0
    p2_wins = 0

    for i in tqdm(range(NUM_GAMES), desc=f"Model vs {opponent_name}"):
        model_is_p1 = i % 2 == 0
        model_won, value_preds = run_game(model, opponent_fn, model_is_p1)

        if model_won:
            wins += 1
        if model_won and model_is_p1:
            p1_wins += 1
        elif model_won and not model_is_p1:
            p2_wins += 1

        binary_preds = [1 if v > 0 else 0 for v in value_preds]
        actual_label = 1 if model_won else 0
        all_value_preds.extend(binary_preds)
        all_actuals.extend([actual_label] * len(binary_preds))

    win_rate = wins / NUM_GAMES
    f1 = compute_f1(all_value_preds, all_actuals)

    print(f"\nResults vs {opponent_name} ({NUM_GAMES} games):")
    print(f"  Win rate : {win_rate:.1%} ({wins}/{NUM_GAMES})")
    print(f"  P1 wins  : {p1_wins}/50")
    print(f"  P2 wins  : {p2_wins}/50")
    print(f"  Value F1 : {f1:.4f}")
    return win_rate, f1

if __name__ == "__main__":
    model = load_model()
    print(f"Device: {DEVICE}\n")
    print("=" * 50)

    evaluate(model, random_agent,     "Random")
    print("=" * 50)
    evaluate(model, expectimax_agent, "Expectimax")
    print("=" * 50)
    evaluate(model, minimax_agent, "Minimax (alpha-beta)")
    print("=" * 50)
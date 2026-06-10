import torch
from quoridor_cnn import encodeState
from train import *

#returns the index of the best next move and the predicted outcome of the game
def recommend_move(model, game_state):
    model.eval()
    with torch.no_grad():
        # encode current state
        state = encodeState(game_state)
        # get policy logits and value
        policy_logits, value = model(state.unsqueeze(0))
        # get legal moves and build index lookup
        legal_moves = game_state.get_all_child_states(True)
        # mask illegal moves
        move_indices = {action_to_index(move): move for _, move in legal_moves}
        mask = torch.full((ACTION_SPACE,), float('-inf'))
        for idx in move_indices.keys():
            mask[idx] = 0.0
        masked_logits = policy_logits.squeeze(0) + mask
        # apply softmax
        normalized_logits = torch.softmax(masked_logits, dim=0)
        # pick best index
        best_move_idx = torch.argmax(normalized_logits).item()
        # return corresponding move and value estimate
        best_move = move_indices[best_move_idx]
        outcome = value.item()
        return best_move, outcome
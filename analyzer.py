import torch
import torch.nn as nn
import sys
import os
import random
import io
import re

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from console.states.game_state import GameState, Mappings
from console.model.quoridor_cnn import QuoridorModel, encodeState
from console.util.wall_direction import WallDirection
from console.util.color import Color
from console.model.train import ACTION_SPACE, action_to_index

MODEL_PATH = os.path.join(PROJECT_ROOT, "quoridor_model.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = QuoridorModel()
    model.policy = nn.Linear(256, ACTION_SPACE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_model_top3(model, game_state):
    gs = game_state.copy()
    gs.player_two_walls_num = gs.player_two_wall_num
    gs.board = gs.board.reshape(17, 17)
    with torch.no_grad():
        state = encodeState(gs).unsqueeze(0).to(DEVICE)
        gs.board = gs.board.flatten()
        policy_logits, _ = model(state)
        legal_moves = gs.get_all_child_states(gs.player_one)
        move_indices = {action_to_index(move): move for _, move in legal_moves}
        if not move_indices:
            return [], None
        mask = torch.full((ACTION_SPACE,), float("-inf")).to(DEVICE)
        for idx in move_indices:
            mask[idx] = 0.0
        masked_logits = policy_logits.squeeze(0) + mask
        probs = torch.softmax(masked_logits, dim=0)
        sorted_indices = torch.argsort(probs, descending=True).tolist()
        top3 = [i for i in sorted_indices if i in move_indices][:3]
        best_move = move_indices[top3[0]] if top3 else None
        return top3, best_move

def rate_move(action, top3):
    if not top3:
        return "Normal"
    try:
        idx = action_to_index(action)
    except Exception:
        return "Normal"
    if idx == top3[0]:
        return "Good"
    elif idx in top3:
        return "Normal"
    else:
        return "Bad"

def color_rating(rating):
    if rating == "Good":
        return Color.GREEN + "Good" + Color.RESET
    elif rating == "Normal":
        return Color.YELLOW + "Normal" + Color.RESET
    else:
        return Color.RED + "Bad" + Color.RESET

def format_action(action):
    if len(action) == 2:
        r, c = action
        return f"Move to ({Mappings.INPUT_MAPPINGS_REVERSED[r]}, {Mappings.INPUT_MAPPINGS_REVERSED[c]})"
    else:
        r1, c1 = action[0], action[1]
        return f"Wall at ({Mappings.INPUT_MAPPINGS_REVERSED[r1]}, {Mappings.INPUT_MAPPINGS_REVERSED[c1]})"

def random_action(game_state):
    moves = game_state.get_all_child_states(game_state.player_one)
    if not moves:
        return None
    return random.choice(moves)[1]

def get_human_action(game_state):
    while True:
        value = input(f"Your move (M row,col or W row,col direction [N/S/E/W]): ")
        if value.upper().startswith("M"):
            parts = value[1:].split(",")
            if len(parts) != 2:
                print(Color.RED + "Invalid format." + Color.RESET)
                continue
            x_str, y_str = parts[0].strip().upper(), parts[1].strip().upper()
            if x_str not in Mappings.INPUT_MAPPINGS or y_str not in Mappings.INPUT_MAPPINGS:
                print(Color.RED + "Invalid coordinates." + Color.RESET)
                continue
            move = (Mappings.INPUT_MAPPINGS[x_str], Mappings.INPUT_MAPPINGS[y_str])
            if move not in game_state.get_available_moves(False):
                print(Color.RED + "Illegal move." + Color.RESET)
                continue
            return move
        elif value.upper().startswith("W"):
            parts = value[1:len(value)-1].split(",")
            if len(parts) != 2:
                print(Color.RED + "Invalid format." + Color.RESET)
                continue
            x_str, y_str = parts[0].strip().upper(), parts[1].strip().upper()
            if x_str not in Mappings.INPUT_MAPPINGS or y_str not in Mappings.INPUT_MAPPINGS:
                print(Color.RED + "Invalid coordinates." + Color.RESET)
                continue
            dir_str = value[-1].upper()
            if dir_str not in ["N", "S", "E", "W"]:
                print(Color.RED + "Invalid direction." + Color.RESET)
                continue
            direction = {"S": WallDirection.SOUTH, "N": WallDirection.NORTH,
                         "E": WallDirection.EAST, "W": WallDirection.WEST}[dir_str]
            x_int = Mappings.INPUT_MAPPINGS[x_str]
            y_int = Mappings.INPUT_MAPPINGS[y_str]
            is_valid, coords = game_state.check_wall_placement((x_int, y_int), direction)
            if not is_valid:
                print(Color.RED + "Illegal wall placement." + Color.RESET)
                continue
            return tuple(coords)
        else:
            print(Color.RED + "Use M for move or W for wall." + Color.RESET)

def play_game(model, human_mode):
    game      = GameState()
    history   = []
    max_moves = 300
    moves_count = 0

    print("\n" + "=" * 60)
    print("  You are P2 (Red). Opponent is P1 (Green).")
    print("  P2 starts at row A (0), needs to reach row Q (16).")
    print("=" * 60)

    while not game.is_end_state() and moves_count < max_moves:
        print()
        game.print_game_stats()
        print()
        game.print_board()
        print()

        if not game.player_one:
            state_before = game.copy()
            if human_mode:
                action = get_human_action(game)
            else:
                action = random_action(game)
                print(f"Auto P2 plays: {format_action(action)}")
            history.append((state_before, action))
        else:
            action = random_action(game)
            if action is None:
                break

        game = game.execute_action(action)
        moves_count += 1

    print()
    game.print_board()
    winner = game.get_winner()
    if winner == "P1":
        print(Color.GREEN + "\n  P1 wins!" + Color.RESET)
    else:
        print(Color.RED + "\n  P2 wins!" + Color.RESET)

    return history, winner

def strip_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def analyze_game(model, history, winner):
    output_lines = []

    def log(text=""):
        output_lines.append(strip_ansi(text))

    log("NOTE: Open this file in VS Code or a monospace editor for proper board alignment.")
    log("=" * 60)
    log("  GAME ANALYSIS")
    log("=" * 60)

    for i, (state, action) in enumerate(history):
        log(f"\n--- Move {i + 1} ---")
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        state.print_board()
        sys.stdout = old_stdout
        log(buf.getvalue())

        top3, best_move = get_model_top3(model, state)
        rating = rate_move(action, top3)

        log(f"  Your move  : {format_action(action)}")
        if best_move:
            log(f"  Model move : {format_action(best_move)}")
        log(f"  Rating     : {color_rating(rating)}")

    ratings = [rate_move(a, get_model_top3(model, s)[0]) for s, a in history]
    good   = ratings.count("Good")
    normal = ratings.count("Normal")
    bad    = ratings.count("Bad")
    total  = len(ratings)

    log("\n" + "=" * 60)
    log("  FINAL SUMMARY")
    log("=" * 60)
    log(f"  Total moves : {total}")
    log(f"  {Color.GREEN}Good   : {good}  ({good/total*100:.0f}%){Color.RESET}")
    log(f"  {Color.YELLOW}Normal : {normal}  ({normal/total*100:.0f}%){Color.RESET}")
    log(f"  {Color.RED}Bad    : {bad}  ({bad/total*100:.0f}%){Color.RESET}")
    log(f"  Result : {'Won' if winner == 'P2' else 'Lost'}")
    log("=" * 60)

    output_path = os.path.join(PROJECT_ROOT, "analysis.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
    print(f"\nAnalysis saved to analysis.txt — open it in VS Code to review all moves.")


if __name__ == "__main__":
    model = load_model()
    print(f"Device: {DEVICE}")
    print("\nWelcome to Quoridor Game Analyzer!")
    print("Your moves will be analyzed by the AI after the game.\n")

    mode = input("Choose mode - (H) Human or (A) Auto random player: ").strip().upper()
    human_mode = mode != "A"

    if not human_mode:
        print("\nAuto mode: random agent will play as P2.")
    else:
        print("\nHuman mode: you play as P2.")
        print("Commands: Mrow,col to move (e.g. MA,I) | Wrow,col+direction to place wall (e.g. WB,IN)")

    history, winner = play_game(model, human_mode)
    analyze_game(model, history, winner)
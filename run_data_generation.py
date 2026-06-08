"""
plays NUM_GAMES games and saves results to game_data.json.

Each record in the JSON represents one (state, player, action, winner) tuple:
{
  "game_index":   int,   -- which game this came from (0-indexed)
  "step_index":   int,   -- move number within that game
  "player":       int,   -- 1 or 2
  "action":       any,   -- the action taken (serialized)
  "winner":       int,   -- 1 or 2 (eventual winner of this game)
  "state":        any    -- the board state before the action (serialized)
}
"""

import json
import sys
import os
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import signal

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from console.agents.generate_data import run_game_from_start

NUM_GAMES   = 10000
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "game_data.json")
SAVE_EVERY  = 100


def serialize_state(state):
    if state is None:
        return None
    if hasattr(state, "to_dict"):
        return state.to_dict()
    if hasattr(state, "__dict__"):
        try:
            return {k: _make_serializable(v) for k, v in vars(state).items()}
        except Exception:
            pass
    return str(state)


def _make_serializable(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return {k: _make_serializable(v) for k, v in vars(obj).items()}
    return str(obj)


def serialize_action(action):
    """Actions are usually a tuple/list of ints — handle that cleanly."""
    return _make_serializable(action)

def timeout_handler(signum, frame):
    raise TimeoutError("Game took too long")

def run_game_wrapper(_):
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        results = run_game_from_start()
        signal.alarm(0)
        serialized = []
        for step_idx, (state, player, action, winner) in enumerate(results):
            serialized.append({
                "step_index": step_idx,
                "player":     int(player),
                "action":     serialize_action(action),
                "winner":     int(winner),
                "state":      serialize_state(state),
            })
        return serialized
    except Exception:
        return []

def main():
    all_records = []

    print(f"Starting data generation: {NUM_GAMES} games → {OUTPUT_FILE}")
    print("-" * 60)

    with Pool(processes=6) as pool:
        for i, result in enumerate(tqdm(pool.imap(run_game_wrapper, range(NUM_GAMES)), total=NUM_GAMES)):
            try:
                for record in result:
                    record["game_index"] = i
                    all_records.append(record)
            except Exception as exc:
                tqdm.write(f"ERROR: {exc}")

            if (i + 1) % SAVE_EVERY == 0 or (i + 1) == NUM_GAMES:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(all_records, f, indent=2)

    print("-" * 60)
    print(f"Done. {len(all_records)} records saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

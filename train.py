import json
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from console.states.game_state import GameState, BoardPieceStatus
from console.model.quoridor_cnn import QuoridorModel, encodeState

DATA_PATH  = os.path.join(PROJECT_ROOT, "game_data.json")
SAVE_PATH  = os.path.join(PROJECT_ROOT, "quoridor_model.pth")
EPOCHS     = 20
BATCH_SIZE = 64
LR         = 1e-3
VAL_SPLIT  = 0.1

#Action encoding
# 81 pawn positions (9x9) + 64 horizontal walls + 64 vertical walls = 209
ACTION_SPACE = 209

def parse_array(val):
    """Handle both list-of-strings and numpy string formats from JSON."""
    import re
    if isinstance(val, list):
        return np.array([int(x) for x in val], dtype=int)
    return np.array([int(x) for x in re.findall(r'-?\d+', val)], dtype=int)

def action_to_index(action):
    action = [int(x) for x in action]
    if len(action) == 2:
        r, c = action
        return (r // 2) * 9 + (c // 2)
    r1, c1, r2, c2 = action[0], action[1], action[2], action[3]
    if r1 == r2:
        return 81 + ((r1 - 1) // 2) * 8 + (c1 // 2)
    else:
        return 145 + (r1 // 2) * 8 + ((c1 - 1) // 2)

#State reconstruction & encoding
def reconstruct_and_encode(state_dict):
    gs = GameState(initialize=False)
    gs.board                 = parse_array(state_dict["board"]).reshape(17, 17)
    gs.player_one            = state_dict["player_one"]
    gs.player_one_pos        = parse_array(state_dict["player_one_pos"])
    gs.player_two_pos        = parse_array(state_dict["player_two_pos"])
    gs.player_one_walls_num  = state_dict["player_one_walls_num"]
    gs.player_two_wall_num   = state_dict["player_two_wall_num"]
    gs.player_two_walls_num  = gs.player_two_wall_num
    return encodeState(gs)

# Dataset
class QuoridorDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            records = json.load(f)

        self.samples = []
        skipped = 0
        for rec in tqdm(records, desc="Loading dataset"):
            try:
                state_tensor = reconstruct_and_encode(rec["state"])
                action_idx   = action_to_index(rec["action"])
                value_target = 1.0 if rec["player"] == rec["winner"] else -1.0
                self.samples.append((
                    state_tensor,
                    torch.tensor(action_idx,   dtype=torch.long),
                    torch.tensor(value_target, dtype=torch.float32),
                ))
            except Exception:
                skipped += 1

        print(f"  {len(self.samples)} samples loaded, {skipped} skipped")

    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

#Training loop
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    dataset  = QuoridorDataset(DATA_PATH)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = QuoridorModel().to(device)
    model.policy = nn.Linear(256, ACTION_SPACE).to(device)

    optimizer     = torch.optim.Adam(model.parameters(), lr=LR)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn  = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_p = train_v = 0

        for states, actions, values in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            states, actions, values = states.to(device), actions.to(device), values.to(device)

            policy_out, value_out = model(states)
            p_loss = policy_loss_fn(policy_out, actions)
            v_loss = value_loss_fn(value_out.squeeze(), values)
            loss   = p_loss + v_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_p += p_loss.item()
            train_v += v_loss.item()

        # Validation
        model.eval()
        val_p = val_v = 0
        with torch.no_grad():
            for states, actions, values in val_loader:
                states, actions, values = states.to(device), actions.to(device), values.to(device)
                p_out, v_out = model(states)
                val_p += policy_loss_fn(p_out, actions).item()
                val_v += value_loss_fn(v_out.squeeze(), values).item()

        avg_tp = train_p / len(train_loader)
        avg_tv = train_v / len(train_loader)
        avg_vp = val_p   / len(val_loader)
        avg_vv = val_v   / len(val_loader)
        val_total = avg_vp + avg_vv

        print(f"Epoch {epoch:3d} | train policy={avg_tp:.4f} value={avg_tv:.4f} | val policy={avg_vp:.4f} value={avg_vv:.4f}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  → Best model saved (val_loss={val_total:.4f})")

    print(f"\nDone. Model saved to {SAVE_PATH}")

if __name__ == "__main__":
    train()

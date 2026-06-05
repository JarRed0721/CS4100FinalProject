import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import random
from .quoridor_cnn import *


def train_model(model, data, epochs=10, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #efficiency
    model = model.to(device)
        
    loader, val_loader  = validation_data_loader(data, 0.1, batch_size)
        
    #adam optimizer adapts learning rate as you go, which is more optimal (optimizer = alg updating weights)
    optimizer      = torch.optim.Adam(model.parameters(), lr=lr) 
    policy_loss_fn = nn.CrossEntropyLoss() #compares model policy to true action
    value_loss_fn  = nn.MSELoss() #compares predicted winner to actual winner
        
    for epoch in range(epochs): #epochs = # times you go through same data to refine weights
        model.train()  # turns dropout on - prevents model over-relying on single neurons
        total_p_loss = total_v_loss = 0.0 #accumulators to track progress

        for states, actions, value_targets in loader:  # loader auto-batches (batch = section of data)
            states        = states.to(device)
            actions       = actions.to(device)
            value_targets = value_targets.to(device)

            # forward pass
            policy_logits, value_pred = model(states) 

            # compute loss
            p_loss = policy_loss_fn(policy_logits, actions)
            v_loss = value_loss_fn(value_pred.squeeze(1), value_targets)
            loss   = p_loss + v_loss

            # backward pass
            optimizer.zero_grad()   # clear gradients from last batch
            loss.backward()         # compute new gradients
            optimizer.step()        # update weights

            total_p_loss += p_loss.item()
            total_v_loss += v_loss.item()
            
            
        #Next part is validation, determines if model is memorizing or learning
        #Ideal: in print statements, both losses drop together. If train drops but val doesn't, problem
        model.eval()   # turns dropout off
        val_p_loss = val_v_loss = 0.0

        with torch.no_grad():   # no updating weights
            for states, actions, value_targets in val_loader:
                states        = states.to(device)
                actions       = actions.to(device)
                value_targets = value_targets.to(device)

                policy_logits, value_pred = model(states)
                val_p_loss += policy_loss_fn(policy_logits, actions).item()
                val_v_loss += value_loss_fn(value_pred.squeeze(1), value_targets).item()

        n     = len(loader)
        n_val = len(val_loader)
        print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train  policy={total_p_loss/n:.4f}  value={total_v_loss/n:.4f} | "
        f"Val    policy={val_p_loss/n_val:.4f}  value={val_v_loss/n_val:.4f}"
        )
            
def validation_data_loader(data, val_split, batch_size):
    all_games = list(data)
    random.shuffle(all_games)                        # shuffle so val isn't just the last few games

    split = int(len(all_games) * (1 - val_split))   # e.g. 100 games, 0.1 → split at 90

    train_data = all_games[:split]                   # 90 games for training
    val_data   = all_games[split:]                   # 10 games for validation

    loader     = DataLoader(QuoridorDataset(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(QuoridorDataset(val_data),   batch_size=batch_size, shuffle=False)
    return [loader, val_loader]
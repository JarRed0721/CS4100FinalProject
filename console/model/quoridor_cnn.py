from console.states.game_state import BoardPieceStatus
import torch 
import torch.nn as nn
import numpy as np

def encodeState(gameState):
    grid = gameState.board
    game_state_layers = np.zeros((7, 17, 17))
    # layer 1: player 1's pawns
    game_state_layers[0] = (grid == BoardPieceStatus.OCCUPIED_BY_PLAYER_1).astype(np.float32)
    #layer 2: player 2's pawns
    game_state_layers[1] = (grid == BoardPieceStatus.OCCUPIED_BY_PLAYER_2).astype(np.float32)
    #layer 3: occupied wall slots
    game_state_layers[2] = (grid == BoardPieceStatus.OCCUPIED_WALL).astype(np.float32)
    #layer 4: empty wall slots
    game_state_layers[3] = (grid == BoardPieceStatus.FREE_WALL).astype(np.float32)
    #layer 5: player 1's wall count
    game_state_layers[4] = np.full((17, 17),(gameState.player_one_walls_num / 10))
    #layer 6: player 2's wall count
    game_state_layers[5] = np.full((17, 17), (gameState.player_two_walls_num / 10))
    #layer 7: player's turn:
    if (gameState.player_one):
        game_state_layers[6] = np.ones((17, 17))
    else:
        game_state_layers[6] = np.zeros((17,17))
    tensor = torch.FloatTensor(game_state_layers)
    return tensor


    
class QuoridorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reLU = nn.ReLU()
        self.drop = nn.Dropout(0.3) # tune dropout rate depending on how model fits to data
        self.conv1 = nn.Conv2d(7, 32, kernel_size=(3,3), padding=1) # might have to adjust num of out channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1) # might have to adjust num of out channels
        #we might not need third layer?
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1) # might have to adjust num of out channels
        self.pool = nn.MaxPool2d(kernel_size=(3,3))

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3200, 256)

        self.policy = nn.Linear(256, 132)

        self.value_linear1 = nn.Linear(256, 64)
        self.value_activation = nn.ReLU()
        self.value_linear2 = nn.Linear(64, 1)
        self.value_tanh = nn.Tanh() ## normalize output to [-1,1] to match game output labels
    
    def forward(self, x):
        # input = 7x17x17, output = 32x17x17
        x = self.conv1(x)
        x = self.reLU(x)
        x = self.drop(x)
        # input = 32x17x17, output = 64x17x17
        x = self.conv2(x)
        x = self.reLU(x)
        x = self.drop(x)
        # input = 64x17x17, output = 128x17x17
        x = self.conv3(x)
        x = self.reLU(x)
        x = self.drop(x)
        # input = 128x17x17, output = 128x5x5
        x = self.pool(x)
        # input = 128x5x5, output = 3200
        x = self.flatten(x)
        # input = 3200, output = 256
        x = self.linear(x)
        x = self.reLU(x)

        # input = 256, output = 132 (# of moves in Quorridor)
        policy = self.policy(x)

        #input = 256, output = 1
        v = self.value_linear1(x)
        v = self.value_activation(v)
        v = self.value_linear2(v)
        value = self.value_tanh(v)

        return policy, value






    

    

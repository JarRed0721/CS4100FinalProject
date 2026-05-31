from console.states.game_state import BoardPieceStatus
import torch 
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


    

    

    

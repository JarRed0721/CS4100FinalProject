from ..states.game_state import GameState
import random

#takes in a state, and determine a random move to make, then return (state, move, outcome state)

def make_state_tuple_random_move(game: GameState):
    #all moves are formatted output as (copy_state, position) where position seems to be the move
    all_states = game.get_all_child_states(True) #not sure if True is the right move here or not- sets p1 as max player
    idx_chosen = random.randint(0, len(all_states) - 1)
    move_chosen = all_states[idx_chosen]
    return (game, move_chosen[1], move_chosen[0]) # (prev state, move made, copied outcome state)
    
    

#gets state tuple given a specific move
def make_state_tuple_given_move(game: GameState, action):
    outcome = game.execute_action(action)
    outcome.player_one = not outcome.player_one #because execute_action changes it to other player
    
    return (game, action, outcome) # (prev state, move made, copied outcome state)
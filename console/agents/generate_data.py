from console.states.game_state import GameState
from console.agents.given_agents.minimax import minimax
from console.agents.given_agents.minimax_ab_pruning import minimax_alpha_beta_pruning
from console.agents.given_agents.expectimax import expectimax
from console.agents.given_agents.monte_carlo_tree_search import SearchNode
import random
import math

def run_game_from_start():
    return run_simulation_from_state(GameState())

# given a state, will run a random agent on the game until completion
# returns a list of tuples formatted as (prev_state, player num, action, eventual_winner)
def run_simulation_from_state(game: GameState):
    algorithms = [
        "minimax-alpha-beta-pruning",
        "expectimax",
        "monte-carlo-tree-search",
    ]
    p1_alg = algorithms[random.randint(0, len(algorithms) - 1)]
    p2_alg = algorithms[random.randint(0, len(algorithms) - 1)]
    results = []

    while not game.is_end_state():
        if game.player_one:
            action = player_simulation(game, 1, p1_alg)
            results.append((game, 1, action, -1))
        else:
            action = player_simulation(game, 2, p2_alg)
            results.append((game, 2, action, -1))
        game = game.execute_action(action)

    winner = int(game.get_winner()[1:])
    return [
        tuple(winner if (i + 1) % 4 == 0 else elem for i, elem in enumerate(t))
        for t in results
    ]


# finds the action the given player should take
def player_simulation(game_state: GameState, player_number, player_alg):
    if player_number == 1:
        maximizer = True
    else:
        maximizer = False
    action = (0, 0)
    if player_alg == "minimax":
        return minimax_agent(game_state, maximizer, is_alpha_beta=False)
    elif player_alg == "minimax-alpha-beta-pruning":
        return minimax_agent(game_state, maximizer, is_alpha_beta=True)
    elif player_alg == "expectimax":
        return expectimax_agent(game_state, maximizer)
    elif player_alg == "monte-carlo-tree-search":
        start = SearchNode(state=game_state, player_one_maximizer=maximizer)
        selected_node = start.best_action()
        return selected_node.parent_action


def minimax_agent(game_state, player_one_minimax, is_alpha_beta):
    d = {}
    for child in game_state.get_all_child_states(player_one_minimax):
        if not is_alpha_beta:
            value = minimax(
                child[0],
                3,
                maximizing_player=False,
                player_one_minimax=player_one_minimax,
            )
        else:
            value = minimax_alpha_beta_pruning(
                child[0],
                3,
                -math.inf,
                math.inf,
                maximizing_player=False,
                player_one_minimax=player_one_minimax,
            )
        d[value] = child
    return choose_action(d)


def expectimax_agent(game_state, player_one_maximizer):
    d = {}
    for child in game_state.get_all_child_states(player_one_maximizer):
        value = expectimax(child[0], 2, False, player_one_maximizer)
        d[value] = child
    return choose_action(d)


def choose_action(d):
    if len(d.keys()) == 0:
        return None
    k = max(d)
    winner = d[k]
    action = winner[1]

    return action

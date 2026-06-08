from console.states.game_state import GameState, Mappings
from console.util.wall_direction import WallDirection
from console.util.color import Color


class Game:
    def __init__(self):
        self.game_state = GameState()
        Game.print_colored_output("### WELCOME TO QUORIDOR ###", Color.CYAN)
        print("\nCommands [case insensitive]:")
        self.print_commands()
        print("{0:-<100}".format(""))

    def print_commands(self):
        print("1. Move your piece:" + Color.CYAN + " mx,y " + Color.RESET + "where x is the row letter and y is the column letter")
        print("2. Place a wall:   " + Color.CYAN + " wx,yd " + Color.RESET + "where d is one of [S, N, E, W]")
        print("3. Press          " + Color.CYAN + " x " + Color.RESET + " to exit the game.")

    def player_input(self, player_number):
        while True:
            value = input(f"Player {player_number} - Enter move: ")
            if value == "x" or value == "X":
                exit(0)
            elif value.lower() == "help":
                print()
                self.print_commands()
                print()
            elif value.upper().startswith("M"):
                parts = value[1:].split(",")
                if len(parts) != 2:
                    Game.print_colored_output("Illegal move!", Color.RED)
                    continue
                x_string, y_string = parts
                if x_string.upper() not in Mappings.INPUT_MAPPINGS or y_string.upper() not in Mappings.INPUT_MAPPINGS:
                    Game.print_colored_output("Illegal move!", Color.RED)
                else:
                    move = (Mappings.INPUT_MAPPINGS[x_string.upper()], Mappings.INPUT_MAPPINGS[y_string.upper()])
                    if move not in self.game_state.get_available_moves(False):
                        Game.print_colored_output("Illegal move!", Color.RED)
                    else:
                        self.game_state.move_piece(move)
                        break
            elif value.upper().startswith("W"):
                parts = value[1:len(value) - 1].split(",")
                if len(parts) != 2:
                    Game.print_colored_output("Illegal wall placement!", Color.RED)
                    continue
                x_string, y_string = parts
                if x_string.upper() not in Mappings.INPUT_MAPPINGS or y_string.upper() not in Mappings.INPUT_MAPPINGS:
                    Game.print_colored_output("Illegal wall placement!", Color.RED)
                else:
                    dir_string = value[-1].upper()
                    if dir_string not in ["N", "S", "E", "W"]:
                        Game.print_colored_output("Illegal wall direction!", Color.RED)
                    else:
                        direction = {"S": WallDirection.SOUTH, "N": WallDirection.NORTH,
                                     "E": WallDirection.EAST, "W": WallDirection.WEST}[dir_string]
                        x_int = Mappings.INPUT_MAPPINGS[x_string.upper()]
                        y_int = Mappings.INPUT_MAPPINGS[y_string.upper()]
                        is_valid, coords = self.game_state.check_wall_placement((x_int, y_int), direction)
                        if not is_valid:
                            Game.print_colored_output("Illegal wall placement!", Color.RED)
                        else:
                            self.game_state.place_wall(coords)
                            break
            else:
                Game.print_colored_output("Illegal command!", Color.RED)

    def check_end_state(self):
        if self.game_state.is_end_state():
            winner = self.game_state.get_winner()
            Game.print_colored_output(f"The winner is {winner}!", Color.GREEN)
            return True
        return False

    def play(self):
        while True:
            print()
            self.game_state.print_game_stats()
            print()
            self.game_state.print_board()
            print()

            if self.check_end_state():
                break

            player_number = 1 if self.game_state.player_one else 2
            self.player_input(player_number)
            self.game_state.player_one = not self.game_state.player_one

    @staticmethod
    def print_colored_output(text, color):
        print(color + text + Color.RESET)

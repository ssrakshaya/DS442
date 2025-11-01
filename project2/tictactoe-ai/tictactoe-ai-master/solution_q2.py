import sys
from typing import List, Tuple, Optional
from copy import deepcopy

#these stay constant 
GRID_SIZE = 6
TEAM_ALPHA = 'Alpha' #alpha in minimax  
TEAM_BRAVO = 'Bravo' #beta in minimax
EMPTY = 'Empty' #is the board empty or not

# Move types
STRATEGIC_DEPLOYMENT = 'Strategic Deployment'
TACTICAL_ASSAULT = 'Tactical Assault'

class Game: 
    #the class represents what state the game is at 

    def __init__(self, tile_values: List[List[int]]):
        """
        Initialize game state with tile values
        
        Args:
            tile_values: 6x6 grid of tile values
        """
        self.tile_values = tile_values

        #Tracking which player is the alpha or beta or empty, so you start everything with an empty
        self.board = []  # start with empty board

        for r in range(GRID_SIZE):        # for each row
            row = []
            for c in range(GRID_SIZE):    # for each column
                row.append(EMPTY)
            self.board.append(row)

        self.alpha_score = 0
        self.bravo_score = 0
        self.move_count = 0


def read_input_file(filename = 'input.txt'): #-> List[List[int]]
    """
    Reading in the initial state from input.txt
    Returns the grid of tile values
    """
    grid = [] #grid to hold the 6*6 list
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file] #reading into lines
        
        for line in lines:
            #split each line by commas into strings with numbers
            parts = line.split(",")

            #convert each string to an integer
            row = []
            for x in parts:
                row.append(int(x.strip()))
            grid.append(row)
        
    return grid

        
def main():
    """Main entry point"""
    # Read input file
    tile_values = read_input_file() #the method reads in input.txt automatically 
    
    # Play the game with Minimax AI
    # Using depth 3 for reasonable performance
    # Increase depth for stronger play but slower execution
    play_game(tile_values, max_depth=3)


if __name__ == "__main__":
    main()
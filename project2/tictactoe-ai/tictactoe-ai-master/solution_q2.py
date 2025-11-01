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
    
    def copy(self):
        #making a deep copy of the game
        #the deep copy method allows me to create a brand new copy of every nested object in the game
        #if not, when using minimax, you would just keep overwriting a current board. 

        new_state = GameState(self.tile_values)
        new_state.board = deepcopy(self.board)
        new_state.alpha_score = self.alpha_score
        new_state.bravo_score = self.bravo_score
        new_state.move_count = self.move_count
        return new_state


    def is_terminal(self):
        """Check if game is over (all tiles occupied)"""
        for row in self.board:
            for cell in row:
                if cell == EMPTY:
                    return False
        return True

    def get_current_team(self):
        
        #getting what team's turn it is
        #self.move_count % 2 == 0 -> Alpha’s turn (even move count)
        # self.move_count % 2 == 1 -> Bravo’s turn (odd move count)

        if self.move_count % 2 == 0:
            return TEAM_ALPHA
        else:
            return TEAM_BRAVO
    
    def get_opponent_team(self, team: str):
        """Get the opponent of the given team"""
        #given the name of one team, it will give you the other team's name 
        #(alternating from current team to next)
        if team == TEAM_ALPHA:
            return TEAM_BRAVO
        else:
            return TEAM_ALPHA
    
    def is_valid_position(self, row: int, col: int):
        #Check if position is within the grid 
        return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE
    

    def get_adjacent_positions(self, row: int, col: int):
        
        #getting all the orthogonal postiions (left right up down)
        #orthogonal is the only way the players can move! 
        # (-1, 0) move up
        # (1, 0) move down
        # (0, -1) move left
        # (0, 1) move right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right, -1 stands for the direction you have moved 
        adjacent = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc #Compute the neighbor coordinates
            if self.is_valid_position(new_row, new_col): #making sure coordiantes are valid
                adjacent.append((new_row, new_col)) #If valid, add (new_row, new_col) to the list.
        return adjacent #return list of adjacent cells

    
    def get_legal_moves(self, team: str):
        #returning  a tuple of strings
        #This method figures out every possible move that a given team (Alpha or Bravo) 
        # can make right now, based on the current board.

        """
        Get all legal moves for a team
        
        Returns:
            List of moves as (move_type, position, from_position)
            - Strategic Deployment: (STRATEGIC_DEPLOYMENT, (row, col), None)
            - Tactical Assault: (TACTICAL_ASSAULT, (to_row, to_col), (from_row, from_col))
        """
        moves = [] #empty list, and will add the deployment and assault moves into this
        
        # Strategic Deployment: any unoccupied tile
        #this scans the entire input text board, for every position:
        #if there are empty tiles, the team can deploy another unit there, 
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.board[row][col] == EMPTY:
                    moves.append((STRATEGIC_DEPLOYMENT, (row, col), None)) #appending a tuple for deploying another unit
        
        #Tactical Assault happens from any controlled tile to adjacent unoccupied tile
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE): 
                if self.board[row][col] == team: #if the tile is controlled by curent tema, look at neighboring tiles 
                    #Moving to each adjacent unoccupied tile
                    for adj_row, adj_col in self.get_adjacent_positions(row, col): #calling the adjacent positions method
                        if self.board[adj_row][adj_col] == EMPTY: #if all are empty
                            moves.append((TACTICAL_ASSAULT, (adj_row, adj_col), (row, col))) #append the tactical move to it
        
        return moves

    def apply_move(self, move: Tuple[str, Tuple, Optional[Tuple]], team: str):
        #the method applies either a deployment or assult to the current game state
        #this updates what tiles are owned by what team
        #it also updated the total score of the team

        #move type houses "strategic deployment" or "tactical assault", position has the row and col
        move_type, position, from_position = move
        row, col = position
        opponent = self.get_opponent_team(team) #finds the opposint team 
        
        if move_type == STRATEGIC_DEPLOYMENT:
            #getting control of the tile
            self.board[row][col] = team #the team places a unit of workers onto the tile
            tile_value = self.tile_values[row][col] #find how many points the tile is
            if team == TEAM_ALPHA:
                self.alpha_score += tile_value #add the points to the correct team! 
            else:
                self.bravo_score += tile_value
                
        elif move_type == TACTICAL_ASSAULT:
            #Occupy the new tile
            self.board[row][col] = team
            tile_value = self.tile_values[row][col] #you gain the new tile as well and the original! 
            if team == TEAM_ALPHA:
                self.alpha_score += tile_value
            else:
                self.bravo_score += tile_value
            
            #Getting adjacent enemy tiles
            for adj_row, adj_col in self.get_adjacent_positions(row, col): #looking at all four adjacent tiles
                if self.board[adj_row][adj_col] == opponent:
                    #changing the enemy tile to our own tiles
                    self.board[adj_row][adj_col] = team
                    captured_value = self.tile_values[adj_row][adj_col]
                    
                    #Increasing scores based on whose tiles got captured
                    if team == TEAM_ALPHA:
                        self.alpha_score += captured_value
                        self.bravo_score -= captured_value
                    else:
                        self.bravo_score += captured_value
                        self.alpha_score -= captured_value
        
        self.move_count += 1 #updates the move count, so it will saywhose turn it is next, alpha to beta, or vice versa

    def evaluate(self):
        #doing alpha beta evaluation! 
        #the function gives an int score for how good the board is based on alpha's values

        return self.alpha_score - self.bravo_score

    def format_board(self):
        #making the board pretty
        lines = []
        for row in range(GRID_SIZE): #looping thrugh every ro
            row_values = []
            for col in range(GRID_SIZE):
                value = self.tile_values[row][col] #point value of tile from the input.txt 
                owner = self.board[row][col] #who controls the square - alpha, beta, or no one? 
                
                #show value of tile and who owns it
                if owner == TEAM_ALPHA:
                    row_values.append(f"A{value:2d}")
                elif owner == TEAM_BRAVO:
                    row_values.append(f"B{value:2d}")
                else:
                    row_values.append(f" {value:2d}")
            
            #join the row into one string 
            row_string = " ".join(row_values)
            # Add that string to the list of lines
            lines.append(row_string)

        #join all the row strings into the long board
        board_string = "\n".join(lines)

        return board_string
    

    def format_move_description(self, move: Tuple[str, Tuple, Optional[Tuple]]):
        #formatting the description of where eah player is moving 

        move_type, position, from_position = move #move_type is either 'Strategic Deployment' or 'Tactical Assault' and position is the destination tile (a tuple (row, col))
        row, col = position
        
        # Convert to letter-number notation (A-F, 1-6)
        col_letter = chr(ord('A') + col)
        row_number = row + 1
        
        if move_type == STRATEGIC_DEPLOYMENT:
            return f"Strategic Deployment at [{col_letter},{row_number}]"
        else:
            from_row, from_col = from_position
            from_col_letter = chr(ord('A') + from_col)
            from_row_number = from_row + 1
            return f"Tactical Assault from [{from_col_letter},{from_row_number}] to [{col_letter},{row_number}]"




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
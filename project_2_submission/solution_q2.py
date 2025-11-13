import sys
from typing import List, Tuple, Optional
from copy import deepcopy

#these stay constant 
GRID_SIZE = 6
TEAM_ALPHA = 'Alpha' #alpha in minimax  
TEAM_BETA = 'Beta' #beta in minimax
EMPTY = 'Empty' #is the board empty or not

# Move types
STRATEGIC_DEPLOYMENT = 'Strategic Deployment'
TACTICAL_ASSAULT = 'Tactical Assault'

class GameState: 
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
        self.beta_score = 0
        self.move_count = 0
    
    def copy(self):
        #making a deep copy of the game
        #the deep copy method allows me to create a brand new copy of every nested object in the game
        #if not, when using minimax, you would just keep overwriting a current board. 

        new_state = GameState(self.tile_values)
        new_state.board = deepcopy(self.board)
        new_state.alpha_score = self.alpha_score
        new_state.beta_score = self.beta_score
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
        # self.move_count % 2 == 1 -> beta’s turn (odd move count)

        if self.move_count % 2 == 0:
            return TEAM_ALPHA
        else:
            return TEAM_BETA
    
    def get_opponent_team(self, team: str):
        """Get the opponent of the given team"""
        #given the name of one team, it will give you the other team's name 
        #(alternating from current team to next)
        if team == TEAM_ALPHA:
            return TEAM_BETA
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
        #This method figures out every possible move that a given team (Alpha or Beta) 
        # can make right now, based on the current board.

        
        #Get all legal moves for a team
        #What it returns! 
        #  List of moves as (move_type, position, from_position)
        #  Strategic Deployment: (STRATEGIC_DEPLOYMENT, (row, col), None)
        #  Tactical Assault: (TACTICAL_ASSAULT, (to_row, to_col), (from_row, from_col))
      
        moves = [] #empty list, and will add the deployment and assault moves into this
        deployments = []
        capturing_assaults = []
        noncapturing_assaults = []
        
        # Strategic Deployment: any unoccupied tile
        #this scans the entire input text board, for every position:
        #if there are empty tiles, the team can deploy another unit there, 
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.board[row][col] == EMPTY:
                    deployments.append((STRATEGIC_DEPLOYMENT, (row, col), None)) #appending a tuple for deploying another unit

        #Tactical Assault happens from any controlled tile to adjacent unoccupied tile
        opponent = self.get_opponent_team(team)
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.board[row][col] == team:
                    #moving to any tiles that are next to us and unoccupied
                    for adj_row, adj_col in self.get_adjacent_positions(row, col):
                        if self.board[adj_row][adj_col] == EMPTY:
                            #Check if this move would capture any enemy tiles
                            #meaning, is the destination adjacent to any enemy?)
                            will_capture = any(
                                self.board[ar][ac] == opponent
                                for (ar, ac) in self.get_adjacent_positions(adj_row, adj_col)
                            )
                            
                            if will_capture:
                                capturing_assaults.append((TACTICAL_ASSAULT, (adj_row, adj_col), (row, col)))
                            else:
                                noncapturing_assaults.append((TACTICAL_ASSAULT, (adj_row, adj_col), (row, col)))
        
        #If capturing assaults exist, only include those and the deployments
        #If no capturing assults, include all assaults and deployments
        if capturing_assaults:
            return deployments + capturing_assaults
        else:
            return deployments + noncapturing_assaults
    


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
                self.beta_score += tile_value
                
        elif move_type == TACTICAL_ASSAULT:
            #Occupy the new tile
            self.board[row][col] = team
            tile_value = self.tile_values[row][col] #you gain the new tile as well and the original! 
            if team == TEAM_ALPHA:
                self.alpha_score += tile_value
            else:
                self.beta_score += tile_value
            
            #Getting adjacent enemy tiles
            for adj_row, adj_col in self.get_adjacent_positions(row, col): #looking at all four adjacent tiles
                if self.board[adj_row][adj_col] == opponent:
                    #changing the enemy tile to our own tiles
                    self.board[adj_row][adj_col] = team
                    captured_value = self.tile_values[adj_row][adj_col]
                    
                    #Increasing scores based on whose tiles got captured
                    if team == TEAM_ALPHA:
                        self.alpha_score += captured_value
                        self.beta_score -= captured_value
                    else:
                        self.beta_score += captured_value
                        self.alpha_score -= captured_value
        
        self.move_count += 1 #updates the move count, so it will saywhose turn it is next, alpha to beta, or vice versa

    def evaluate(self):
        #doing alpha beta evaluation! 
        #the function gives an int score for how good the board is based on alpha's values

        return self.alpha_score - self.beta_score

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
                elif owner == TEAM_BETA:
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
        
        #converting the numeric coordiantes to letters of where they are
        col_letter = chr(ord('A') + col)
        row_number = row + 1
        
        if move_type == STRATEGIC_DEPLOYMENT: #there is no from tile, just wher you end up
            return f"Strategic Deployment at [{col_letter},{row_number}]"
        else:
            from_row, from_col = from_position #where did we start from 
            from_col_letter = chr(ord('A') + from_col) #making it characters that are easy to understand
            from_row_number = from_row + 1
            return f"Tactical Assault from [{from_col_letter},{from_row_number}] to [{col_letter},{row_number}]"

class MinimaxAgent:
    #minimax controls the territory (we use minimax algorithm to make this work! )
    def __init__(self, max_depth: int = 3):
        
        #initialize the minimax agent
        #max_depth: Maximum search depth for minimax
        
        self.max_depth = max_depth
        self.nodes_explored = 0

    def get_best_move(self, state: GameState, team: str): #-> Tuple[str, Tuple, Optional[Tuple]]

        #given the curent game state (with the boards and its scores) and which player (alpha or beta) is moving
        #figuring out the Get the best move using Minimax algorithm
    
        self.nodes_explored = 0 #Resets the counter that tracks how many game states the algorithm will evaluate during this search.
        
        # Determine if we're maximizing or minimizing (alpha maximizies, and beta minimizes)
        is_maximizing = (team == TEAM_ALPHA)
        
        best_move = None #stores the value that currently looks the best
        if is_maximizing:
            best_value = float('-inf') #alpha is initialized to negative infinity in alpha beta pruning
        else:
            best_value = float('inf') #beta is initialized to positive infinity because it is trying to score the lowest 
        
        #Try all legal moves, calling the method that makes sure the moves are possible
        legal_moves = state.get_legal_moves(team)
        
        if not legal_moves:
            return None #the team cannot move because there are no good legal moves (perhaps running out of spaces)
        
        for move in legal_moves: #loop through all possible moves
            #Create new state and apply move
            new_state = state.copy() #make a deep copy of the state so that we dont modify the actual board
            new_state.apply_move(move, team) #Apply that move to the copy, producing a new possible board state.
            
            #Get value of this move
            if is_maximizing:
                value = self.minimax(new_state, self.max_depth - 1, float('-inf'), float('inf'), False)
                if value > best_value: #if the current value is higher
                    best_value = value #change alphas value to the larger one
                    best_move = move #change the best move as well
            else:
                value = self.minimax(new_state, self.max_depth - 1, float('-inf'), float('inf'), True)
                if value < best_value: #if new beta value is smaller
                    best_value = value #change the beta value to the smaller one (minimizing agent)
                    best_move = move #change move too
        
        return best_move #best move found by minimax
    
    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, is_maximizing: bool):
        
        #Minimax algorithm with alpha-beta pruning
        
        #Arguments:
        #    state: Current game state
        #    depth: Remaining search depth
        #   alpha: Alpha value for pruning
        #  beta: Beta value for pruning
        # is_maximizing: True if maximizing player (Alpha), False if minimizing (Beta)
        
        #Returns: Evaluation score of the state
        
        #this is where the code decides what moves to takes, and uses alpha beta pruning

        self.nodes_explored += 1 #look at how many nodes have been exploreed
        
        #Base case: terminal state or depth limit reached
        if depth == 0 or state.is_terminal(): #where recursion stops,
            return state.evaluate() #return alpha-beta
        
        if is_maximizing:
            #Alpha's turn (maximizing agent)
            max_eval = float('-inf') #initialized to negative inf (lowest possible number)
            legal_moves = state.get_legal_moves(TEAM_ALPHA)
            
            for move in legal_moves: #exploring all the possible alpha moves

                new_state = state.copy() #Copy the board so we don’t modify the real one.
                new_state.apply_move(move, TEAM_ALPHA) #do the move
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, False) #call minimax recursively, and depth increase by 1
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score) #keeping track of highest value Alpha 
                if beta <= alpha: #beta pruning here, because beta is smaller than alpha
                    break  # Beta cutoff
            
            return max_eval
        else:
            # Beta's turn (minimizing)
            min_eval = float('inf')
            legal_moves = state.get_legal_moves(TEAM_BETA) #beta wnats to minimize the value (to make alpha as small as possible)
            
            for move in legal_moves: #explore all the possible beta moves
                new_state = state.copy() #apply the beta move on a copt of the board
                new_state.apply_move(move, TEAM_BETA)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, True) #recursively call minimax
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  #Alpha cutoff
            
            return min_eval #return the beta score that is the best


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

    if len(grid) != 6 or any(len(row) != 6 for row in grid):
        raise ValueError("input.txt must be a 6x6 grid of integers")
        
    return grid

def play_game(tile_values: List[List[int]], max_depth: int = 3):
    #playing the whole game 
    #take in the tile values and the max depth 

    # Initialize game state and AI agent
    state = GameState(tile_values)
    agent = MinimaxAgent(max_depth=max_depth)
    
    print("=" * 60)
    print("TERRITORY CONTROL GAME - MINIMAX AI")
    print("=" * 60)
    print()
    
    #Play until game is over
    while not state.is_terminal():
        current_team = state.get_current_team() #using method to grab best team
        
        #Get best move from Minimax
        best_move = agent.get_best_move(state, current_team)
        
        if best_move is None:
            print(f"No legal moves available for {current_team}")
            break
        
        # Apply the move
        state.apply_move(best_move, current_team)
        
        # Print move information
        print(f"Move {state.move_count}:")
        print(f"Minimax action for team: {current_team}")
        print(f"Action: {state.format_move_description(best_move)}")
        print()
        print(state.format_board())
        print()
        print(f"Total score - Alpha: {state.alpha_score}")
        print(f"Total score - Beta: {state.beta_score}")
        print("-" * 60)
        print()
    
    #Game over - announce winner
    print("=" * 60)
    print("GAME OVER")
    print("=" * 60)
    if state.alpha_score > state.beta_score: #if alpha scores higher than beta
        print(f"The Winner is: {TEAM_ALPHA}") #dynamically printing alphas win
    elif state.beta_score > state.alpha_score:
        print(f"The Winner is: {TEAM_BETA}") #printing betaos win
    else:
        print("The game is a TIE!")
    
    print(f"Final Score - Alpha: {state.alpha_score}, Beta: {state.beta_score}")
    print("=" * 60)


        
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
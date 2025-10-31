from os import curdir
from typing import Optional
from enum import Enum

Square = int


class Symbol(Enum):
    CIRCLE = "O"
    CROSS = "X"
    EMPTY = "#"


class Board:
    """Responsible for storing current state, making and validating moves, and updating the game

    Attributes:
        first_move (Symbol): Symbol to make the first move, alternate after every match
        p1_score (int): First player's score
        p2_score (int): Second player's score
        size (int): Board dimension     
        squares (dict[[int,int],Square]): Container to convert row,col into Square
        table (list[Symbol]): Container to store the current board state, use Square as index 
        turn (Symbol): Current turn to make a move
        win_conditions (list[list[Square]]): All possible connections to win the game
        in_response_mode (bool): - Tracks if we're in response mode
        get_first_three (Optional[Symbol]): - Who made the first 3-in-a-row
    """

    def __init__(self, size: int = 3):
        """Creating a board with given size

        Args:
            size (int, optional): Board dimension
        """
        self.size: int = size #board size
        self.p1_score: int = 0 #score for the first player (perhaps AI)
        self.p2_score: int = 0 #score for the second played

        self.squares: dict[[int, int], Square] = self.get_squares() #dictionary of squares convert row,col into Square
        self.table: list[Symbol] = self.get_table() #stores the current board state, use Square as index
        self.win_conditions: list[list[Square]] = self.get_win_conditions() #All possible connections to win the game

        self.first_move: Symbol = Symbol.CIRCLE #first move is the symbol for the Ai
        self.turn: Symbol = self.first_move #turn is the current turn

        #the Board TRACKS if we are in response mode (one person made 3 in a row, the other player gets a MOVE)
        self.in_response_mode: bool = False #if True, the board will print the response mode (for debugging)   
        #TRACK who made the first 3 in a row, so we know who wins if resposnse fails
        self.get_first_three: Optional[Symbol] = None #who made the first 3 in a row
        

    def get_win_conditions(self) -> list[list[Square]]:
        """Get all winning connections, for all board sizes

        Returns:
            list[list[Square]]: list of rows, cols, diagonals 
        """
        rows, cols = self.get_rows_cols() 
        diagonals = self.get_diagonals()
        return rows + cols + diagonals

    def get_squares(self) -> dict[[int, int], Square]:
        """Create a dictionary containing all squares

        Returns:
            dict[[int, int], Square]: (row,col) as key, square name as value
        """
        return {(r, c): r * self.size + c
                for r in range(self.size) for c in range(self.size)}

    def get_table(self) -> list[Symbol]:
        """Table to store the current board state

        Returns:
            list[Symbol]: List of tiles filled with empty Symbol
        """
        return [Symbol.EMPTY for _ in range(self.size**2)]

    def get_rows_cols(self) -> tuple[list[Square], list[Square]]:
        """Group squares into corresponding rows and columns

        Returns:
            tuple[list[Square], list[Square]]: lists of rows and cols
        """
        rows: list[list[Square]] = [[] for _ in range(self.size)]
        columns: list[list[Square]] = [[] for _ in range(self.size)]
        for index, square in self.squares.items():
            r, c = index
            rows[r].append(square)
            columns[c].append(square)
        return rows, columns

    def get_diagonals(self) -> list[list[Square]]:
        """Calculate diagonal squares for all board sizes

        Returns:
            list[list[Square]]: list of diagonals
        """
        diagonals: list[list] = [[], []]
        i = 0
        j = self.size - 1
        for _ in range(self.size):
            diagonals[0].append(i)
            diagonals[1].append(j)
            i += self.size + 1
            j += self.size - 1
        return diagonals

    @property
    def empty_squares(self) -> list[Square]:
        """Get all empty squares

        Returns:
            list[Square]: list of empty squres
        """
        return [
            square for square in self.squares.values() if self.is_empty(square) #list of empty squares
        ]

    def reset(self):
        """Reset the board and change the turn
        """
        self.table = self.get_table()
        self.first_move = Symbol.CROSS if self.first_move == Symbol.CIRCLE else Symbol.CIRCLE
        self.turn = self.first_move

    def square_pos(self, square: Square) -> Optional[tuple[int, int]]:
        """Get row, col of the square

        Args:
            square (Square): Square number

        Returns:
            Optional[tuple[int, int]]: (row, col) if square exists
        """
        for pos, sq in self.squares.items(): #pos is the row and col, sq is the square number
            if sq == square: #if the square number is the same as the square number passed in, then return the row and col
                return pos
        return None #if the square number is not found, then return None

    def square_name(self, row: int, col: int) -> Square:
        """Convert row, col into square

        Returns:
            Square: corresponding number
        """
        return self.squares[(row, col)] #return the square number for the given row and col

    def square_value(self, square: Square) -> Symbol:
        """Get the symbol of the square

        Args:
            square (Square): Square name

        Returns:
            Symbol: Symbol of the square
        """
        return self.table[square]

    def is_empty(self, square: Square) -> bool:
        """Check if square is empty

        Args:
            square (Square): square name

        Returns:
            bool: True if empty symbol
        """
        return self.table[square] == Symbol.EMPTY

    def get_connection(self) -> list[Square]:
        """Check for connected tiles

        Returns:
            list[Square]: List of connected squares
        """
        for row in self.win_conditions:
            checklist = []
            for square in row:
                if self.is_empty(square):
                    continue
                checklist.append(self.square_value(square))
            if len(checklist) == self.size and len(set(checklist)) == 1:
                return row
        return []

    def is_draw(self) -> bool:
        """Check for draw

        Returns:
            bool: True if board is filled and no connection
        """
        if len(self.empty_squares) == 0 and len(self.get_connection()) == 0:
            return True
        return False

    def winner(self) -> Optional[Symbol]:
        """Get the winner of the match

        Returns:
            Optional[Symbol]: Symbol of connected tiles if exists

            In wild tic-tac-toe:
            - If someone makes 3 in a row first, opponent gets to move n response (if they get that, they would win)
            - Winner is only determined after response move is finished (so if the next player gets a 3 in a row, they win)
        """
        connection = self.get_first_three()

        #if there are no 3 in a rows: so the length of connection is zero
        if len(connection) == 0:
            #check if we are in the response mode, because the responder failed to get 3 in a row
            if self.in_response_mode:
                return self.get_first_three #whoever got the three in a row first 
            return None

        #a playerA  with three in a row
        player_with_connection = self.square_value(connection[0])

        #when in response mode, the second player successfully gets three in a row
        if self.in_response_mode:
            return player_with_connection

        #the first player got a three in a row, but there is no winner
        return None


        #if there are no 3 in a rows: so the length of connection is zero
        # if len(connection) == 0:
        #     return None
        # elif self.square_value(connection[0]) == Symbol.CIRCLE:
        #     return Symbol.CIRCLE
        # else:
        #     return Symbol.CROSS

    def is_gameover(self) -> bool:
        """Check for gameover

        Returns:
            bool: True if there's winner or draw
        
        - i have added a check for in_response_mode and when the board has no more spots to fill, the game ends cause the next player cant make any other moves
        - edge case of board full so no more responses possible
        """

        if self.winner() is not None:
            return True
        
        if self.is_draw():
            return True

        #case for when full board
        if self.in_response_mode and len(self.empty_squares) == 0:
            return True
        
        return False
        
        #return self.winner() is not None or self.is_draw()

    def _update(self):
        """Update the turn and score if there's winner
        """

        # Check if the CURRENT player (who just moved) made 3-in-a-row
        first_three = self.get_first_three()

        #First Case: Someone received a three in a row
        if len(first_three) > 0:
            current_player = self.turn #whoever just made the turn becomes the current player

            #First Situation - if th first player has made 3 in a row, and player B ALSO made 3 in a row 
            if self.in_response_mode:
                #the responder made a three in a row and won, meaning the responder is the winner, and game is over
                self.in_response_mode = False #exit the response made, because game isover
            # A player just made three in a row
            else:
                #go back to response mode
                self.in_response_mode = True

                self.get_first_three = current_player #remember who got first three

                #switching turns so now player 2 gets their move (e=must get 3 in a row to win)
                if self.turn == Symbol.CIRCLE: #switching o to x
                    self.turn = Symbol.CROSS
                else:
                    self.turn = Symbol.CIRCLE #switching x's to o's
        
        #Second Case: NO three in a row has been made at this turn
        else:
            #in response mode (the other player has to make move) but the responder did not get three in a row
            if self.in_response_mode:
                #the player A wins because they made three in a row, so you leave response mode
                self.in_response_mode = False
            #no three in a row at all
            else:
                #switch players normally:
                if self.turn == Symbol.CIRCLE: #switching o to x
                    self.turn = Symbol.CROSS
                else:
                    self.turn = Symbol.CIRCLE #switching x's to o's
    
        #updating scores based on who has won the game or not
        winner_symbol = self.winner()
        if winner_symbol == Symbol.CIRCLE:
            self.p1_score += 1
        elif winner_symbol == Symbol.CROSS:
            self.p2_score += 1


        # self.turn = Symbol.CROSS if self.turn == Symbol.CIRCLE else Symbol.CIRCLE #alternate the turn
        # if self.winner() == Symbol.CIRCLE:
        #     self.p1_score += 1 #increment the score for the Ai
        # elif self.winner() == Symbol.CROSS:
        #     self.p2_score += 1 #increment the score for the opponent

    def push(self, square: Square, value: Symbol):
        """Store the symbol into the square

        Args:
            square (Square): square name
            value (Symbol): symbol
        """

        prev_state = {
            'turn': self.turn,
            'in_response_mode': self.in_response_mode,
            'get_first_three': self.get_first_three
        }

        #updating the state 
        self.table[square] = value #putting the symbol into a certain square
        
        return prev_state


    def undo(self, square: Square, prev_state: dict):
        """Change the square's value to empty

        Args:
            square (Square): square name
        """
        self.table[square] = Symbol.EMPTY #clearing the board

        #restore the previous state
        self.turn = prev_state['turn']
        self.in_response_mode = prev_state['in_response_mode']
        self.get_first_three = prev_state['first_three_connections_player']


    def move(self, square: Square):
        """Mark the square with symbol of current turn if valid and update the board

        Args:
            square (Square): square name
        """
        #Current logic, 1. place symbole, 2. update turn, 3. check winner
        if square >= self.size**2 or square < 0 or not self.is_empty(square):
            print("Invalid move!")
            return
        
        self.push(square, self.turn) #use push to update the board internally
        
        #self._update()

    def print(self):
        """Represent the board in string
        """
        turn = "Player 1" if self.turn == Symbol.CIRCLE else "Player 2"
        if self.winner():
            print("Match Over!")
            print("*" * 13)
        else:
            print("*" * 15)
            print("Turn->> ", turn)
            print('-' * (self.size * 5))
        for index, square in self.squares.items():
            r, c = index
            sign = square if self.is_empty(
                square) else "O" if self.square_value(
                    square) == Symbol.CIRCLE else "X"
            print(' |', end=' ')
            print(sign, end='')
            if c == self.size - 1:
                print(' |')
                print('-' * (self.size * 5))
        print('-' * (self.size * 5))
        print()


if __name__ == "__main__":
    # CLI game for two player mode
    board = Board()
    print("Tic Tac Toe - Duel")
    print("##################")
    board.print()
    running = True
    while running:
        turn = "Player 1" if board.turn == Symbol.CIRCLE else "Player 2"
        move = int(input(f"Enter {turn} 's move: "))
        board.move(move)
        board.print()
        if board.is_gameover():
            running = False
    if board.is_draw():
        print("Draw! What a great match!")
    else:
        print("Player 1" if board.winner() == Symbol.CIRCLE else "Player 2",
              " Wins....!")

from tictactoe.board import Board
import random

Square = int
Score = int


class Engine:

    def __init__(self, ai: str, foe: str, level: int):
        self.ai = ai
        self.foe = foe
        self.level = level

    def minimax(self, board: Board, ai_turn: bool, depth: int, alpha: float, #alpha is the best score for the Ai, beta is the best score for the opponent
                beta: float) -> tuple: #tuple is the return value of the function

        available_moves = board.empty_squares #list of empty squares

        #random first move
        if len(available_moves) == board.size**2: #if there are no empty squares, then return a random move
            return 0, random.choice(list(range(board.size**2)))

        #ending state of the board
        if board.is_gameover() or depth >= self.level: # if the game is over or the depth is greater than the level, then return the score and the best move
            return self.evaluate_board(board, depth), None

        if ai_turn:
            #maximimsing player
            max_eval = float('-inf') #max_eval is the best score for the Ai 
            best_move = None 
            for move in available_moves:
                # board.push(move, self.ai) #push the move to the board
                # eval_ = self.minimax(board, False, depth + 1, alpha, beta)[0] #evaluate the board
                # board.undo(move)

                # CHANGED: Capture previous state
                prev_state = board.push(move, self.ai)
                eval_ = self.minimax(board, False, depth + 1, alpha, beta)[0]
                # CHANGED: Restore previous state
                board.undo(move, prev_state)

                #Track the best move
                if eval_ > max_eval:
                    max_eval = eval_
                    best_move = move

                
                alpha = max(alpha, max_eval) #alpha is the best score for the Ai 
                if alpha >= beta: #if alpha is greater than beta, then the Ai has found a better move, and we can prune the remaining branches
                    #return max_eval, best_move
                    break
            return max_eval, best_move
        else:
            #minimizing plater (beta)
            min_eval = float('inf')
            best_move = None
            for move in available_moves: 
                
                # board.push(move, self.foe)
                # eval_ = self.minimax(board, True, depth + 1, alpha, beta)[0] #evaluate the board
                # board.undo(move) 

                # CHANGED: Capture previous state
                prev_state = board.push(move, self.foe)
                eval_ = self.minimax(board, True, depth + 1, alpha, beta)[0]
                # CHANGED: Go back to previous state
                board.undo(move, prev_state)

                # Track best move
                if eval_ < min_eval:
                    min_eval = eval_
                    best_move = move

                # if min_eval == eval_: #if the min_eval is the same as the eval_, then the best move is the move
                #     best_move = move

                beta = min(min_eval, beta) #beta is defined at infinity, and given the minimum score
                #alpha beta pruning
                if beta <= alpha: #if beta is less than alpha, then the Ai has found a better move, and wew can prune the remaining branches
                    #return min_eval, best_move
                    break
            return min_eval, best_move

    def evaluate_board(self, board: Board, depth: int) -> Score:

        """
        - winner() and is_gameover() utilize wild tik tac toe
        """

        if board.is_gameover():
            winner = board.winner()
            
            if winner == self.ai:
                #the second player wins, and you return the size 
                return board.size**2 - depth
            elif winner == self.foe:
                #the second player looses 
                return -(board.size**2 + depth)
            else:
                #Draw situation 
                return 0
        
        
        return 0

        # if board.winner() == self.ai:
        #     return board.size**2 - depth #Positive score for Ai Wi n
        # elif board.winner() == self.foe:
        #     return -1 * board.size**2 - depth #negative score for Ai Loss
        # return 0 #0 is the score for Draw

    
    
    def evaluate_best_move(self, board: Board) -> Square:
        best_move = self.minimax(board, True, 0, float('-inf'),
                                 float('inf'))[1]
        return best_move

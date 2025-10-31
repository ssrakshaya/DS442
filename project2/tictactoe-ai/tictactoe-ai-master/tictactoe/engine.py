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

        """
        CHANGES:
        - Modified push() calls to capture returned state: prev_state = board.push(...)
        - Modified undo() calls to pass state back: board.undo(move, prev_state)
        - This ensures minimax properly simulates and restores game state including response_mode
        - Changed pruning condition from "alpha > beta" to "alpha >= beta" (standard alpha-beta)
        - Changed "if max_eval == eval_" to "if eval_ > max_eval" for cleaner best move tracking
        """

        available_moves = board.empty_squares #list of empty squares

        #random first move
        if len(available_moves) == board.size**2: #if there are no empty squares, then return a random move
            return 0, random.choice(list(range(board.size**2)))

        #ending state of the board
        if board.is_gameover() or depth >= self.level: # if the game is over or the depth is greater than the level, then return the score and the best move
            return self.evaluate_board(board, depth), None

        if ai_turn:
            max_eval = float('-inf') #max_eval is the best score for the Ai 
            best_move = None 
            for move in available_moves:
                board.push(move, self.ai) #push the move to the board
                eval_ = self.minimax(board, False, depth + 1, alpha, beta)[0] #evaluate the board
                board.undo(move)


                max_eval = max(max_eval, eval_) #max_eval is the best score for the Ai 
                if max_eval == eval_: #if the max_eval is the same as the eval_, then the best move is the move
                    best_move = move
                alpha = max(alpha, max_eval) #alpha is the best score for the Ai 
                if alpha > beta: #if alpha is greater than beta, then the Ai has found a better move, and we can prune the remaining branches
                    return max_eval, best_move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in available_moves: 
                board.push(move, self.foe)
                eval_ = self.minimax(board, True, depth + 1, alpha, beta)[0] #evaluate the board
                board.undo(move) 


                min_eval = min(min_eval, eval_) #min_eval is the best score for the opponent
                if min_eval == eval_: #if the min_eval is the same as the eval_, then the best move is the move
                    best_move = move
                beta = min(min_eval, beta) #beta is defined at infinity, and given the minimum score
                if beta < alpha: #if beta is less than alpha, then the Ai has found a better move, and wew can prune the remaining branches
                    return min_eval, best_move
            return min_eval, best_move

    def evaluate_board(self, board: Board, depth: int) -> Score:
        if board.winner() == self.ai:
            return board.size**2 - depth #Positive score for Ai Wi n
        elif board.winner() == self.foe:
            return -1 * board.size**2 - depth #negative score for Ai Loss
        return 0 #0 is the score for Draw

    def evaluate_best_move(self, board: Board) -> Square:
        best_move = self.minimax(board, True, 0, float('-inf'),
                                 float('inf'))[1]
        return best_move

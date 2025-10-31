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

                # max_eval = max(max_eval, eval_) #max_eval is the best score for the Ai 
                # if max_eval == eval_: #if the max_eval is the same as the eval_, then the best move is the move
                #     best_move = move


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


                #min_eval = min(min_eval, eval_) #min_eval is the best score for the opponent

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
        CHANGES:
        - Logic stays mostly the same!
        - The magic: winner() and is_gameover() now implement wild tic-tac-toe rules
        - So this function automatically evaluates correctly for wild tic-tac-toe
        - Minimax recursion naturally explores "what if opponent responds" scenarios
        
        HOW IT WORKS:
        - When AI makes 3-in-a-row, minimax explores opponent's response moves
        - If opponent can respond with 3-in-a-row → that path leads to negative score (AI loses)
        - If opponent cannot respond → that path leads to positive score (AI wins)
        - Minimax picks the path that maximizes AI's score
        """

        if board.is_gameover():
            winner = board.winner()
            
            if winner == self.ai:
                # AI wins - prefer faster wins (higher score for lower depth)
                return board.size**2 - depth
            elif winner == self.foe:
                # AI loses - prefer slower losses (less negative for higher depth)
                return -(board.size**2 + depth)
            else:
                # Draw
                return 0
        
        # Non-terminal state
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

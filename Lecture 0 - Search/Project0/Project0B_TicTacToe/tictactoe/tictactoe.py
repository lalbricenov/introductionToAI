"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    totalX = 0
    totalO = 0
    for row in board:
        totalX += row.count(X)
        totalO += row.count(O)
    return X if totalX == totalO else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions.append((i, j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    newBoard = copy.deepcopy(board)
    newBoard[action[0]][action[1]] = player(newBoard)
    return newBoard


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check horizontals
    for row in board:
        if row[0] == row[1] and row[1] == row[2]:
            return row[0]
    # Check verticals
    for j in range(3):
        if board[0][j] == board[1][j] and board[1][j] == board[2][j]:
            return board[0][j]
    # Check diagonals
    if board[0][0] == board[1][1] and board[2][2] == board[1][1]:
        return board[0][0]
    if board[0][2] == board[1][1] and board[2][0] == board[1][1]:
        return board[0][2]
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board):
        return True
    elif len(actions(board)) == 0:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if X == winner(board):
        return 1
    elif O == winner(board):
        return -1
    else:
        return 0


def minValue(board, currentMax):
    # currentMax is the value that X has currently in his maximization
    # The player that tries to minimize is O
    if terminal(board):
        return utility(board)
    else:
        # actionsNow = actions(board)
        minV = float("inf")
        for action in actions(board):
            value = maxValue(result(board, action), minV)
            minV = min(value, minV)
            if minV < currentMax:
                break
        return minV


def maxValue(board, currentMin):
    # The player that tries to maximize is X
    if terminal(board):
        return utility(board)
    else:
        maxV = float("-inf")
        for action in actions(board):
            value = minValue(result(board, action), maxV)
            maxV = max(maxV, value)
            if maxV > currentMin:
                break
        return maxV


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # If it is a terminal state
    if terminal(board):
        return None
    else:
        if player(board) == X:  # max
            maxV = float("-inf")
            # print(maxV)
            optimalAction = None
            for a in actions(board):
                # print(a)
                newMaxV = minValue(result(board, a), maxV)
                # print(newMaxV)
                if maxV < newMaxV:
                    optimalAction = a
                    maxV = newMaxV
            return optimalAction
        else:  # min
            minV = float("inf")
            optimalAction = None
            for a in actions(board):
                newMinV = maxValue(result(board, a), minV)
                if minV > newMinV:
                    optimalAction = a
                    minV = newMinV
            return optimalAction

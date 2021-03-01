"""
Submission for Project 2 of Columbia University's AI EdX course (Adversarial Search, 2048 Game).

    author: @rgkimball
    date: 2/28/2021
"""

from BaseAI import BaseAI


def utility(grid):
    score = 0
    # Placing the highest tiles in the corner is good (arbitrarily choosing bottom-left for this strategy)
    for i, row in enumerate(grid.map):
        for j, value in enumerate(row):
            score += value * (j + i)
    return score


def get_children(grid):
    """
    Only return viable nodes if their state differs from the initial.

    :param grid: Grid
    :return: list(Grid, ...)
    """
    nodes = {}
    for move in grid.getAvailableMoves():
        new = grid.clone()
        new.move(move)
        if grid.map != new.map:
            nodes[move] = new
    return nodes


class PlayerAI(BaseAI):

    def getMove(self, grid):

        moves = get_children(grid)
        utilities = list(zip(moves, map(utility, moves.values())))

        max_utility = float('-inf')
        max_child = None

        for g, u in utilities:
            if u > max_utility:
                max_child, max_utility = g, u

        return max_child

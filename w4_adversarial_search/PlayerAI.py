"""
Submission for Project 2 of Columbia University's AI EdX course (Adversarial Search, 2048 Game).

    author: @rgkimball
    date: 2/28/2021
"""
from time import time_ns
from BaseAI import BaseAI


def _monotonic(grid_map):
    """
    We prefer monotonically increasing rows, and we magnify this preference as our max tile increases.
    Using sqrt to taper this penalty as the game progresses.

    :param grid_map: list of rows, which are lists of tile values
    :return: int
    """
    sq_max = max(i for row in grid_map for i in row) ** 0.5
    return  int(all(i <= j for i, j in zip(grid_map, grid_map[1:]))) * sq_max


def _open_spaces(grid_map):
    """
    We prefer boards with open spaces, since this implies combinations have occurred.

    :param grid_map: list of rows, which are lists of tile values
    :return: int
    """
    return sum(1 if i == 0 else 0 for i in grid_map)


def _edge_score(grid_map):
    """
    Prefer to keep higher value tiles around the edges, with a bonus for corners and a penalty for inner tiles.

    :param grid_map: list of rows, which are lists of tile values
    :return: int
    """
    edges = (0, 3)
    board_sum = sum(c for row in grid_map for c in row)
    edge_sum = sum(c for i, row in enumerate(grid_map) for j, c in enumerate(row) if (i in edges) or (j in edges))
    corner_sum = sum(c for i, row in enumerate(grid_map) for j, c in enumerate(row) if (i in edges) and (j in edges))
    inner_sum = sum(c for i, row in enumerate(grid_map) for j, c in enumerate(row) if (i not in edges) and (j not in edges))
    return (2 * corner_sum + edge_sum - inner_sum) / board_sum


def _potential_merges(grid_map):
    """
    Returns the number of horizontally and vertically adjacent tiles of equal value; i.e. the potential 1-turn merges.

    :param grid_map: list of rows, which are lists of tile values
    :return: int
    """
    rows = sum(1 if i > 0 and i == j else 0 for row in grid_map for i, j in zip(row, row[1:]))
    colpairs = zip(grid_map, grid_map[1:])
    columns = sum(1 if c1[i] > 0 and c1[i] == c2[i] else 0 for c1, c2 in colpairs for i, _ in enumerate(c1))
    return rows + columns


def heuristic(grid_map):
    """
    Naively sums the underlying heuristic function scores, without respect for their distributions.

    :param grid_map: list of rows, which are lists of tile values
    :return: int, final heuristic score
    """

    im = _monotonic(grid_map)
    os = _open_spaces(grid_map)
    es = _edge_score(grid_map)
    pm = _potential_merges(grid_map)
    return im + os + es + pm


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

    max_search_depth = 5
    time_limit = 0.15  # seconds

    def getMove(self, grid):
        self.start = time_ns()
        self.depth = 0
        move, _ = self.maximize(grid, alpha=float('-inf'), beta=float('inf'))
        return move

    def maximize(self, grid, alpha, beta):
        self.depth += 1
        if self.depth > self.max_search_depth or not grid.canMove():
            return None, heuristic(grid.map)

        if self.terminate(grid):
            return None, heuristic(grid.map)

        max_utility, max_child = float('-inf'), None

        for child in get_children(grid):
            _, utility = self.minimize(child, alpha, beta)
            if utility > max_utility:
                max_child, max_utility = child, utility

        return max_child, max_utility

    def minimize(self, grid, alpha, beta):

        if self.terminate():
            return None, heuristic(grid.map)

        min_utility, min_child = float('inf'), None

        for child in get_children(grid):
            _, utility = self.maximize(child, alpha, beta)
            if utility < min_utility:
                min_child, min_utility = child, utility

        return min_child, min_utility

    def terminate(self, grid):
        return any([
            time_ns() - self.start >= self.time_limit,
            self.depth > self.max_search_depth,
            not grid.canMove(),
        ])

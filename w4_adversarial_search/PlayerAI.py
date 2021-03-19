"""
Submission for Project 2 of Columbia University's AI EdX course (Adversarial Search, 2048 Game).

    author: @rgkimball
    date: 2/28/2021
"""
from time import time
from BaseAI import BaseAI


def _monotonic(grid_map, p=1.3):
    """
    Reward monotonic rows & columns, increasing exponentially with the rank.

    :param grid_map: list of rows, which are lists of tile values
    :return: int
    """
    max_tile = max(v for r in grid_map for v in r)
    horizontal = [i ** p - j ** p if i > j else j ** p - i ** p for r in grid_map for i, j in zip(r, r[1:])]
    transpose = [[grid_map[i][j] for i in range(4)] for j in range(4)]
    vertical = [i ** p - j ** p if i > j else j ** p - i ** p for r in transpose for i, j in zip(r, r[1:])]
    corner_bonus = max_tile if max_tile == grid_map[0][0] else 1
    return sum(horizontal) * corner_bonus + sum(vertical) / 2


def _open_spaces(grid_map):
    """
    We prefer boards with open spaces, since this implies combinations have occurred.

    :param grid_map: list of rows, which are lists of tile values
    :return: int
    """
    return sum(1 if i == 0 else 0 for row in grid_map for i in row)


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
    Returns the sum of horizontally and vertically adjacent tiles of equal value; i.e. the potential 1-turn merges.
    Use of the sum encourages high-value combinations.

    :param grid_map: list of rows, which are lists of tile values
    :return: int
    """
    rows = sum(1 if i > 0 and i == j else 0 for row in grid_map for i, j in zip(row, row[1:]))
    colpairs = zip(grid_map, grid_map[1:])
    columns = sum(c1[i] ** 2 if c1[i] == c2[i] else 0 for c1, c2 in colpairs for i, _ in enumerate(c1))
    return rows + columns


def _sum_power(grid_map, power=2):
    """
    Robert Xiao's sum power heuristic to measure raw board value.

    :param grid_map:
    :param power:
    :return:
    """
    return sum([i ** power for row in grid_map for i in row])
    # return sum([float(''.join(row)) ** power for row in grid_map])


def heuristic(grid_map):
    """
    Combination of the underlying heuristic function scores.

    :param grid_map: list of rows, which are lists of tile values
    :return: int, final heuristic score
    """

    im = _monotonic(grid_map)
    os = _open_spaces(grid_map)
    sp = _sum_power(grid_map)
    pm = _potential_merges(grid_map)

    # Weights
    im *= 1/4
    sp *= 1/6
    os *= 1/100
    pm *= 2

    # print('Heuristic score (os+pm-im-sp): ', os, pm, im, sp, os + pm - im - sp)
    # return os + pm - im - sp
    return os * pm * im * sp


def get_children(grid, turn='player'):
    """
    Only return viable nodes if their state differs from the initial.

    :param turn: string, either player or pc - determines how child nodes are expanded from the current grid state.
    :param grid: Grid
    :return: list(Grid, ...)
    """
    nodes = []
    if turn == 'player':
        moves = grid.getAvailableMoves()
        moves = sorted(moves, key=lambda x: PlayerAI.order_preference.index(x))
        for move in moves:
            new = grid.clone()
            new.move(move)
            if grid.map != new.map:
                nodes.append((move, new))
    elif turn == 'pc':
        spaces = grid.getAvailableCells()
        for value in (2, ):
            for space in spaces:
                this = grid.clone()
                this.setCellValue(space, value)
                nodes.append((space, this))

    return nodes


class PlayerAI(BaseAI):

    max_search_depth = 600
    depth_searched = 0
    time_limit = 0.24  # seconds
    prob_2 = 0.9  # % chance that a new tile is a 2, instead of a 4
    order_preference = (0, 2, 1, 3)

    def getMove(self, grid):
        # Ensure we stop searching before our turn is over
        self.start = time()
        # Used to limit the ultimate search depth where heuristic scores are calculated
        self.depth = 0
        self.explored = 0

        move, _ = self.maximize(grid, float('-inf'), float('inf'))
        return move

    def maximize(self, grid, alpha, beta):
        self.depth += 1
        if self.depth > self.depth_searched:
            self.depth_searched = self.depth
        self.explored += 1

        if self.terminate(grid):
            return None, heuristic(grid.map)

        max_utility, max_child = float('-inf'), None

        for move, child in get_children(grid, turn='player'):
            _, utility = self.minimize(child, alpha, beta)
            self.depth -= 1  # reset depth for next iteration

            if utility > max_utility:
                max_child, max_utility = move, utility
            if max_utility >= beta:
                break  # tree prune
            if max_utility > alpha:
                alpha = max_utility
            if self.clock_limit():
                break

        return max_child, max_utility

    def minimize(self, grid, alpha, beta):
        self.depth += 1
        if self.depth > self.depth_searched:
            self.depth_searched = self.depth
        self.explored += 1

        if self.terminate(grid):
            return None, heuristic(grid.map)

        min_utility, min_child = float('inf'), None

        for move, child in get_children(grid, turn='pc'):
            _, utility = self.maximize(child, alpha, beta)
            self.depth -= 1  # reset depth for next iteration
            if utility < min_utility:
                min_child, min_utility = move, utility
            if min_utility <= alpha:
                break  # tree prune
            if min_utility < beta:
                beta = min_utility
            if self.clock_limit():
                break

        return min_child, min_utility

    def terminate(self, grid):
        return any([
            self.clock_limit(),
            self.depth >= self.max_search_depth,
            not grid.canMove(),
        ])

    def clock_limit(self):
        return time() - self.start >= self.time_limit

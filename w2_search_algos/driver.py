"""
Submission for Project 1 of Columbia University's AI EdX course (8-puzzle).

    author: @rgkimball
    date: 2/9/2021
"""

import os
import sys
import math
import time
import heapq
from collections import deque
from multiprocessing import Pool, cpu_count


class PuzzleState(object):
    """The Class that Represents the Puzzle"""
    action_hierarchy = ('Up', 'Down', 'Left', 'Right')

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n * n != len(config) or n < 2:
            raise ValueError("The length of config is not correct!")

        self.n = n
        self.cost = cost
        self.parent = parent
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1
        self.action = action
        self.dimension = n
        self.config = config
        self.children = []

        for i, item in enumerate(self.config):
            if item == 0:
                self.blank_row = i // self.n
                self.blank_col = i % self.n
                break

    def display(self):
        for i in range(self.n):
            line = []
            offset = i * self.n
            for j in range(self.n):
                line.append(self.config[offset + j])
            print(line)

    def move_left(self):

        if self.blank_col == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    def move_right(self):

        if self.blank_col == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + 1
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    def move_up(self):
        if self.blank_row == 0:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index - self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    def move_down(self):
        if self.blank_row == self.n - 1:
            return None
        else:
            blank_index = self.blank_row * self.n + self.blank_col
            target = blank_index + self.n
            new_config = list(self.config)
            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        """expand the node"""

        # add child nodes in order of UDLR
        if len(self.children) == 0:
            up_child = self.move_up()
            if up_child is not None:
                self.children.append(up_child)
            down_child = self.move_down()
            if down_child is not None:
                self.children.append(down_child)
            left_child = self.move_left()
            if left_child is not None:
                self.children.append(left_child)
            right_child = self.move_right()

            if right_child is not None:
                self.children.append(right_child)

        return self.children

    def __eq__(self, other):
        if isinstance(other, PuzzleState):
            return str(self.config) == str(other.config)
        else:
            raise ValueError('The equivalence of incompatible types is ambiguous.')

    def __lt__(self, other):
        if isinstance(other, PuzzleState):
            l, r = map(calculate_total_cost, [self, other])
            if l < r:
                return True
            elif l == r:
                la, ra = map(self.action_hierarchy.index, [self.action, other.action])
                if la < ra:
                    return True
                elif la == ra:
                    return self.depth > other.depth
                return False
            else:
                return False
        else:
            raise ValueError('The comparison of incompatible types is ambiguous.')

    def __hash__(self):
        return hash(str(self.config))

    def __repr__(self):
        return str(self.config)


MEM_USAGE = {}


def record_usage(*args):
    """
    Create a timestamped snapshot of the current process' RAM usage.

    :return: None, result appended to MEM_USAGE global
    """
    global MEM_USAGE
    MEM_USAGE[time.time()] = sum(map(sys.getsizeof, args))


def write_output(
        algo: str,
        state: PuzzleState,
        max_depth: int,
        expanded: int,
):
    """
    Function that writes to output.txt

    :return:
    """
    global MEM_USAGE
    this, depth, actions = state, 1, []
    while this.parent is not None:
        actions.append(this.action)
        depth += 1
        this = this.parent

    output = {
        'path_to_goal': list(reversed(actions)),
        'cost_of_path': len(actions),
        'nodes_expanded': expanded,
        'search_depth': state.cost,
        'max_search_depth': max_depth,
        'running_time': max(MEM_USAGE.keys()) - min(MEM_USAGE.keys()),
        'max_ram_usage': max(MEM_USAGE.values()) / (1024 ** 2),
    }

    fnam = f'{algo}_output.txt'
    contents = '\n'.join([f'{k}: {v}' for k, v in output.items()])
    with open(fnam, 'w') as fo:
        fo.write(contents)
    print(contents)
    print(f'Output file saved to {fnam}')
    MEM_USAGE = {}
    return output


def bfs_search(initial_state: PuzzleState):
    """
    Implements a breadth-first search algorithm to solve the 8-puzzle game.

    :param initial_state: PuzzleState object
    :return: N/A
    """
    frontier, explored = deque(), set()
    frontier.append(initial_state)
    explored.add(initial_state)
    max_depth, expanded = 0, 0

    while frontier:
        state = frontier.popleft()
        record_usage(frontier, explored, state)

        if test_goal(state):
            return write_output('bfs', state, max_depth, expanded)

        expanded += 1
        for node in state.expand():
            if node not in explored:
                explored.add(node)
                frontier.append(node)
                max_depth = max(max_depth, node.cost)

    print(f'No solution found after {len(explored)} expansions.', initial_state.config)


def dfs_search(initial_state: PuzzleState):
    """
    DFS search

    :param initial_state:
    :return:
    """
    frontier, explored = deque(), set()
    frontier.append(initial_state)
    explored.add(initial_state)
    max_depth, expanded = 0, 0

    while frontier:
        state = frontier.pop()
        record_usage(frontier, explored, state)

        if test_goal(state):
            return write_output('dfs', state, max_depth, expanded)

        expanded += 1
        for node in reversed(state.expand()):
            if node not in explored:
                explored.add(node)
                frontier.append(node)
                max_depth = max(max_depth, node.cost)

    print(f'No solution found after {len(explored)} expansions.', initial_state.config)


def a_star_search(initial_state: PuzzleState):
    """
    Implements A* search

    :param initial_state:
    :return:
    """
    frontier, explored = list(), set()
    heapq.heappush(frontier, initial_state)
    explored.add(initial_state)
    max_depth, expanded = 0, 0

    while frontier:
        state = heapq.heappop(frontier)
        record_usage(frontier, explored, state)

        if test_goal(state):
            return write_output('ast', state, max_depth, expanded)

        expanded += 1
        for node in state.expand():
            if node not in explored:
                explored.add(node)
                heapq.heappush(frontier, node)
                max_depth = max(max_depth, node.cost)
            elif node in frontier:
                frontier.remove(node)
                heapq.heappush(frontier, node)
                heapq.heapify(frontier)

    return initial_state


def calculate_total_cost(state: PuzzleState):
    """
    Calculate the total estimated cost of a state

    :param state: PuzzleState object
    :return: total cost of the state
    """
    x = sum(calculate_manhattan_dist(i, v, state.dimension) for i, v in enumerate(state.config))
    return x + state.cost


def calculate_manhattan_dist(idx, value, n):
    """
    Calculate the manhattan distance of a tile
    :the Manhattan distance of tile `value` from index `idx` in a `n` x `n` board.

    :param idx: int, the current location of the tile
    :param value: int, the value of this tile
    :param n: int, the dimensions of the board
    :return: int, Manhattan distance
    """
    if value == 0:
        return 0
    return abs(value // n - idx // n) + abs(value % n - idx % n)


def test_goal(puzzle_state: PuzzleState):
    """
    Test whether a given state is the goal state.

    :param puzzle_state: PuzzleState object containing an iterable 'config' with numeric values.
    :return: bool, True if the state is the goal state.
    """
    arr = list(puzzle_state.config)
    return all(a < b for a, b in zip(arr, arr[1:]))


def format_game(raw):
    """
    Utility function to build a PuzzleState object from a raw input string.

    :param raw: str containing tile values, like '1,4,2,7,5,8,3,0,6'
    :return: PuzzleState object
    """
    split = raw.replace('\n', '').split(",")
    tiles = tuple(map(int, split))
    size = int(math.sqrt(len(tiles)))
    return PuzzleState(tiles, size)


def pool_run(game, run, algo, destination):
    game = game.replace('\n', '')
    output = run(format_game(game))
    output.update({
        'algo': algo,
        'path_to_goal': ''.join([v[0] for v in output['path_to_goal']][:32]),
        'starting_grid': game.replace(',', ''),
    })
    if not os.path.isfile(destination):
        with open(destination, 'a+') as fo:
            fo.write(','.join(map(str, output.keys())) + '\n')
            fo.write(','.join(map(str, output.values())) + '\n')
    else:
        with open(destination, 'a+') as fo:
            fo.write(','.join(map(str, output.values())) + '\n')


def main():
    """
    Main Function that reads in Input and Runs corresponding Algorithm

    :return:
    """

    algos = {
        'bfs': bfs_search,
        'dfs': dfs_search,
        'ast': a_star_search,
    }

    # Arguments can either specify a single file, from which we can test all puzzles iteratively, or a pair of values
    # indicating the algorithm to use and the puzzle to solve, like: driver.py bfs 7,2,4,5,0,6,8,3,1
    if len(sys.argv) < 3:
        file_name = sys.argv[1]
        direc = os.path.abspath(os.getcwd())
        path = os.path.join(direc, file_name)
        solved_path = os.path.join(direc, file_name.replace('.', '_solved.', 1))

        if os.path.isfile(path):
            with open(path, 'r') as fo:
                games = fo.readlines()
            pool = Pool(processes=cpu_count())
            queue = []
            for algo, run in algos.items():
                for game in games:
                    queue.append((game, run, algo, solved_path))
            pool.starmap(pool_run, queue)
        else:
            raise ValueError('Invalid arguments, should be an individual game or file path containing multiple games.')
    else:
        sm = sys.argv[1].lower()
        game = format_game(sys.argv[2])

        # PEP 634 is finally here! No more dictionary switch cases!
        if sm in algos.keys():
            algos[sm](game)
        else:
            raise ValueError("Enter a valid code! Algorithm should be one of {}".format(', '.join(algos.keys())))


if __name__ == '__main__':
    main()

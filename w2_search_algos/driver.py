"""
Submission for Project 1 of Columbia University's AI EdX course (8-puzzle).

    author: @rgkimball
    date: 2/9/2021
"""

import sys
import math
import time
from collections import deque
from multiprocessing import Queue


class PuzzleState(object):
    """The Class that Represents the Puzzle"""

    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        if n * n != len(config) or n < 2:
            raise ValueError("The length of config is not correct!")

        self.n = n
        self.cost = cost
        self.parent = parent
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
        'max_ram_usage': max(MEM_USAGE.values()) * (1024 * 10 ** -10),
    }

    fnam = f'{algo}_output.txt'
    contents = '\n'.join([f'{k}: {v}' for k, v in output.items()])
    with open(fnam, 'w') as fo:
        fo.write(contents)
    print(contents)
    print(f'Output file saved to {fnam}')

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
    A * search

    :param initial_state:
    :return:
    """
    # STUDENT CODE GOES HERE ###
    return initial_state


def calculate_total_cost(state: PuzzleState):
    """
    Calculate the total estimated cost of a state

    :param state:
    :return:
    """
    # STUDENT CODE GOES HERE ###
    return state


def calculate_manhattan_dist(idx, value, n):
    """
    Calculate the manhattan distance of a tile

    :param idx:
    :param value:
    :param n:
    :return:
    """
    # STUDENT CODE GOES HERE ###
    return idx, value, n


def test_goal(puzzle_state: PuzzleState):
    """
    Test whether a given state is the goal state.

    :param puzzle_state: PuzzleState object containing an iterable 'config' with numeric values.
    :return: bool, True if the state is the goal state.
    """
    arr = list(puzzle_state.config)
    return all(a < b for a, b in zip(arr, arr[1:]))


def main():
    """
    Main Function that reads in Input and Runs corresponding Algorithm

    :return:
    """

    if len(sys.argv) < 2:
        raise ValueError('Invalid command arguments given, first should be an algo code followed by the init state.')

    sm = sys.argv[1].lower()
    begin_state = sys.argv[2].split(",")
    begin_state = tuple(map(int, begin_state))
    size = int(math.sqrt(len(begin_state)))
    hard_state = PuzzleState(begin_state, size)

    algos = {
        'bfs': bfs_search,
        'dfs': dfs_search,
        'ast': a_star_search,
    }

    # PEP 634 is finally here! No more dictionary switch cases!
    if sm in algos.keys():
        algos[sm](hard_state)
    else:
        raise ValueError("Enter a valid code! Algorithm should be one of {}".format(', '.join(algos.keys())))


def test_algos(file_name):
    import pandas as pd
    global MEM_USAGE
    all_stats = []
    with open(file_name, 'r') as f:
        grids = [tuple(map(int, x.replace('\n', '').split(','))) for x in f.readlines()]
    for i, grid in enumerate(grids):
        print(i, grid)
        # Initialize grid
        this = PuzzleState(tuple(grid), int(math.sqrt(len(grid))))
        combined = {'i': i, 'grid': ','.join([str(i) for i in grid])}
        # Run BFS Algo
        bfs = bfs_search(this)
        if bfs is not None:
            bfs['path_to_goal'] = ''.join(s[0] for s in bfs['path_to_goal'])
            bfs = {f'bfs_{k}': v for k, v in bfs.items()}
            combined.update(bfs)
        MEM_USAGE = {}
        # Run DFS Algo
        dfs = dfs_search(this)
        if dfs is not None:
            del dfs['path_to_goal']
            dfs = {f'dfs_{k}': v for k, v in dfs.items()}
            combined.update(dfs)
        MEM_USAGE = {}
        all_stats.append(combined)
        if i > 1:
            break

    df = pd.DataFrame. from_records(all_stats)
    df.to_csv('all_stats.csv', index=False, mode='a', header=False)


if __name__ == '__main__':
    # main()
    test_algos('solvable_grids.txt')

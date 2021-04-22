"""
Submission for Project 4 of Columbia University's AI EdX course (Constraint Satisfaction/Sudoku).

    author: @rgkimball
    date: 4/18/2021
"""

import numpy as np
from sys import argv
from copy import deepcopy
from itertools import product
from collections import OrderedDict
from datetime import datetime as dt


class SudokuBoard(OrderedDict):
    """
    A representation of a Sudoku game board, where cells are stored with letters A-I along the rows and numbers 1-9
    to enumerate the columns. This class is therefore limited to 9x9 games. New games can be initiated with a string of
    81 digits, i.e.:
    >>> SudokuBoard('000005000007060430406008205002000000008074900601800000003500000060001020100327090')

    We can determine whether or not a particular board is solved using the solved property. This is implemented as a
    property since we need to calculate its value each time we make a change to the cells while solving:
    >>> SudokuBoard('329745618857162439416938275742619853538274961691853742273596184965481327184327596').solved
    >> True

    Several helper methods assist with the search for solutions:

    - cell_neighbors: can give us a list of all contingent cells given a particular cell's key:
        >>> SudokuBoard('...').cell_neighbors('E1')
        >> ['A1','B1','C1','D1','D2','D3','E2','E3','E4','E5','E6','E7','E8','E9','F1','F2','F3','G1','H1','I1']
    - legal_value: can tell us whether a particular value is allowed for a specific cell given its contingent cells:
        >>> SudokuBoard('...').legal_value('E1', 5)
        >> True or False
    - sort_min_remaining: a heuristic function used in backtracking search which sorts all of the unassigned cells and
                          sorts them according to how many legal values remain - we assign the most constrained first.
    - fill_resolved: this is used to preserve the keys while we're trying different values, and the algorithms must
                     populate all of the cells that only have one legal value at the end of their respective iterations.
    """

    columns = list(map(str, range(1, 10)))
    rows = list('ABCDEFGHI')
    domain = list(range(1, 10))

    @property
    def squares(self):
        return [(self.rows[i * 3:i * 3 + 3], self.columns[j * 3:j * 3 + 3]) for i in range(3) for j in range(3)]

    @property
    def unresolved(self):
        return len([k for k in self.keys() if self[k] == 0])

    def __init__(self, string):
        super().__init__()
        self.open = {}
        self.__from_str__(string)

    def __str__(self):
        return ''.join(map(str, self.values()))

    def __repr__(self):
        return "SudokuBoard('{}')".format(str(self))

    def __from_str__(self, board):
        ib = iter(board)
        for row in self.rows:
            for col in self.columns:
                self[row + col] = int(next(ib))
        # Calculate potential values for each cell
        for k, v in self.items():
            if v == 0:
                self.open[k] = self.domain.copy()
            else:
                self.open[k] = [self[k]]

    @property
    def solved(self):
        cl, rw, sq = [], [], []
        for c in self.columns:
            this_column = [v for k, v in self.items() if c in k]
            cl.append(self.__all_unique__(this_column))
        for r in self.rows:
            this_row = [v for k, v in self.items() if r in k]
            rw.append(self.__all_unique__(this_row))
        for c, r in self.squares:
            this_square = {k: self[k] for k in map(lambda x: x[0] + x[1], list(product(c, r)))}
            sq.append(self.__all_unique__(list(this_square.values())))
        return all(cl + rw + sq)

    @staticmethod
    def __all_unique__(lst):
        return len(set(lst)) == len(lst) and 0 not in lst

    def cell_square(self, key):
        row, col = list(key)
        for s in self.squares:
            r, c = s
            if col in c and row in r:
                return {k: self[k] for k in map(lambda x: x[0] + x[1], list(product(r, c)))}
        return {}

    def cell_neighbors(self, key):
        """
        Provides a list of keys with which the specified target cell is constrained.

        :param key: str, reference to a target cell, like 'A1'
        :return: list of references to neighbor cells: ['A2', 'A3', ..., 'H1', 'I1']
        """
        r, c = list(key)
        col = [k for k in self.keys() if c in k if k != key]
        row = [k for k in self.keys() if r in k if k != key]
        square = [k for k in self.cell_square(key).keys() if k != key]
        return sorted(set(col + row + square))

    def legal_value(self, key, assignment):
        """
        Returns the valid options for assignment within a particular cell. If the cell is already we assigned we assume
        it is assigned correctly and return its value, otherwise we compare it to neighboring cells via cell_neighbors()

        :param key: cell reference as a str, like 'A1'
        :param assignment: int, the intended value we want to assign to the cell referenced by key
        :return: set of possible values for assignment, like (3, 5)
        """
        for cell in self.cell_neighbors(key):
            if len(self.open[cell]) == 1 and assignment == self[cell]:
                return False
            else:
                if assignment in self.open[cell]:
                    self.open[cell].remove(assignment)
                    if len(self.open[cell]) == 0:
                        return False
        return True

    def sort_min_remaining(self):
        """
        Returns a dictionary of all open cells sorted by the cells with the fewest remaining legal values.
        Used in the process of backtracking search to prioritize highly-constrained cell assignments.

        :return: dict
        """
        return {k: v for k, v in sorted(self.open.items(), key=lambda item: len(item[1])) if len(v) > 1}

    def fill_resolved(self):
        """
        Helper function to populate the board keys as we resolve values; iterates through the values in SudokuBoard.open
        and transfers the value to keys if there is only one remaining legal value for that cell.

        :return: N/A
        """
        for key, value in self.open.items():
            if len(value) == 1:
                self[key] = value[0]

    def copy(self):
        """
        Permits us to attempt multiple values in cells without tampering with the original internal data structures.

        :return: new SudokuBoard object independent from the original.
        """
        new = SudokuBoard(str(self))
        new.open = deepcopy(self.open)
        return new


class Sudoku:
    """
    A wrapper class for the Sudoku game solver. Most of the logic is implemented in SudokuBoard, but the game itself
    is initiated and managed here. We preserve both the initial state and the current state while we're solving the
    board.

    Using the solve() method, we first attempt to find missing values using arc-consistency (AC-3), and more complicated
    puzzles (i.e. most of them) are then passed along to the backtracking search to systematically eliminate values for
    cells that violate selections for its neighboring cells (vertically, horizontally, or within its square).

    This class exposes cell values from its underlying SudokuBoard object, stored in Sudoku.board.
    """

    def __init__(self, board):
        self.board = SudokuBoard(board)
        self.starting_board = SudokuBoard(board)

    def __str__(self):
        return str(self.board)

    def __repr__(self):
        return "Sudoku('{}')".format(str(self.starting_board))

    def __getitem__(self, item):
        return self.board[item]

    def solve(self):
        if self.board.solved:
            return str(self) + " NONE"
        ac3(self.board)
        if self.board.solved:
            return str(self) + " AC3"

        # If the board remains unsolved, we need to run backtracking search to find the remaining cells.
        self.board = bts(self.board.copy())
        return str(self) + " BTS"


def revise(board: SudokuBoard, x_i, x_j):
    """
    Associate function of AC-3 to check whether any allowable values of a cell are illegal based on the value of a
    specific neighbor cell.

    :param board: A SudokuBoard object
    :param x_i: cell reference for target cell as a str, like 'A1'
    :param x_j: cell reference for neighbor cell as a str, like 'A2'
    :return: True if any values were found to violate the condition, False if no contradictions are found.
    """
    revised = False
    for value in board.open.get(x_i, [board[x_i]]):
        # Check to see whether any potential values for x_i violate any constraints for x_j
        if all(not value != y for y in board.open[x_j]):
            board.open[x_i].remove(value)
            revised = True
    return revised


def ac3(board: SudokuBoard):
    """
    Runs the Arc-Consistency (AC-3) algorithm to quickly resolve simple puzzles and narrow the search space before more
    advanced algorithms to take over.

    Uses class parameters as inputs to form CSP:
     - Domain (D): self.board.domain/self.board.open
     - Variables (X): keys in self.board
     - Constraints (C): enforced by self.revise(), where neighbor cells must not have the same value.

    :param board: A SudokuBoard object
    :return: True when the algorithm has terminated
    """
    queue = [(x_i, x_j) for x_i in board.keys() for x_j in board.cell_neighbors(x_i)]
    while queue:
        ki, kj = queue.pop()
        if revise(board, ki, kj):
            if not len(board.open[ki]):
                return False
            for x in board.cell_neighbors(ki):
                if x != ki:
                    queue.append((x, ki))
        board.fill_resolved()
    return True


def bts(board: SudokuBoard, debug=False):
    """
    Implements backtracking search algorithm to solve tougher puzzles.

    :param board: A SudokuBoard object
    :param debug: boolean, prints extra info to the console if needed to trace the path of the algo.
    :return: a solved SudokuBoard object if a solution is found, otherwise the algorithm will run indefinitely.
    """
    if not ac3(board):
        return False
    if board.solved:
        return board

    unassigned = board.sort_min_remaining()
    key = list(unassigned)[0]

    if debug:
        test = {k: v for k, v in board.open.items() if len(v) == 1}
        print(len(test) * '.', 81 - len(test), 'left', key, unassigned[key])

    for value in unassigned[key]:
        new = board.copy()
        if new.legal_value(key, value):
            new[key], new.open[key] = value, [value]
            # Recursively assign new values if we've settled on a valid one until it breaks:
            new = bts(new)
            if new:
                return new


def write_solution(string, fnam='output.txt'):
    """
    Simple helper to create or append to a file with new lines to the specified location.

    :param string: str, line to be added to the file
    :param fnam: name of the file as a str
    :return: N/A
    """
    with open(fnam, 'a+') as fo:
        fo.write(string + '\n')


if __name__ == '__main__':

    argument = argv[1]

    if len(argument) == 81 and argument.isdecimal():
        start = dt.now()
        game = Sudoku(argument)
        solution = game.solve()
        write_solution(solution)
        print(argument, solution, dt.now() - start)
    else:
        boards = np.loadtxt(argument, delimiter=' ', dtype=str)
        for i, b in enumerate(boards):
            start = dt.now()
            game = Sudoku(b)
            solution = game.solve()
            write_solution(solution, 'multi_output.txt')
            print(i, b, solution, dt.now() - start)

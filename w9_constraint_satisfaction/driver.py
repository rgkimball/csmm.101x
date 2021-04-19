"""
Submission for Project 4 of Columbia University's AI EdX course (Constraint Satisfaction/Sudoku).

    author: @rgkimball
    date: 4/18/2021
"""

import numpy as np
from sys import argv
from itertools import product


class SudokuBoard(dict):

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
            sq.append(self.__all_unique__(this_square))
        return all(cl + rw + sq)

    @staticmethod
    def __all_unique__(lst):
        return len(set(lst)) == len(lst)

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


class Sudoku:

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
        self.ac3()
        if self.board.solved:
            return str(self) + " AC3"

        self.bts()
        return str(self) + " BTS"

    def revise(self, x_i, x_j):
        revised = False
        for value in self.board.open.get(x_i, [self.board[x_i]]):
            # Check to see whether any potential values for x_i violate any constraints for x_j
            if all(not value != y for y in self.board.open[x_j]):
                self.board.open[x_i].remove(value)
                revised = True
        return revised

    def ac3(self):
        queue = [(x_i, x_j) for x_i in self.board.keys() for x_j in self.board.cell_neighbors(x_i)]
        while queue:
            ki, kj = queue.pop()
            if self.revise(ki, kj):
                if not len(self.board.open[ki]):
                    return False
                for x in self.board.cell_neighbors(ki):
                    if x != ki:
                        queue.append((x, ki))
            self.fill_resolved()
        return True

    def fill_resolved(self):
        for key, value in self.board.open.items():
            if len(value) == 1:
                self.board[key] = value[0]


def write_solution(string, fnam='output.txt'):
    with open(fnam, 'a+') as fo:
        fo.write(string)


if __name__ == '__main__':

    argument = argv[1]

    if len(argument) == 81 and argument.isdecimal():
        game = Sudoku(argument)
        solution = game.solve()
        write_solution(solution)
    else:
        boards = np.loadtxt(argument, delimiter=' ')
        for b in boards:
            game = Sudoku(b)
            solution = game.solve()
            write_solution(solution, 'multi_output.txt')

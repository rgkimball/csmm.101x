"""
Submission for Project 4 of Columbia University's AI EdX course (Constraint Satisfaction/Sudoku).

    author: @rgkimball
    date: 4/18/2021
"""

import numpy as np
from sys import argv
from itertools import product


class Sudoku:

    columns = list(map(str, range(1, 10)))
    rows = list('ABCDEFGHI')

    @property
    def squares(self):
        return [(self.rows[i:i + 3], self.columns[j:j + 3]) for i in range(3) for j in range(3)]

    def __init__(self, board):
        self.board = self.__parse_board__(board)
        self.starting_board = self.__parse_board__(board)

    def __parse_board__(self, board):
        grid = {}
        ib = iter(board)
        for row in self.rows:
            for col in self.columns:
                grid[row + col] = next(ib)
        return grid

    @staticmethod
    def __flatten_board__(board):
        return board

    def __getitem__(self, item):
        return self.board[item]

    def solve(self):
        assignment = ac3(self.starting_board)
        if self.solved(assignment):
            return self.__flatten_board__(self.board) + " AC3"

        assignment = self.bts(self.starting_board)
        return self.__flatten_board__(self.board) + " BTS"

    def solved(self):
        cl, rw, sq = [], [], []
        for c in self.columns:
            this_column = [v for k, v in self.board.items() if c in k]
            cl.append(self.__all_unique__(this_column))
        for r in self.rows:
            this_row = [v for k, v in self.board.items() if r in k]
            rw.append(self.__all_unique__(this_row))
        for c, r in self.squares:
            this_square = {k: self.board[k] for k in map(lambda x: x[0] + x[1], list(product(c, r)))}
            sq.append(self.__all_unique__(this_square))
        return all(cl + rw + sq)

    @staticmethod
    def __all_unique__(lst):
        return len(set(lst)) == len(lst)


if __name__ == '__main__':

    argument = argv[1]

    if len(argument) == 81 and argument.isdecimal():
        game = Sudoku(argument)
        game.solved()
    else:
        boards = np.loadtxt(argument, delimiter=' ')
        for b in boards:
            game = Sudoku(b)
            game.solve()

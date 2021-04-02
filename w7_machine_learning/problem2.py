"""
Submission for Project 3, Part 2 of Columbia University's AI EdX course (Machine Learning: Linear Regression).

    author: @rgkimball
    date: 3/31/2021
"""

from sys import argv
import numpy as np

LEARNING_RATES = [
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5,
    1,
    5,
    10,
]


def zscore(arr):
    """
    Used to normalize input feature data

    :param arr:
    :return:
    """
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


def descend(x, y, a, n_iter):
    """
    Gradient descent algorithm for linear regression

    :param x: ndarray, features
    :param y: 1darray, labels
    :param a: float, the learning rate "alpha"
    :param n_iter: number of times to iterate GD algorithm
    :return: tuple: number of iterations, regression coefficients in 1darray, and the final cost (squared errors)
    """
    r_B = 0
    n = len(y)
    coef = np.zeros((x.shape[1], 1))
    for i in range(n_iter):
        coef = coef - a / n * x.T.dot(x.dot(coef) - y)
        r_B = ((x.dot(coef) - y).T.dot(x.dot(coef) - y) / (2 * n))[0][0]
    return n_iter, coef, r_B


def optimize_alpha(alphas, x, y, n_iter):
    """
    Optimize the learning rate via binary search toward lowest cost solution.

    :param alphas: set of all alphas tried and cost values
    :param x: features, ndarray
    :param y: labels, 1darray
    :param n_iter: number of iterations for gradient descent using each alpha
    :return: same as descend():
        tuple: number of iterations, regression coefficients in 1darray, and the final cost (squared errors)
    """
    n, c, a = None, None, None
    for _ in range(int(n_iter // 10)):
        a_cp = alphas.copy()
        low1 = min(a_cp, key=a_cp.get)
        del a_cp[low1]
        low2 = min(a_cp, key=a_cp.get)
        a = (low1 + low2) / 2
        n, c, alphas[a] = descend(x, y, a, n_iter)
    return a, n, c, alphas[a]


def linreg(x, y, intercept=True, n_iter=100, alphas=None):
    """
    Implements a gradient descent approach to fitting a linear regression model.
    Uses a generator to allow iteartion through different learning rates.

    :param x: numpy array of covariates
    :param y: 1d array of features
    :param intercept: boolean, whether to include an intercept term
    :param n_iter: number of times to iterate through gradient descent algorithm
    :param alphas: list of learning rates
    :return: yields a tuple
    """
    if alphas is None:
        alphas = LEARNING_RATES
    # We will populate the dictionary with the cost of each result
    alphas = {k: None for k in alphas}

    x = np.apply_along_axis(zscore, axis=0, arr=x)
    y = y.reshape((y.shape[0]), 1)
    n = len(y)

    # Add column of 1's for intercept
    if intercept:
        x = np.insert(x, 0, [1] * n, axis=1)
    if n_iter < 1:
        return 0, 0, None

    # Initiate gradient descent
    for a in alphas:
        n, coef, r = descend(x, y, a, n_iter)
        alphas[a] = r
        yield a, n, coef, r
    yield optimize_alpha(alphas, x, y, n_iter)


if __name__ == '__main__':
    in_file, out_file = argv[1:3]

    data = np.genfromtxt(in_file, delimiter=',')
    # Assume the last column is labels:
    n_X = data.shape[1] - 1
    features, labels = data[:, :n_X], data[:, n_X]

    stout = []
    for alpha, iterations, betas, _ in linreg(features, labels):
        stout.append(','.join(list(map(lambda d: str(float(d)), [alpha, iterations] + [x[0] for x in betas]))))

    if out_file is not None:
        with open(out_file, 'w+') as fo:
            fo.write('\n'.join(stout))

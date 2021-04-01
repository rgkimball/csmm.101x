"""
Submission for Project 3, Part 1 of Columbia University's AI EdX course (Machine Learning: Perceptron).

    author: @rgkimball
    date: 3/31/2021
"""

from sys import argv
import numpy as np
from matplotlib import pyplot as plt


def sign(lb):
    """
    Used to cast floats into binary classes: 1 or -1 (zero ties -> -1)

    :param lb: label x_i
    :return: int, 1 or -1
    """
    return 1 if lb > 0 else -1


def update_weights(weights, bias, label, features):
    for i in range(len(label)):
        g = label[i] * sign(weights.dot(features[i].T) + bias)
        if g <= 0:  # check if any labels have negative polarity
            weights = weights + (label[i] * features[i])
            bias += label[i]
    return weights, bias


def perceptron(x, y):
    """
    Wrapper function to create a perceptron classifier from linearly separable data

    :param x: covariates in training dataset
    :param y: labels for training
    :param output: optional, str - name of a file to write weights & bias from stout
    :return: N/A, prints results to console & writes to file specified by argument
    """
    y = np.array(list(map(sign, y)))
    weights, bias = np.array([0] * (x.shape[1])), 0
    converged = False
    out = []
    xlin = np.linspace(x.min(), x.max(), 1000)
    while not converged:
        new_w, new_b = update_weights(weights, bias, y, x)
        ax.plot(xlin, (-weights[0] / weights[1]) * xlin + (-bias / weights[1]), c='g', ls='--', lw=1.5)
        if np.array_equal(np.append(weights, bias), np.append(new_w, new_b)):
            converged = True
            ax.plot(xlin, (-weights[0] / weights[1]) * xlin + (-bias / weights[1]), c='k', ls='-', lw=2)
        else:
            weights, bias = new_w, new_b
            out.append(','.join(map(str, np.append(weights, bias))))
            print(out[-1])
    return out


if __name__ == '__main__':
    in_file, out_file = argv[1:3]

    data = np.genfromtxt(in_file, delimiter=',')
    # Assume the last column is labels:
    n_X = data.shape[1] - 1
    features, labels = data[:, :n_X], data[:, n_X]

    fig, ax = plt.subplots(figsize=(11, 8))
    classes = np.unique(np.array(list(map(sign, data[2]))))
    for cls in classes:
        this = data[data[:, 2] == cls]
        ax.plot(this[:, 0], this[:, 1], marker='x' if cls > 0 else 'o', linestyle='', ms=12)

    stout = perceptron(features, labels)
    plt.savefig(out_file + '.png')
    if out_file is not None:
        with open(out_file, 'w+') as fo:
            fo.write('\n'.join(stout))


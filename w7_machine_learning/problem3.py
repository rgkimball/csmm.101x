"""
Submission for Project 3, Part 3 of Columbia University's AI EdX course (Machine Learning: Classification).

    author: @rgkimball
    date: 4/1/2021
"""

from sys import argv
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def run_algo(fn, params, X, y, X_test, y_test):
    print(f'Running {fn}')
    model = GridSearchCV(fn(), params, cv=5, scoring='accuracy')
    model.fit(X, y)
    return model.best_params_, model.best_score_, model.score(X_test, y_test)


def run_all(features, labels, split=0.6, cfg=None):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=1 - split, stratify=labels)

    results = {k: run_algo(
        cfg[k][0], cfg[k][1],
        X=X_train, y=y_train,
        X_test=X_test, y_test=y_test,
    ) for k in CFG.keys()}

    return results


if __name__ == '__main__':

    CFG = {
        'svm_linear': (
            SVC,
            {
                'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                'kernel': ['linear'],
            },
        ),
        'svm_polynomial': (
            SVC,
            {
                'C': [0.1, 1, 3],
                'kernel': ['poly'],
                'gamma': [0.1, 0.5],
                'degree': [4, 5, 6],
            },
        ),
        'svm_rbf': (
            SVC,
            {
                'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                'kernel': ['rbf'],
                'gamma': [0.1, 0.5, 1, 3, 6, 10],
            },
        ),
        'logistic': (
            SVC,
            {
                'C': [0.1, 0.5, 1, 5, 10, 50, 100],
                'kernel': ['sigmoid'],
            },
        ),
        'knn': (
            KNeighborsClassifier,
            {
                'n_neighbors': list(range(1, 51)),
                'leaf_size': list(np.arange(5, 65, 5))
            },
        ),
        'decision_tree': (
            DecisionTreeClassifier,
            {
                'max_depth': list(range(1, 51)),
                'min_samples_split': list(range(2, 11))
            },
        ),
        'random_forest': (
            RandomForestClassifier,
            {
                'max_depth': list(range(1, 51)),
                'min_samples_split': list(range(2, 11))
            },
        ),
    }

    in_file, out_file = argv[1:3]

    data = np.genfromtxt(in_file, delimiter=',', skip_header=1)
    # Assume the last column is labels:
    n_X = data.shape[1] - 1
    features, labels = data[:, :n_X], data[:, n_X]

    stout = run_all(features, labels, cfg=CFG)

    if out_file is not None:
        with open(out_file, 'w+') as fo:
            for name, results in stout.items():
                best_param, best_score, score = results
                fo.write(f'{name},{best_score},{score}\n')
    print('Done')

"""
Submission for Project 5 of Columbia University's AI EdX course (NLP).

    author: @rgkimball
    date: 4/22/2021
"""

import os
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection._search_successive_halving import HalvingRandomSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Local
from utils import cfg, log, env
from preprocessing import preprocess


def load_data(split):
    log.info(f'Loading {split}ing data.')
    if not os.path.isfile(cfg[env][f'{split}_combined']):
        preprocess(split)
    df = pd.read_csv(cfg[env][f'{split}_combined'])
    return df['text'].values, df['polarity'].values


sgd_params_grid = {
    'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber',
             'epsilon_insensitive', 'squared_epsilon_insensitive'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': np.delete(np.round(abs(np.random.normal(0.0003, 0.0003, 30)), 4), 0),
    'l1_ratio': np.round(np.random.uniform(0, 1, 30), 2),
    'epsilon': np.round(np.random.uniform(0, 1, 30), 2),
}


def is_number(s):
    """
    Used to convert config file numeric values to floats.

    :param s: string of a number: 0, 0.00, -0.05 are all valid
    :return: bool, True if we can convert it otherwise False
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def train_sgd(model_name, vectorizer, train_X, test_X, train_y, grid_search=False):
    log.info(f'Vectorizing {model_name} vocabulary.')
    vec = vectorizer.fit_transform(train_X)
    test_vec = vectorizer.transform(test_X)

    sgd_params = {k: float(v) if is_number(v) else v for k, v in cfg[model_name].items() if k != 'output'}
    if grid_search:
        log.info('Performing grid search over model parameters.')
        model = HalvingRandomSearchCV(
            SGDClassifier(verbose=1), sgd_params_grid, cv=5, scoring='accuracy', n_jobs=10,
        )
        # Switch off convergence warnings during search
        warnings.filterwarnings(
            action='ignore',
            message='ConvergenceWarning: Maximum number of iteration reached before convergence.'
        )
        model.fit(vec, train_y)

        pd.DataFrame(model.cv_results_).to_csv(os.path.join(cfg['data']['out'], f'grid_{model_name}.csv'), index=False)
        sgd_params = model.best_params_

    log.info(f'Training classifier, hyperparameters: {sgd_params}')
    sgd = SGDClassifier(**sgd_params).fit(vec, train_y)

    in_sample = sgd.predict(vec)
    accuracy = accuracy_score(train_y, in_sample, normalize=True)
    log.info(f'In-sample accuracy, {model_name} model: {accuracy}')

    out_sample = sgd.predict(test_vec)

    np.savetxt(
        os.path.join(cfg[env]['output'], cfg[model_name]['output']),
        out_sample, fmt='%i', delimiter=','
    )


if __name__ == "__main__":

    train_input, train_labels = load_data('train')
    test_input, test_labels = load_data('test')

    """ Problem #1:
    Train a SGD classifier using unigram representation,
    predict sentiments on imdb_te.csv, 
    and write output to unigram.output.txt
    """
    train_sgd(
        'unigram_sgd',
        CountVectorizer(analyzer='word', min_df=1),
        train_X=train_input,
        train_y=train_labels,
        test_X=test_input,
    )

    """ Problem #2
    Train a SGD classifier using bigram representation,
    predict sentiments on imdb_te.csv,
    and write output to bigram.output.txt
    """
    train_sgd(
        'bigram_sgd',
        CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1),
        train_X=train_input,
        train_y=train_labels,
        test_X=test_input,
    )

    """ Problem #3
    Train a SGD classifier using unigram representation with tf-idf,
    predict sentiments on imdb_te.csv, 
    and write output to unigramtfidf.output.txt
    """
    train_sgd(
        'unigram_tfidf',
        TfidfVectorizer(min_df=1),
        train_X=train_input,
        train_y=train_labels,
        test_X=test_input,
    )

    """ Problem #4
    Train a SGD classifier using bigram representation with tf-idf,
    predict sentiments on imdb_te.csv,
    and write output to bigramtfidf.output.txt
    """
    train_sgd(
        'bigram_tfidf',
        TfidfVectorizer(min_df=1, ngram_range=(1, 2)),
        train_X=train_input,
        train_y=train_labels,
        test_X=test_input,
    )

    log.info('Done, exiting.')

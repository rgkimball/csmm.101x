"""
Submission for Project 5 of Columbia University's AI EdX course (NLP).

    author: @rgkimball
    date: 4/22/2021
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
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


def train_sgd(model_name, vectorizer, train_X, test_X, train_y):
    log.info(f'Vectorizing {model_name} vocabulary.')
    vec = vectorizer.fit_transform(train_X)

    sgd_params = {k: v for k, v in cfg[model_name].items() if k != 'output'}
    log.info(f'Training classifier, hyperparameters: {sgd_params}')
    sgd = SGDClassifier(**sgd_params).fit(vec, train_y)

    in_sample = sgd.predict(vec)
    accuracy = accuracy_score(train_y, in_sample, normalize=True)
    log.info(f'In-sample accuracy, {model_name} model: {accuracy}')

    # Ensure our columns are consistent with the fitted vocabulary
    vec = vectorizer.transform(test_X)
    out_sample = sgd.predict(vec)

    # FIXME DELETE PLEASE:
    accuracy = accuracy_score(test_labels, out_sample, normalize=True)
    log.info(f'OOS accuracy, {model_name} model: {accuracy}')

    np.savetxt(cfg[model_name]['output'], out_sample, fmt='%i', delimiter=',')


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

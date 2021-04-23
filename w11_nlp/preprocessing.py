"""
Preprocessing functions, Project 5 of Columbia University's AI EdX course (NLP).


    author: @rgkimball
    date: 4/22/2021
"""

# Std Lib
import os
# PyPI
import pandas as pd
# Local
from utils import cfg, log, env, clean_text, file_as_list, progress


def preprocess(split='train'):

    pos, neg = map(lambda x: os.path.join(cfg[env][split], x), ('pos', 'neg'))
    data_sets = [
        ('positive', pos, 1),
        ('negative', neg, 0),
    ]

    df = pd.DataFrame()
    for name, direc, label in data_sets:
        data = []
        for example in progress(os.listdir(direc), suffix=name):
            this = file_as_list(os.path.join(direc, example))
            data.append((list(map(clean_text, this)), example))

        this = pd.DataFrame(
            [(line, label) for file in data for line in file[0]],
            columns=['text', 'polarity'],
        )
        df = df.append(this)
    df.reset_index(inplace=True, drop=True)

    log.info('Saving pre-processed dataframe.')
    df.to_csv(cfg[env][f'{split}_combined'], index=True)

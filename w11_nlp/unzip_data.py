"""
Data extraction, used in preparation for Project 5 of Columbia University's AI EdX course (NLP).

Required tar archive is available here: http://ai.stanford.edu/~amaas/data/sentiment/
Then update the location in config.ini and run this script to unpack it.

    author: @rgkimball
    date: 4/22/2021
"""

import os
from configparser import ConfigParser
import tarfile as tf

if __name__ == '__main__':
    cfg = ConfigParser()
    cfg.read('config.ini')

    archives = filter(lambda x: x.endswith('.tar.gz'), os.listdir(cfg['data']['raw']))

    for file in archives:
        tar = tf.TarFile.open(os.path.join(cfg['data']['raw'], file))
        tar.extractall(cfg['data']['raw'])

import re
import sys
import logging
from configparser import ConfigParser
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)d:[%(name)s:%(lineno)s]: %(message)s')
log = logging.getLogger('base')


def detect_vocaerum():
    """
    Used to dynamically choose which files to access based on whether we are currently running from the Vocaerum env.
    """
    return '/home/vlibs/python' in sys.path


env = 'remote' if detect_vocaerum() else 'local'


def file_as_list(filename):
    with open(filename, 'r', encoding='utf8') as fo:
        data = map(lambda x: x.replace('\n', ''), fo.readlines())
    return list(data)


def progress(iterable, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd='\r'):
    """
    Implements a basic progress bar wrapping function for iterable objects, like the following:

    |██████████████████████████████████████████████████| 100.0%

    :param iterable: object to iterate over (can be fed into a loop or similar iter operation.
    :param prefix: str, text to print before the bar
    :param suffix: str, text to print after the bar
    :param decimals: int, decimals for the precision of the percentage
    :param length: int, to specify the width of the bar
    :param fill: str character to fill the completed sections with.
    :param printEnd: str end character, typically a new line to reset the console.
    :return: prints a progress bar at each iteration
    """
    total = len(iterable)

    def print_progress(iteration):
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filled = int(length * iteration // total)
        bar = fill * filled + '-' * (length - filled)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

    print_progress(0)
    for i, item in enumerate(iterable):
        yield item
        print_progress(i + 1)
    print()


cfg = ConfigParser()
cfg.read('config.ini')
stopwords = file_as_list(cfg['data']['stopwords'])
# Optional, extend the class-provided set using sklearn's default stopwords
stopwords += ENGLISH_STOP_WORDS
stopwords = list(set(stopwords))

word_chars = re.compile(r'[\W_^\s\/\\]')
html = re.compile(r'<[^>]*>')


def clean_text(text):
    """
    Sequence of steps to be applied to raw review data to normalize or otherwise prepare it for modeling.

    :param text: str
    :return: str, but cleaner.
    """
    # Remove HTML tags and punctuation (but retaining slashes, since things like dates and "10/10" are common)
    text = html.sub(' ', text)
    text = word_chars.sub(' ', text)
    # Normalize all to lowercase
    text = text.lower()
    # Remove stopwords
    words = filter(lambda x: x not in stopwords, text.split())
    # Reassemble and return
    text = ' '.join(words)
    return text

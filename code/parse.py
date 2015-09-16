#!usr/bin/env python
# -*- coding: utf-8! -*-

from __future__ import print_function

from collections import Counter, namedtuple, OrderedDict
import os
import string
import glob
import random as rnd
from operator import itemgetter

import matplotlib.pyplot as plt

from nltk.tokenize import wordpunct_tokenize
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import analysis

plt.rcdefaults()

# define a data structure for our metadata:
Oeuvre = namedtuple('Oeuvre', ['title_en_long', 'title_en', 'start_date', 'pub_date_en',
                               'container_text', 'title_fr_long', 'title_fr', 'pub_date_fr',
                               'orig_lang'])

def parse_metadata():
    """
    Parse the metadata for Beckett's oeuvre in metadata.csv.
    """
    lines = []
    for line in open('../data/beckett_metadata.csv', 'r'):
        line = line.strip().lower()
        if not line:
            continue
        cells = [c.strip() for c in line.split(";")]
        cells = [int(c) if c.isdigit() else c for c in cells]
        lines.append(cells)
    lines = tuple(zip(*lines))
    return Oeuvre(*lines)

def tokenize(text):
    """
    Tokenize a string using nltk's tokenizer, lowercasing all words
    and removing tokens that are not purely alphabetical.
    """
    text = text.lower() # lowercase
    words = wordpunct_tokenize(text) # wrap around nltk's tokenizer
    words = [w.strip() for w in words if w.strip()] # rm empty words
    words = [w for w in words if w.isalpha()] # remove non-alphabetic words
    return words

def load_texts(lang='en', min_len=None):
    """
    Load and tokenize all the texts we have for a given language.
    """
    corpus = OrderedDict()
    for filepath in sorted(glob.glob('../data/'+lang+'/*.txt')):
        text_name = os.path.splitext(os.path.basename(filepath))[0][:-4] # rm extension etc. from filename
        words = tokenize(text=open(filepath, 'r', encoding='utf8').read())
        if min_len:
            if len(words) <= min_len:
                continue
        corpus[text_name] = words
    return corpus

def metadata_to_table(metadata, min_len=1000):
    """
    Create a metadata table (including word counts after tokenization).
    """
    print('>>> Creating metadata table')
    fr_corpus = load_texts(lang='fr', min_len=min_len)
    en_corpus = load_texts(lang='en', min_len=min_len)

    # convert to a pandas dataframe for easy processing:
    df = pd.DataFrame(list(zip(*metadata)), columns=metadata._fields)

    # collect word counts (where available):
    en_cnts, fr_cnts = [], []
    for t_en, t_fr in zip(metadata.title_en, metadata.title_fr):
        try:
            en_cnts.append(len(en_corpus[t_en]))
        except KeyError:
            en_cnts.append("NA")
        try:
            fr_cnts.append(len(fr_corpus[t_fr]))
        except KeyError:
            fr_cnts.append("NA")
    
    # add word counts to the dataframe:
    df['word counts (en)'] = en_cnts
    df['word counts (fr)'] = fr_cnts

    # write our dataframe away to Excel format:
    df = df.set_index('start_date')
    df.to_excel('Table1.xlsx')


def extract_mfw(corpus, nb=50):
    """
    Iterate over the tokenized texts in a corpus and extract the nb mfw.
    """
    cnt = Counter()
    for text_name, words in corpus.items():
        cnt.update(words)
    return set([w for w, c, in cnt.most_common(nb)])

def save_mfw(mfw, filepath='mfw.txt'):
    """
    Save an alphabetically sorted mfw list to a file for manual culling.
    """
    with open(filepath, 'w', encoding='utf8') as f:
        for w in sorted(mfw):
            f.write(w+'\n')

def load_mfw(filepath='mfw.txt'):
    """
    Load the mfw from a file (potentially culled using hashtags).
    """
    mfw = set()
    for line in open(filepath, 'r'):
        line = line.strip()
        if line and not line.startswith('#'):
            mfw.add(line)
    return mfw


def slice_corpus(corpus, slice_size=1000, rnd_sample=False):
    """
    Takes a dict of texts and returns a list of evenly sliced samples.
    Samples are consecutive but non-overlapping (see indixes added to the names).
    If rnd_sample is True, only a single randomly selected sample is returned.
    """
    sampled_corpus = OrderedDict()

    for text, words in corpus.items():

        # generate idx tuples:
        start_idx, end_idx, idxs = 0, slice_size, []
        while end_idx < len(words):
            idxs.append((start_idx, end_idx))
            start_idx += slice_size
            end_idx += slice_size

        if not rnd_sample:

            for i, idx in enumerate(idxs):
                slice_ = words[idx[0]: idx[1]]
                sampled_corpus[text+'_'+str(i+1)] = slice_ # add idx per text

        else:
            idx = rnd.sample(idxs, 1)[0] # randomly select a slice
            slice_ = words[idx[0]:idx[1]]
            sampled_corpus[text+'_0'] = slice_ # add idx per text

    return sampled_corpus

def vectorize(samples, vocab):

    def identity(x):
        return x

    vectorizer = TfidfVectorizer(analyzer=identity, vocabulary=vocab, use_idf=False)
    X = vectorizer.fit_transform(samples.values()).toarray()
    X = StandardScaler().fit_transform(X)
    return vectorizer, X

def temporal_sort(corpus_matrix, corpus_names, sort_dates):
    dates = [sort_dates[t.split('_')[0]] for t in corpus_names]
    zipped = zip(corpus_matrix, corpus_names, dates)
    sorted_ = sorted(zipped, key=itemgetter(2))
    corpus_matrix, corpus_names, _ = zip(*sorted_)
    return corpus_matrix, corpus_names


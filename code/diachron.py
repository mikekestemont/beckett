import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sb
sb.set_style("dark")

import os
import string
import codecs
import glob
from operator import itemgetter
from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import PCA

from HACluster import *
import PLM

from nltk.tokenize import wordpunct_tokenize

def identity(x):
    return x

Oeuvre = namedtuple('Oeuvre', ['dates', 'titles', 'texts'])

def load_data(genres=['prose'], data_dir="../data",
              min_nb_tokens=1000):
    items = []
    # iterate over relevant genres:
    for genre in genres:
        for filename in glob.glob(data_dir+"/"+genre+"/*.txt"):
            print "\t+ "+filename,
            with codecs.open(filename, 'r', 'utf-8') as F:
                words = wordpunct_tokenize(F.read().lower())
            if len(words) >= min_nb_tokens:
                print ">>> "+str(len(words))+" words loaded:",
                print (" ".join(words[:6])).strip()
                genre, date, title = os.path.basename(filename).replace(".txt", "").split("_")
                date = int(date)
                items.append((date, title, words))
            else:
                print ">>> file too short"
    # sort texts chronologically:
    items.sort(key=itemgetter(0))
    return Oeuvre(*zip(*items))

def sample(oeuvre, sample_size=2500):
    dates, titles, samples = [], [], []
    for date, title, text in zip(*oeuvre):
        if len(text) > sample_size: # more than one sample
            start_idx, end_idx, cnt = 0, sample_size, 0
            while end_idx <= len(text):
                dates.append(date)
                titles.append(str(title)+"_"+str(cnt+1))
                samples.append(text[start_idx:end_idx])
                cnt+=1
                start_idx+=sample_size
                end_idx+=sample_size
        else:
            dates.append(str(date)+"_1")
            titles.append(str(title)+"_1")
            samples.append(text)
    return Oeuvre(dates, titles, samples)

def load_stopwords(filepath="../data/stopwords.txt"):
    return set(codecs.open(filepath, 'r', 'utf-8').read().lower().split())

sample_size = 1000
genres = ['drama']

oeuvre = load_data(genres=genres, min_nb_tokens=sample_size)
oeuvre = sample(oeuvre=oeuvre, sample_size=sample_size)
stopwords = load_stopwords()

vectorizer = TfidfVectorizer(analyzer=identity,
                             vocabulary=stopwords,
                             #max_features=1000,
                             use_idf=False)
X = vectorizer.fit_transform(oeuvre.texts).toarray()

def vnc():
    dist_matrix = DistanceMatrix(X, lambda u,v: np.sum((u-v)**2)/2)
    # initialize a clusterer, with default linkage methode (Ward)
    clusterer = VNClusterer(dist_matrix)
    # start the clustering procedure
    clusterer.cluster(verbose=0)
    # plot the result as a dendrogram
    clusterer.dendrogram().draw(title="Becket's oeuvre - VNC analysis",#clusterer.linkage.__name__,
                                labels=oeuvre.titles,#oeuvre.dates,
                                show=False, save=True,
                                fontsize=3)
#vnc()

def plm(break_date=1955, nb=50):
    big_docs = {"before":[], "after":[]}
    for text, date in zip(oeuvre.texts, oeuvre.dates):
        if date < break_date:
            big_docs["before"].extend(text)
        else:
            big_docs["after"].extend(text)
    plm = PLM.ParsimoniousLM(big_docs.values(), 0.1)
    plm.fit(big_docs.values(), big_docs.keys())
    for category, lm in plm.fitted_:
        print category
        words = plm.vectorizer.get_feature_names()
        scores = []
        for word, score in sorted(zip(words, lm), key=lambda i:i[1], reverse=True)[:nb]:
            scores.append((word, np.exp(score)))
        print scores
#plm()

def tau(nb=10):
    from scipy.stats import kendalltau

    df = pd.DataFrame(X)
    df.columns = vectorizer.get_feature_names()
    df.index = oeuvre.titles

    scores = []
    ranks = range(1,len(df.index)+1)
    for feat in df.columns:
        tau, p = kendalltau(ranks, df[feat].tolist())
        scores.append((feat, tau))
    scores.sort(key=itemgetter(1))

    nb = 5
    top, bottom = scores[:nb], scores[-nb:]

    fig = sb.plt.figure()
    sb.set_style("darkgrid")
    for (feat, tau), col in zip(top, sb.color_palette("Set1")[:nb]):
        sb.plt.plot(ranks, df[feat].tolist(), label=feat, c=col)
    sb.plt.legend(loc="best")
    sb.plt.xlabel('Diachrony', fontsize=10)
    sb.plt.ylabel('Frequency', fontsize=10)
    sb.plt.savefig("top_tau.pdf")

    fig = sb.plt.figure()
    sb.set_style("darkgrid")
    for (feat, tau), col in zip(bottom, sb.color_palette("Set1")[:nb]):
        sb.plt.plot(ranks, df[feat].tolist(), label=feat, c=col)
    sb.plt.legend(loc="best")
    sb.plt.xlabel('Diachrony', fontsize=10)
    sb.plt.ylabel('Frequency', fontsize=10)
    sb.plt.savefig("bottom_tau.pdf")
tau()

def ngram_viewer(items=[]):
    items = set(items)
    df = pd.DataFrame(X)
    df.columns = vectorizer.get_feature_names()
    df.index = oeuvre.titles
    ranks = range(1,len(df.index)+1)

    fig = sb.plt.figure()
    sb.set_style("darkgrid")

    # remove OOV items
    items = {item for item in items if item in df}

    for item, colour in zip(items, sb.color_palette("Set1")[:len(items)]):
        sb.plt.plot(ranks, df[item].tolist(), label=item, c=colour)
    sb.plt.legend(loc="best")
    sb.plt.xlabel('Diachrony', fontsize=10)
    sb.plt.ylabel('Frequency', fontsize=10)
    sb.plt.savefig("ngram_viewer.pdf")
#ngram_viewer(["no", "less", "neither"])

# un- als prefix?
# leestekens beter weglaten

def pca():
    import pylab as Plot
    # scale X:
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    P = PCA(n_components=2)
    Xr = P.fit_transform(Xs)
    loadings = P.components_.transpose()
    sb.set_style("darkgrid")
    fig, ax1 = plt.subplots()
    #Plot.tick_params(axis='both',which='both',top='off', left='off', right="off", bottom="off", labelbottom='off', labelleft="off", labelright="off")
    # first samples:
    x1, x2 = Xr[:,0], Xr[:,1]
    ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none');
    for x,y,l in zip(x1, x2, oeuvre.titles):
        print(l)
        ax1.text(x, y, l ,ha='center', va="center", size=10, color="darkgrey")
    # now loadings:
    sb.set_style("dark")
    ax2 = ax1.twinx().twiny()
    l1, l2 = loadings[:,0], loadings[:,1]
    ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');
    for x,y,l in zip(l1, l2, vectorizer.get_feature_names()):
        l = l.encode('utf8')
        print(l)
        ax2.text(x, y, l ,ha='center', va="center", size=10, color="black")
    plt.savefig("pca.pdf", bbox_inches=0)
#pca()










#!usr/bin/env python
# -*- coding: utf-8! -*-

from collections import Counter, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances
from librosa.segment import agglomerative
from HACluster import VNClusterer, Clusterer
from ete3 import Tree, NodeStyle, TreeStyle, AttrFace, faces, TextFace


class OrderedCounter(Counter, OrderedDict):
     'Counter that remembers the order elements are first encountered'

     def __repr__(self):
         return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

     def __reduce__(self):
         return self.__class__, (OrderedDict(self),)


def pca_cluster(slice_matrix, slice_names, feature_names, prefix='en',
                nb_clusters=3):
    """
    Run pca on matrix and visualize samples in 1st PCs, with word loadings projected
    on top. The colouring of the samples is provided by running a cluster analysis
    on the samples in these first dimensions. 
    """
    sns.set_style('dark')
    sns.plt.rcParams['axes.linewidth'] = 0.2
    fig, ax1 = sns.plt.subplots()    

    
    pca = PCA(n_components=2)
    pca_matrix = pca.fit_transform(slice_matrix)
    pca_loadings = pca.components_.transpose()
    
    # first plot slices:
    x1, x2 = pca_matrix[:,0], pca_matrix[:,1]
    ax1.scatter(x1, x2, 100, edgecolors='none', facecolors='none')

    # clustering on top (for colouring):
    clustering = AgglomerativeClustering(linkage='ward', affinity='euclidean', n_clusters=nb_clusters)
    clustering.fit(pca_matrix)

    # add slice names:
    for x, y, name, cluster_label in zip(x1, x2, slice_names, clustering.labels_):
        ax1.text(x, y, name.split('_')[0][:3], ha='center', va="center",
                 color=plt.cm.spectral(cluster_label / 10.),
                 fontdict={'family': 'Arial', 'size': 10})

    # now loadings on twin axis:
    ax2 = ax1.twinx().twiny()
    l1, l2 = pca_loadings[:,0], pca_loadings[:,1]
    ax2.scatter(l1, l2, 100, edgecolors='none', facecolors='none');

    for x, y, l in zip(l1, l2, feature_names):
        ax2.text(x, y, l ,ha='center', va="center", size=8, color="darkgrey",
            fontdict={'family': 'Arial', 'size': 9})
    
    # control aesthetics:
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    sns.plt.tight_layout()
    sns.plt.savefig('../outputs/'+prefix+'_pca.pdf', bbox_inches=0)
    plt.clf()

def natural_cluster(slice_matrix, slice_names, prefix='en'):
    """
    Perform plain cluster analysis on sample matrix, without
    taking into account the chronology of the corpus.
    """
    dist_matrix = pairwise_distances(slice_matrix, metric='euclidean')
    clusterer = Clusterer(dist_matrix, linkage='ward')
    clusterer.cluster(verbose=0)
    short_names = [l.split('_')[0][:5]+l.split('_')[1] for l in slice_names]
    tree = clusterer.dendrogram.ete_tree(short_names)
    tree.write(outfile='../outputs/'+prefix+'_natural_clustering.newick')

def vnc_cluster(slice_matrix, slice_names, prefix='en'):
    dist_matrix = pairwise_distances(slice_matrix, metric='euclidean')
    clusterer = VNClusterer(dist_matrix, linkage='ward')
    clusterer.cluster(verbose=0)
    short_names = [l.split('_')[0][:5]+l.split('_')[1] for l in slice_names]
    t = clusterer.dendrogram.ete_tree(short_names)
    t.write(outfile='../outputs/'+prefix+"_vnc_clustering.newick")

def segment_cluster(slice_matrix, slice_names, nb_segments):
    slice_matrix = np.asarray(slice_matrix).transpose() # librosa assumes that data[1] = time axis
    segment_starts = agglomerative(data=slice_matrix, k=nb_segments)
    break_points = []
    for i in segment_starts:
        if i > 0: # skip first one, since it's always a segm start!
            break_points.append(slice_names[i])
    return(break_points)


def bootstrap_segmentation(n_iter, nb_mfw_sampled, corpus_matrix,
                           slice_names, prefix='en', nb_segments=3, random_state=2015):
    np.random.seed(random_state)

    corpus_matrix = np.asarray(corpus_matrix)
    sample_cnts = OrderedCounter()
    for sn in slice_names:
        sample_cnts[sn] = []
        for i in range(nb_segments):
            sample_cnts[sn].append(0)

    for nb in range(n_iter):
        print('===============\niteration:', nb+1)
        # sample a subset of the features in our matrix:
        rnd_indices = np.random.randint(low=0, high=corpus_matrix.shape[1], size=nb_mfw_sampled)
        sampled_matrix = corpus_matrix[:,rnd_indices]
    
        # get which breaks are selected and adjust the cnts:
        selected_breaks = segment_cluster(sampled_matrix, slice_names, nb_segments=nb_segments)
        for i, break_ in enumerate(selected_breaks):
            sample_cnts[break_][i] += 1

    plt.rcParams['font.family'] = 'arial'
    plt.rcParams['font.size'] = 8
    plt.clf()
    plt.figure(figsize=(10,20))

    sample_names, breakpoints_cnts = zip(*sample_cnts.items())
    pos = [i for i, n in enumerate(sample_names)][::-1] # reverse for legibility
    plt.yticks(pos, [n[:3].replace('_', '') if n.endswith(('_1', '_0')) else ' ' for n in sample_names])

    axes = plt.gca()
    axes.set_xlim([0,n_iter])
    colors = sns.color_palette('hls', nb_segments)

    for i in range(nb_segments-1):
        cnts = [c[i] for c in breakpoints_cnts]
        plt.barh(pos, cnts, align='center', color=colors[i], linewidth=0, label="Boundary "+str(i+1))

    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='on')
    plt.tick_params(axis='x', which='both', top='off')
    plt.legend()
    plt.savefig('../outputs/'+prefix+'_bootstrap_segment'+str(nb_segments)+'.pdf')



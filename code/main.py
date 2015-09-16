import os

import parse
import analysis

###################################################
#### PREPROCESSING ################################
###################################################

# make sure that we have an output dir:
if not os.path.isdir('../outputs/'):
  os.mkdir('../outputs/')

# we parse the data from metadata.csv:
oeuvre = parse.parse_metadata()

# get the complete english/french corpus and extract the mfw:
"""
en_corpus = parse.load_texts(lang='en', min_len=None)
en_mfw = parse.extract_mfw(en_corpus, nb=300)
parse.save_mfw(en_mfw, 'mfw_en.txt')

fr_corpus = parse.load_texts(lang='fr', min_len=None)
fr_mfw = parse.extract_mfw(fr_corpus, nb=300)
parse.save_mfw(fr_mfw, 'mfw_fr.txt')
"""

# after manual culling, we reload the mfw:
en_mfw = parse.load_mfw('mfw_en_culled.txt')
fr_mfw = parse.load_mfw('mfw_fr_culled.txt')
print('English mfw:', len(en_mfw), '(', ', '.join(list(en_mfw)[:20]), '...)')
print('French mfw:', len(fr_mfw), '(', ', '.join(list(fr_mfw)[:20]), '...)')


###################################################
#### EXPLORATORY ANALYSES #########################
###################################################
en_corpus = parse.load_texts(lang='en', min_len=4500)
en_sliced_corpus = parse.slice_corpus(en_corpus, slice_size=4500) # length of Ho (sleutelwerk)
vectorizer_en, vectorized_en = parse.vectorize(samples=en_sliced_corpus, vocab=en_mfw)

fr_corpus = parse.load_texts(lang='fr', min_len=4500)
fr_sliced_corpus = parse.slice_corpus(fr_corpus, slice_size=4500)
vectorizer_fr, vectorized_fr = parse.vectorize(samples=fr_sliced_corpus, vocab=fr_mfw)


# natural clustering: generate newick-files which we later manipulate in figtree:
analysis.natural_cluster(vectorized_en, list(en_sliced_corpus.keys()), prefix='en')
analysis.natural_cluster(vectorized_fr, list(fr_sliced_corpus.keys()), prefix='fr')

# pca (for loadings inspection):
analysis.pca_cluster(slice_matrix=vectorized_en, slice_names=list(en_sliced_corpus.keys()),
                     feature_names=vectorizer_en.get_feature_names(), prefix='en',
                     nb_clusters=4)
analysis.pca_cluster(slice_matrix=vectorized_fr, slice_names=list(fr_sliced_corpus.keys()),
                     feature_names=vectorizer_fr.get_feature_names(), prefix='fr',
                     nb_clusters=4)

###################################################
#### VNC ANALYSES #################################
###################################################
en_corpus = parse.load_texts(lang='en', min_len=3761)
en_sliced_corpus = parse.slice_corpus(en_corpus, slice_size=3761)
vectorizer_en, vectorized_en = parse.vectorize(samples=en_sliced_corpus, vocab=en_mfw)

fr_corpus = parse.load_texts(lang='fr', min_len=3761)
fr_sliced_corpus = parse.slice_corpus(fr_corpus, slice_size=3761)
vectorizer_fr, vectorized_fr = parse.vectorize(samples=fr_sliced_corpus, vocab=fr_mfw)

# get a dict with the date for each title:
en_title_to_date = {k:v for k,v in zip(oeuvre.title_en, oeuvre.start_date)}
fr_title_to_date = {k:v for k,v in zip(oeuvre.title_fr, oeuvre.start_date)}

# sort our vectorized matrices:
en_sorted_vectors, en_sorted_names = parse.temporal_sort(corpus_matrix=vectorized_en,
                                       corpus_names=list(en_sliced_corpus.keys()),
                                       sort_dates=en_title_to_date)
fr_sorted_vectors, fr_sorted_names = parse.temporal_sort(corpus_matrix=vectorized_fr,
                                       corpus_names=list(fr_sliced_corpus.keys()),
                                       sort_dates=fr_title_to_date)
# apply vnc analyses:
analysis.vnc_cluster(en_sorted_vectors, en_sorted_names, prefix='en')
analysis.vnc_cluster(fr_sorted_vectors, fr_sorted_names, prefix='fr')


###################################################
#### SEGMENTATION #################################
###################################################

min_len = 3000 #1106
en_corpus = parse.load_texts(lang='en', min_len=min_len)
en_sliced_corpus = parse.slice_corpus(en_corpus, slice_size=min_len)
vectorizer_en, vectorized_en = parse.vectorize(samples=en_sliced_corpus, vocab=en_mfw)

fr_corpus = parse.load_texts(lang='fr', min_len=min_len)
fr_sliced_corpus = parse.slice_corpus(fr_corpus, slice_size=min_len)
vectorizer_fr, vectorized_fr = parse.vectorize(samples=fr_sliced_corpus, vocab=fr_mfw)

# get a dict with the date for each title:
en_title_to_date = {k:v for k,v in zip(oeuvre.title_en, oeuvre.start_date)}
fr_title_to_date = {k:v for k,v in zip(oeuvre.title_fr, oeuvre.start_date)}

# simple segmentation:
en_breakpoints = analysis.segment_cluster(slice_matrix=en_sorted_vectors,
                                          slice_names=en_sorted_names,
                                          nb_segments=5)
print('Breakpoints English oeuvre:', en_breakpoints)
fr_breakpoints = analysis.segment_cluster(slice_matrix=fr_sorted_vectors,
                                          slice_names=fr_sorted_names,
                                          nb_segments=5)
print('Breakpoints French oeuvre:', fr_breakpoints)

# bootstrap segmentation:
analysis.bootstrap_segmentation(n_iter=1000,
                                nb_mfw_sampled=int(len(en_mfw)/10*50),
                                corpus_matrix=en_sorted_vectors,
                                slice_names=en_sorted_names,
                                prefix='en', nb_segments=3)
analysis.bootstrap_segmentation(n_iter=1000,
                                nb_mfw_sampled=int(len(fr_mfw)/10*50),
                                corpus_matrix=fr_sorted_vectors,
                                slice_names=fr_sorted_names,
                                prefix='fr', nb_segments=3)


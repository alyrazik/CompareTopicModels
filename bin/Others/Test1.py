# imports

import plac
import logging
import sys
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import re
import string
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis
from pyLDAvis import sklearn as sklearn_lda
import pickle
from sklearn.datasets import fetch_20newsgroups
import os
from scipy.io import savemat, loadmat
import itertools
import random
import string
from scipy import sparse

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('Basic_logger')

# Constants
Number_Topics = 50
MODEL_PATH = ""
KEYWORDS = ""
min_df = 10
max_df = 0.7
stop_words = []

# command line arguments


@plac.annotations(
    model_path=plac.Annotation(
        'Path where the model is stored.',
        type=str
    ),
    keywords = plac.Annotation(
        'List of strings of the companies to search for in the posts.',
        type=str
    )
)
def main(
        model_path=MODEL_PATH, keywords=KEYWORDS
):
    """ Run LDA2VIZ to test """
    sb.set_style("darkgrid")
    # tokenizer = RegexpTokenizer(r'\b\w{3,}\b')

    # Read data
    print('reading data...')
    train_data = fetch_20newsgroups(subset='train')
    test_data = fetch_20newsgroups(subset='test')
    # [\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"] this means match any alphanumeric word then +| means greedly match
    # the word one or more times then followed by any of the set of symbols in [.,!?;-  ]
    # findall returns a list of all matches within the string (FYI, search finds the indices)
    init_docs_tr = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', train_data.data[doc]) for doc in
                    range(len(train_data.data))]  # 11314 lists in a list, each list contains the words in document.
    init_docs_ts = [re.findall(r'''[\w']+|[.,!?;-~{}`´_<=>:/@*()&'$%#"]''', test_data.data[doc]) for doc in
                    range(len(test_data.data))]

    def contains_punctuation(word):
        return any(character in string.punctuation for character in word)

    def contains_numeric(word):
        return any(character.isdigit() for character in word)

    init_docs = init_docs_tr + init_docs_ts
    init_docs = [[w.lower() for w in init_docs[doc] if not contains_punctuation(w)] for doc in range(len(init_docs))]
    init_docs = [[w for w in init_docs[doc] if not contains_numeric(w)] for doc in range(len(init_docs))]
    init_docs = [[w for w in init_docs[doc] if len(w) > 1] for doc in range(len(init_docs))]
    init_docs = [" ".join(init_docs[doc]) for doc in range(len(init_docs))]  # list of strings (docs)

    # Create count vectorizer; converts a collection (list) of docs to a matrix of token counts.
    # the output format is scipy.sparse.csr_matrix
    # min_df and max_df specify the min/ max proportion of documents where a term should exist for it to be included
    # in the vocabulary as a token.
    print('counting document frequency of words...')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words=stop_words)
    cvz = cvectorizer.fit_transform(init_docs).sign()  # sign returns 1 for positive counts, 0 for no counts.
    # cvz is 18846 documents in rows and 19148 tokens in columns (entries are binary, 1 or 0 to indicate existence of a
    # token or not.

    # Get vocabulary
    print('building the vocabulary...')
    sum_counts = cvz.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0, v]
        # a vector whose elements r the document frequency of each token in vocab across corpus
    id2word = dict([(cvectorizer.vocabulary_.get(w), w) for w in cvectorizer.vocabulary_])
    # cvectorizer.vocabulary_ is a dictionary with tokens as keys and values represent their index.
    del cvectorizer
    print('  initial vocabulary size: {}'.format(v_size))

    # Sort elements in vocabulary
    idx_sort = np.argsort(sum_counts_np)  # [10, 102, 1, 5  ] indices of sorted tokens (in doc freq.)
    vocab_aux = [id2word[idx_sort[cc]] for cc in range(v_size)]
    # vocab_aux is the list of words in corpus from least doc frequency to the most doc frequency.

    # Filter out stopwords (if any)
    vocab_aux = [w for w in vocab_aux if w not in stop_words]
    print('  vocabulary size after removing stopwords from list: {}'.format(len(vocab_aux)))

    # Create dictionary and inverse dictionary
    vocab = vocab_aux
    del vocab_aux
    word2id = dict([(w, n) for n, w in enumerate(vocab)])

    # Split in train/test/valid
    print('tokenizing documents and splitting into train/test/valid...')
    num_docs_tr = len(init_docs_tr)
    trSize = num_docs_tr - 100
    tsSize = len(init_docs_ts)
    vaSize = 100
    idx_permute = np.random.permutation(num_docs_tr).astype(int)  # returns a permuted (random) range(num_docs_tr).

    # Remove words not in train_data
    vocab = list(set([w for idx_d in range(trSize) for w in init_docs[idx_permute[idx_d]].split() if w in word2id]))
    # set takes an iterable and converts it to an unordered set with distinct elements.
    word2id = dict([(w, j) for j, w in enumerate(vocab)])
    id2word = dict([(j, w) for j, w in enumerate(vocab)])
    print('  vocabulary after removing words not in train: {}'.format(len(vocab)))

    # Split in train/test/valid
    docs_tr = [[word2id[w] for w in init_docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(trSize)]
    docs_va = [[word2id[w] for w in init_docs[idx_permute[idx_d + trSize]].split() if w in word2id] for idx_d in
               range(vaSize)]
    docs_ts = [[word2id[w] for w in init_docs[idx_d + num_docs_tr].split() if w in word2id] for idx_d in range(tsSize)]

    print('  number of documents (train): {} [this should be equal to {}]'.format(len(docs_tr), trSize))
    print('  number of documents (test): {} [this should be equal to {}]'.format(len(docs_ts), tsSize))
    print('  number of documents (valid): {} [this should be equal to {}]'.format(len(docs_va), vaSize))

    # Remove empty documents
    print('removing empty documents...')

    def remove_empty(in_docs):
        return [doc for doc in in_docs if doc != []]

    docs_tr = remove_empty(docs_tr)
    docs_ts = remove_empty(docs_ts)
    docs_va = remove_empty(docs_va)

    # Remove test documents with length=1
    docs_ts = [doc for doc in docs_ts if len(doc) > 1]

    # Split test set in 2 halves
    print('splitting test documents in 2 halves...')
    docs_ts_h1 = [[w for i, w in enumerate(doc) if i <= len(doc) / 2.0 - 1] for doc in docs_ts]
    docs_ts_h2 = [[w for i, w in enumerate(doc) if i > len(doc) / 2.0 - 1] for doc in docs_ts]

    # Getting lists of words and doc_indices
    print('creating lists of words...')

    def create_list_words(in_docs):
        return [x for y in in_docs for x in y]

    words_tr = create_list_words(docs_tr)
    words_ts = create_list_words(docs_ts)
    words_ts_h1 = create_list_words(docs_ts_h1)
    words_ts_h2 = create_list_words(docs_ts_h2)
    words_va = create_list_words(docs_va)

    print('  len(words_tr): ', len(words_tr))
    print('  len(words_ts): ', len(words_ts))
    print('  len(words_ts_h1): ', len(words_ts_h1))
    print('  len(words_ts_h2): ', len(words_ts_h2))
    print('  len(words_va): ', len(words_va))

    # Get doc indices
    print('getting doc indices...')

    def create_doc_indices(in_docs):
        aux = [[j for i in range(len(doc))] for j, doc in enumerate(in_docs)]
        return [int(x) for y in aux for x in y]

    doc_indices_tr = create_doc_indices(docs_tr)
    doc_indices_ts = create_doc_indices(docs_ts)
    doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
    doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
    doc_indices_va = create_doc_indices(docs_va)

    print(
        '  len(np.unique(doc_indices_tr)): {} [this should be {}]'.format(len(np.unique(doc_indices_tr)), len(docs_tr)))
    print(
        '  len(np.unique(doc_indices_ts)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts)), len(docs_ts)))
    print('  len(np.unique(doc_indices_ts_h1)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h1)),
                                                                               len(docs_ts_h1)))
    print('  len(np.unique(doc_indices_ts_h2)): {} [this should be {}]'.format(len(np.unique(doc_indices_ts_h2)),
                                                                               len(docs_ts_h2)))
    print(
        '  len(np.unique(doc_indices_va)): {} [this should be {}]'.format(len(np.unique(doc_indices_va)), len(docs_va)))

    # Number of documents in each set
    n_docs_tr = len(docs_tr)
    n_docs_ts = len(docs_ts)
    n_docs_ts_h1 = len(docs_ts_h1)
    n_docs_ts_h2 = len(docs_ts_h2)
    n_docs_va = len(docs_va)

    # Remove unused variables
    del docs_tr
    del docs_ts
    del docs_ts_h1
    del docs_ts_h2
    del docs_va

    # Create bow representation
    print('creating bow representation...')

    def create_bow(doc_indices, words, n_docs, vocab_size):
        return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()

    bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
    bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
    bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
    bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
    bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))

    del words_tr
    del words_ts
    del words_ts_h1
    del words_ts_h2
    del words_va
    del doc_indices_tr
    del doc_indices_ts
    del doc_indices_ts_h1
    del doc_indices_ts_h2
    del doc_indices_va

    # Write the vocabulary to a file
    path_save = './min_df_' + str(min_df) + '/'
    if not os.path.isdir(path_save):
        os.system('mkdir -p ' + path_save)

    with open(path_save + 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    del vocab

    # Split bow intro token/value pairs
    print('splitting bow intro token/value pairs and saving to disk...')

    def split_bow(bow_in, n_docs):
        indices = [[w for w in bow_in[doc, :].indices] for doc in range(n_docs)]
        counts = [[c for c in bow_in[doc, :].data] for doc in range(n_docs)]
        return indices, counts

    bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
    savemat(path_save + 'bow_tr_tokens.mat', {'tokens': bow_tr_tokens}, do_compression=True)
    savemat(path_save + 'bow_tr_counts.mat', {'counts': bow_tr_counts}, do_compression=True)
    del bow_tr
    del bow_tr_tokens
    del bow_tr_counts

    bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
    savemat(path_save + 'bow_ts_tokens.mat', {'tokens': bow_ts_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_counts.mat', {'counts': bow_ts_counts}, do_compression=True)
    del bow_ts
    del bow_ts_tokens
    del bow_ts_counts

    bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
    savemat(path_save + 'bow_ts_h1_tokens.mat', {'tokens': bow_ts_h1_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_h1_counts.mat', {'counts': bow_ts_h1_counts}, do_compression=True)
    del bow_ts_h1
    del bow_ts_h1_tokens
    del bow_ts_h1_counts

    bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
    savemat(path_save + 'bow_ts_h2_tokens.mat', {'tokens': bow_ts_h2_tokens}, do_compression=True)
    savemat(path_save + 'bow_ts_h2_counts.mat', {'counts': bow_ts_h2_counts}, do_compression=True)
    del bow_ts_h2
    del bow_ts_h2_tokens
    del bow_ts_h2_counts

    bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
    savemat(path_save + 'bow_va_tokens.mat', {'tokens': bow_va_tokens}, do_compression=True)
    savemat(path_save + 'bow_va_counts.mat', {'counts': bow_va_counts}, do_compression=True)
    del bow_va
    del bow_va_tokens
    del bow_va_counts

    print('Data ready !!')
    print('*************')


if __name__ == '__main__':
    plac.call(main)


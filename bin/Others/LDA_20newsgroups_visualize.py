from bin.Others.Vanilla_LDA import data_to_lda, print_topics
from sklearn.datasets import fetch_20newsgroups
import string
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
import pyLDAvis
import pyLDAvis.sklearn
import pickle

# configuration

# constants

max_df = 0.7
min_df = 10
stop_words = []
number_topics = 10
number_words = 20


def main():

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

    df = pd.DataFrame(init_docs, columns=['documents'])
    # Create count vectorizer; converts a collection (list) of docs to a matrix of token counts.
    # the output format is scipy.sparse.csr_matrix
    # min_df and max_df specify the min/ max proportion of documents where a term should exist for it to be included
    # in the vocabulary as a token.
    print('counting document frequency of words...')
    cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words='english', strip_accents='unicode',
                                  lowercase=True)
    cvz = cvectorizer.fit_transform(init_docs).sign()  # sign returns 1 for positive counts, 0 for no counts.
    # cvz is 18846 documents in rows and 19148 tokens in columns (entries are binary, 1 or 0 to indicate existence of a
    # token or not.

    lda = LDA(n_components=number_topics,
              doc_topic_prior=1,
              topic_word_prior=0.05,
              n_jobs=-1, random_state=12345)

    lda_data, lda_obj = data_to_lda(df, 'documents', cvectorizer, lda)

    # Print the topics found by the LDA model
    print("Topics found via LDA on Count Vectorised data for ALL categories:")
    print_topics(lda_obj, cvectorizer, number_words)

    display_data = pyLDAvis.sklearn.prepare(lda_obj, cvz, cvectorizer)

    with open('display_data.pkl', 'wb') as f:
        pickle.dump(display_data, f)


if __name__ == '__main__':
    main()
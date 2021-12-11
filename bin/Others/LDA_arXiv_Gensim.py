import pandas as pd
import json
import time
from COTM.Vanilla_LDA import model_from_text
from COTM.cleaning import clean_tokenize
from gensim import models
import gensim
from gensim import corpora
import matplotlib.pyplot as plt
import pickle
import pyLDAvis
from pyLDAvis import gensim_models
import nltk
import re


# Constants
keywords = ['topic modeling', 'neural topic model', 'topic model', 'neural topic modelling']
N_TOPICS = 4  # max number of topics to use in your loop starting from 2 to N_topics-1
N_PASSES = 100
ITERATIONS = 400
CHUNKSIZE = 50
max_df = 0.3
min_df = 0.05
BIGRAMS_FREQUENCY = 10
TRIGRAM_FREQUENCY = 10
DataPath = '//home//mohamed//PycharmProjects//TopicModel//arXiv_metadata//arxiv-metadata-oai-snapshot.json'

# Read file that contains metadata about papers (no paper text)


def get_metadata():
    """define a generator function since file is too big to handle in memory all at once"""
    with open(DataPath, 'r') as file:
        for line in file:
            yield line


def main():
    metadata = get_metadata()  # now metadata is an iterable

    titles = []
    authors = []
    abstracts = []
    update_dates = []

    print("Reading documents...")
    t1 = time.time()
    for paper in metadata:
        paper_dict = json.loads(paper)
        if 'cs.' in paper_dict['categories']:
            abstract = paper_dict['abstract'].lower()
            if any(phrase in abstract for phrase in keywords):
                titles.append(paper_dict['title'])
                authors.append(paper_dict['authors'])
                abstracts.append(abstract)
                update_dates.append(paper_dict['update_date'])
    t = time.time()-t1
    print('Reading time is ', t)

    df = pd.DataFrame(zip(titles, authors, abstracts, update_dates),
                      columns=['title', 'author', 'abstract', 'update_date'])

    # df.dropna(axis=0, subset=['title', 'author', 'abstract', 'update_date'], inplace=True)
    print("Number of documents: ", df.shape[0])

    ##############################################################################################################

    # clean
    print("Cleaning text...")
    t = time.time()
    df['abstract'] = df['abstract'].apply(clean_tokenize)
    print("Cleaning time is ", time.time()-t)
    # construct bigrams and trigrams and include as one token in text (hyphenated)
    # Trigrams:
    print('Constructing Trigrams...')
    finder = nltk.collocations.TrigramCollocationFinder.from_documents(df['abstract'])
    finder.apply_freq_filter(TRIGRAM_FREQUENCY)
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    trigrams_PMIscores = pd.DataFrame(finder.score_ngrams(trigram_measures.pmi),
                                      columns=['trigram', 'pmi'])
    # match trigrams with "noun anything noun" only
    trigrams_PMIscores['tag'] = trigrams_PMIscores['trigram']\
        .apply(lambda x: ' '.join(v for (k, v) in nltk.pos_tag(x)))

    pattern = re.compile(r'NN.* \w.* NN.*')
    selection = trigrams_PMIscores['tag'].apply(lambda x: True if re.search(pattern, ' '.join(x.split())) else False)
    trigrams = trigrams_PMIscores[selection].sort_values(by='pmi', ascending=False)['trigram'].apply(
        lambda x: ' '.join(x))
    print("showing some trigrams...")
    print(trigrams[:10])
    print("Updating corpus with constructed trigrams...")

    def replace_trigrams(s, grams_list):
        for trigram in grams_list:
            s.replace(trigram, trigram + ' ' + '_'.join(trigram.split()))  # add the hyphenated gram to the individual words
        return s
    df['abstract'] = df['abstract'].apply(lambda x: replace_trigrams(' '.join(x), trigrams))
    print(df['abstract'])
    # Bigrams
    print("Constructing Bigrams...")
    finder = nltk.collocations.BigramCollocationFinder.from_documents(df['abstract'].apply(lambda x: x.split()))
    finder.apply_freq_filter(BIGRAMS_FREQUENCY)    # only include grams that occur 50 times or more.
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    bigram_scores = finder.score_ngrams(bigram_measures.pmi)
    bigram_pmi = pd.DataFrame(bigram_scores, columns='bigram pmi'.split())
    # remove bigrams that are not both nouns.
    bigram_pmi['tag'] = bigram_pmi['bigram'].apply(lambda x: (nltk.pos_tag(x)[0][1], nltk.pos_tag(x)[1][1]))
    bigram_pmi = bigram_pmi[bigram_pmi['tag'] == ('NN', 'NN')].sort_values(by='pmi', ascending=False)
    bigrams = bigram_pmi['bigram'].apply(lambda x: x[0] + ' ' + x[1])  # a series of tuples into a string
    print("showing some bigrams", bigrams[:10])

    def replace_bigrams(x, grams_list):
        for gram in grams_list:
            x = x.replace(gram, gram + ' ' + '_'.join(gram.split()))   # add the hyphenated gram to the individual tokens
        return x

    print(df['abstract'])
    print("Updating corpus with constructed bigrams...")
    #   pick the 100 highest pmi-scored bigrams?
    df['abstract'] = df['abstract'].apply(lambda x: replace_bigrams(x, bigrams))

    print("showing abstracts after cleaning and bigrams/trigrams addition...")
    print(df['abstract'])
    ##############################################################################################################

    # topic modelling, LDA
    print("Training the LDA model...")
    t = time.time()
    corpus, id2word = model_from_text(df['abstract'],
                                      max_df=max_df,
                                      min_df=min_df)
    dictionary = corpora.Dictionary()
    dictionary.id2token = id2word
    dictionary.token2id = dict((k, v) for v, k in id2word.items())
    # Iterate over num_topics and calculate best coherence score.
    # print(df['abstract'].head())
    scores = []
    ks = [N_TOPICS]    # range(2, N_TOPICS)
    for k in ks:
        lda = models.LdaModel(corpus=corpus,
                              id2word=id2word,
                              num_topics=k,
                              alpha='auto',
                              eta='auto',
                              passes=N_PASSES,
                              iterations=ITERATIONS,
                              chunksize=CHUNKSIZE,
                              eval_every=5)

        coherence_score = gensim.models.coherencemodel.CoherenceModel(model=lda,
                                                                      dictionary=dictionary,
                                                                      texts=df['abstract'].apply(lambda x: x.split()),
                                                                      coherence='u_mass')
        scores.append(coherence_score.get_coherence())
        print('For a number of topics of ', k, ', Coherence score:', coherence_score.get_coherence())
        # To show initial topics
        print("Showing most probable words per topic for number of topics ", k)
        print(lda.show_topics(k, num_words=5, formatted=False))
    print(f'training {k} times took {time.time()-t} seconds.')
    plt.plot(ks, scores)
    plt.scatter(ks, scores)
    plt.title('Number of topics vs. coherence')
    plt.xlabel("Number of topics")
    plt.ylabel("Coherence score")
    plt.show()

    # Run model on data.
    print("Inferring topic distribution of documents...")
    df['topic'] = lda[corpus]  # returns list of tuples in the format (topic number, topic proportion).

    # below is done since the list of tuples doesn't include all topics.
    for c in range(N_TOPICS):
        df[f'topic_{c}'] = df['topic'].apply(
            lambda x: max([topic_proportion if topic == c else 0 for topic, topic_proportion in x]))

    # print(df[['topic', 'topic_0', 'topic_1']][(df['topic_0'] == 0) | (df['topic_1'] == 0)])
    print('Saving document data into "lda_data_gensim.pkl" file ...')
    with open('lda_data_gensim.pkl', 'wb') as f:
        pickle.dump(df, f)
    # note that you have to load the pickled object using same pandas version as in here.
    # so if you used google colab to unpickle the file, it gives an error as colab uses pandas 1.1.x not 1.3.4
    # Print the topics found by the LDA model

    print('Saving display data into "display_data.pkl" file ...')
    display_data = pyLDAvis.gensim_models.prepare(lda, corpus=corpus, dictionary=dictionary)
    with open('display_data_gensim.pkl', 'wb') as file:
        pickle.dump(display_data, file)


if __name__ == '__main__':
    main()

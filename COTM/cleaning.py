# I provide the cleaning routines here.
# Remove non ASCII words
# Lemmatize
# Remove Punctuation
# Remove any non English comments.
# remove a list of stop_words
from gensim.parsing import preprocessing

import nltk
import spacy
# from collections import Counter
nlp = spacy.load('en_core_web_sm')
# import remove_stopwords, preprocess_string, stem, strip_non_alphanum,
# strip_numeric, strip_punctuation, strip_short


def keep_nouns(s):
    tagged = nltk.pos_tag(s)
    filtered = [word[0] for word in tagged if word[1] in ['NN']]
    return filtered


def clean_tokenize(s, stopwords):
    """Performs text cleaning including lemmatization. Returns a clean list of tokens."""
    processed = s.lower()
    processed = preprocessing.strip_tags(processed)
    processed = preprocessing.strip_punctuation(processed)
    processed = preprocessing.strip_short(processed, minsize=3)
    # pattern = re.compile(r'\b('+r'|'.join(STOPWORDS)+r')\b\S*')
    # processed = pattern.sub('', processed)  # slower way to remove stop words.
    processed = processed.split()
    processed = [token for token in processed if token not in stopwords]

    # processed = preprocessing.preprocess_string(processed,
    #                                             filters=[preprocessing.strip_tags,
    #                                                      preprocessing.strip_non_alphanum,
    #                                                      preprocessing.strip_numeric,
    #                                                      preprocessing.strip_punctuation,
    #                                                      preprocessing.strip_multiple_whitespaces,
    #                                                      preprocessing.strip_short,
    #                                                      # preprocessing.stem_text  # use spacy lemmatizer instead
    #                                                      ]
    #                                             )

    # processed = nlp(processed)  # a document object for spacy
    # processed = ' '.join([word.lemma_ for word in processed])
    # processed = keep_nouns(processed)

    # stop_words = ('the datum in word words use uses using user document study related proposed over ' +
    #               'content recommendation large time between problem large each many state ' +
    #               'efficient one corpus network structure relate level research task event  ' +
    #               'collection process matrix information analysis article generate work text language distribution ' +
    #               'paper challenge experiment art propose framework technique algorithm method result paper base' +
    #               'feature sample medium  approach http https')\
    #     .split()
    # stop_words = []  # to experiment with no custom stop words list. remove # to include.
    # stop_words_dict = Counter(stop_words)
    # processed = [word for word in processed if word not in stop_words_dict]
    return processed

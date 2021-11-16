# I provide the cleaning routines here.
# Reomve non ASCII words
# Lemmatize
# Remove Punctutation
# Remove any non English comments.
# remove a list of stop_words
from gensim.parsing import preprocessing
import nltk
import spacy
nlp = spacy.load('en_core_web_sm')
# import remove_stopwords, preprocess_string, stem, strip_non_alphanum,
# strip_numeric, strip_punctuatuion, strip_short
from collections import Counter


def keep_nouns(s):
    tagged = nltk.pos_tag(s)
    filtered = [word[0] for word in tagged if word[1] in ['NN']]
    return filtered


def clean_all(s):
    """Performs text cleaning including lemmatization. Returns a clean list of tokens."""

    processed = nlp(s)  # a document object for spacy
    processed = ' '.join([word.lemma_ for word in processed])
    processed = preprocessing.preprocess_string(processed,
                                                filters=[preprocessing.remove_stopwords,
                                                         preprocessing.strip_tags,
                                                         preprocessing.strip_non_alphanum,
                                                         preprocessing.strip_numeric,
                                                         preprocessing.strip_punctuation,
                                                         preprocessing.strip_multiple_whitespaces,
                                                         preprocessing.strip_short,
                                                         # preprocessing.stem_text  # use spacy lemmatizer instead
                                                         ]
                                                )

    processed = keep_nouns(processed)

    stop_words = ('the datum in word words use uses using user document study related proposed over ' +
                  'content recommendation large time between problem large each many state ' +
                  'efficient one corpus network structure relate level research task event  ' +
                  'collection process matrix information analysis article generate work text language distribution ' +
                  'paper challenge experiment art propose framework technique algorithm method result paper base' +
                  'feature sample medium  approach http https')\
        .split()
    stop_words = []  # to experiment with no custom stop words list. remove # to include.
    stop_words_dict = Counter(stop_words)
    processed = [word for word in processed if word not in stop_words_dict]
    return processed




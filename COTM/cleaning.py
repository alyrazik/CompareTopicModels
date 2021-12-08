# I provide the cleaning routines here.
# Remove non ASCII words
# Lemmatize
# Remove Punctuation
# Remove any non English comments.
# remove a list of stop_words
from gensim.parsing import preprocessing
import nltk
import spacy
import re
# from collections import Counter
nlp = spacy.load('en_core_web_sm')
# import remove_stopwords, preprocess_string, stem, strip_non_alphanum,
# strip_numeric, strip_punctuation, strip_short
STOPWORDS = frozenset([
    'all', 'six', 'just', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'four', 'not', 'own', 'through',
    'using', 'fifty', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere',
    'much', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'yourselves', 'under',
    'ours', 'two', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very',
    'de', 'none', 'cannot', 'every', 'un', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'regarding',
    'several', 'hereafter', 'did', 'always', 'who', 'didn', 'whither', 'this', 'someone', 'either', 'each', 'become',
    'thereupon', 'sometime', 'side', 'towards', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'doing', 'km',
    'eg', 'some', 'back', 'used', 'up', 'go', 'namely', 'computer', 'are', 'further', 'beyond', 'ourselves', 'yet',
    'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its',
    'everything', 'behind', 'does', 'various', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she',
    'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere',
    'although', 'found', 'alone', 're', 'along', 'quite', 'fifteen', 'by', 'both', 'about', 'last', 'would',
    'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence',
    'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others',
    'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover',
    'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due',
    'been', 'next', 'anyone', 'eleven', 'cry', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves',
    'hundred', 'really', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming',
    'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'kg', 'herself', 'former', 'those', 'he', 'me', 'myself',
    'made', 'twenty', 'these', 'was', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere',
    'nine', 'can', 'whether', 'of', 'your', 'toward', 'my', 'say', 'something', 'and', 'whereafter', 'whenever',
    'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'doesn', 'an', 'as', 'itself', 'at',
    'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps',
    'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which',
    'becomes', 'you', 'if', 'nobody', 'unless', 'whereas', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon',
    'eight', 'but', 'serious', 'nothing', 'such', 'why', 'off', 'a', 'don', 'whereby', 'third', 'i', 'whole', 'noone',
    'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'with',
    'make', 'once', 'the', 'for', 'and', 'are', 'edu',
    'subject', 'lines', 'max',
    'com', 'organization', 'writes', 'article', 'msg', 'like', 'people', 'university', 'space', 'posting', 'know'
])


def keep_nouns(s):
    tagged = nltk.pos_tag(s)
    filtered = [word[0] for word in tagged if word[1] in ['NN']]
    return filtered


def clean_tokenize(s):
    """Performs text cleaning including lemmatization. Returns a clean list of tokens."""
    processed = s.lower()
    processed = preprocessing.strip_tags(processed)
    processed = preprocessing.strip_punctuation(processed)
    processed = preprocessing.strip_short(processed, minsize=3)
    # pattern = re.compile(r'\b('+r'|'.join(STOPWORDS)+r')\b\S*')
    # processed = pattern.sub('', processed)  # slower way to remove stop words.
    processed = processed.split()
    processed = [token for token in processed if token not in STOPWORDS]

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

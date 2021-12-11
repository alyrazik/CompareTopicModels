import pandas as pd
import json
import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from bin.Others.Vanilla_LDA import data_to_lda, print_topics
import pyLDAvis.sklearn

# Constants
keywords = ['topic modeling', 'neural topic model', 'topic model', 'neural topic modelling']
min_df = 10
max_df = 0.4
number_topics = 5  # Neural, LDA, evaluation, .. ,...
number_words = 20
doc_topic_prior = 0.2
topic_word_prior = 0.01
n_jobs = -1
random_state = 12345
max_iter = 20
evaluate_every = 1

# Read file that contains metadata about papers (no paper text)


def get_metadata():
    """define a generator function since file is too big to handle in memory all at once"""

    with open('/arXiv_metadata/arxiv-metadata-oai-snapshot.json',
              'r') as file:
        for line in file:
            yield line


metadata = get_metadata()  # now metadata is an iterable

# The file contents
# id
# submitter
# authors
# title
# comments
# journal-ref   (many missing datapoints)
# doi
# report-no  (many missing datapoints)
# categories
# license
# abstract
# versions
# update_date
# authors_parsed

# the below should be faster implementation but it didn't work (expects a bytes object error)
# it doesn't run faster, and the reason for the error was that python3 expect writer to write to a
# string not a bytes object. so you need to define output = io.StringIO() not BytesIO
# import re
# from io import BytesIO
# from csv import writer
# import io
# output = BytesIO()  # an object that resides in memory
# csv_writer = writer(output)  # writes to an in-memory csv object
# for paper in metadata:
#     paper_dict = json.loads(paper)
#     if 'cs.' in paper_dict['categories']:
#         # row = [v for k, v in paper_dict.items() if k in ['title', 'journal-ref']]
#         row = bytes(paper_dict['title'], encoding='utf-8')
#         csv_writer.writerow(row)    # takes a yielded output from an iterable.
#
# output.seek(0)  # to get back to the start of the BytesIO object
# df = pd.read_csv(output)
# df.head()

titles =[]
authors = []
abstracts = []
update_dates = []

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
print('time is ', t)

df = pd.DataFrame(zip(titles, authors, abstracts, update_dates),
                  columns=['title', 'author', 'abstract', 'update_date'])

# df.dropna(axis=0, subset=['title', 'author', 'abstract', 'update_date'], inplace=True)
print("Number of documents: ", df.shape[0])

cvectorizer = CountVectorizer(min_df=min_df,
                              max_df=max_df,
                              stop_words='english',
                              strip_accents='unicode',
                              lowercase=True)

lda = LDA(n_components=number_topics,
          doc_topic_prior=doc_topic_prior,
          topic_word_prior=topic_word_prior,
          n_jobs=n_jobs, random_state=random_state,
          evaluate_every=evaluate_every,
          max_iter=max_iter)

lda_data, lda_obj = data_to_lda(df, 'abstract', cvectorizer, lda)

print(f"Resolved topics for {lda_data.shape[0]} documents")
with open('lda_data.pkl', 'wb') as f:
    pickle.dump(lda_data, f)  # note that you have to load the pickled object using same pandas version as in here.
    # so if you used google colab to unpickle the file, it gives an error as colab uses pandas 1.1.x not 1.3.4


# Print the topics found by the LDA model
print("Topics found via LDA on Count Vectorised data for ALL categories:")
print_topics(lda_obj, cvectorizer, number_words)

cvz = cvectorizer.fit_transform(df['abstract'])
display_data = pyLDAvis.sklearn.prepare(lda_obj, cvz, cvectorizer)
with open('display_data.pkl', 'wb') as file:
    pickle.dump(display_data, file)
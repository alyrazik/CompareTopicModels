import numpy as np
import pandas as pd
import gc
import os
import json
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from COTM.Vanilla_LDA import data_to_lda, print_topics
from collections import Counter, defaultdict
# from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import re
from io import BytesIO
from csv import writer
import io
year_pattern = r'([1-2][0-9]{3})'

keywords = ['topic modeling', 'neural topic model']
min_df = 10
max_df = 0.4
number_topics = 3
number_words = 10


def get_metadata():
    """define a generator function since file is too big to handle in memory all at once"""

    with open('C://Users//Aly//PycharmProjects//TopicModel//arXiv_metadata//arxiv-metadata-oai-snapshot.json',
              'r') as f:
        for line in f:
            yield line


metadata = get_metadata()  # now metadata is an iterable
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
update_date = []

t1 = time.time()
for paper in metadata:
    paper_dict = json.loads(paper)
    if 'cs.' in paper_dict['categories']:
        abstract = paper_dict['abstract'].lower()
        if any(phrase in abstract for phrase in keywords):
            titles.append(paper_dict['title'])
            authors.append(paper_dict['authors'])
            abstracts.append(abstract)
            update_date.append(paper_dict['update_date'])

t = time.time()-t1
print('time is ', t)

df = pd.DataFrame(zip(titles, authors, abstracts, update_date),
                  columns=['title', 'author', 'abstract', 'update_date'])

df.dropna(axis=0, subset=['title', 'author', 'abstract', 'update_date'], inplace=True)

cvectorizer = CountVectorizer(min_df=min_df, max_df=max_df, stop_words='english', strip_accents='unicode',
                              lowercase=True)

lda = LDA(n_components=number_topics,
          doc_topic_prior=1,
          topic_word_prior=0.05,
          n_jobs=-1, random_state=12345)

lda_data, lda_obj = data_to_lda(df, 'abstract', cvectorizer, lda)

# Print the topics found by the LDA model
print("Topics found via LDA on Count Vectorised data for ALL categories:")
print_topics(lda_obj, cvectorizer, number_words)
from gensim import corpora, models
from COTM.datasets import MyCorpus
import logging
import time
from datetime import datetime
import os
import json
import numpy as np
import gensim
from COTM.metrics import diversity
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric, ConvergenceMetric, DiffMetric
from COTM.learning import DiversityMetric
from gensim.models.callbacks import Callback
import pyLDAvis
from pyLDAvis import gensim_models
import pickle
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from gensim import utils

# Setup logging and constants
logging.basicConfig(level=logging.INFO)
num_topics = 20
models_dir = "..\\saved_models"
saving_time = datetime.now().strftime('%Y-%m-%d_%H_%M')  # '%Y-%m-%d_%H_%M'
model_path = os.path.join(models_dir, f'lda_{num_topics}_{saving_time}')

# Hyperparameters
Hyperparameters = {
    "passes": 15,
    "iterations": 3,
    "num_topics": 20,
    "ETA": 'auto'
}
metrics = {}
# dataset = load_dataset("20-newsgroups", load_to_memory=False)
# dataset = create_tf_dataset(filepath)

# for post in dataset.take(1):
#     print(post)
#     tf.io.decode_json_example(post)
#     # parse_record(post, ['data', 'set'])['set']


# dataset = tf.data.Dataset.from_generator(
#     generate_record,
#     output_types=tf.string,
#     output_shapes=(None, ),
# )
#
# for document in dataset.take(1):
#     print(type(document))
# def parse_record(record, features):
#     """ parses one record of dataset to retrieve fields"""
#     return {feature: record.get('feature') for feature in features}
# below was done to compare time it takes, it took 11.4 seconds compared to 4.1 sec for the generator case above
# t_ = time.time()
# dataset = load_dataset("20-newsgroups", load_to_memory=True)
# dataset = [record.get('data').split() for record in dataset]
# MyDict = corpora.Dictionary(dataset)
# print(time.time()-t)

def main():
    try:
        lda = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
    except FileNotFoundError as error:
        print(error)
        print('Re-training the model...')
        filepath = 'C:\\Users\\Aly\\gensim-data\\20-newsgroups\\20-newsgroups.json'
        # topic, set, data, id
        # 11314 train and 18846 total records.
        t = time.time()
        train_corpus = MyCorpus(filepath)  # generates tokenized documents one at a time
        training_dict = corpora.Dictionary(train_corpus)
        train_corpus = MyCorpus(filepath, my_dictionary=training_dict)  # generate tokenized BOW documents one at a time
        test_corpus = MyCorpus(filepath, my_dictionary=training_dict, my_set='test')
        print(time.time()-t)
        # print("training samples, ", len(train_corpus))
        # print("testing samples, ", len(test_corpus))

        # model
        perplexity_logger = PerplexityMetric(corpus=train_corpus, logger='shell')
        coherence_logger = CoherenceMetric(corpus=train_corpus, dictionary=training_dict, coherence='u_mass')
        diversity_logger = DiversityMetric(num_topics=Hyperparameters['num_topics'])
        convergence_logger = ConvergenceMetric(logger='shell')
        # diff_logger = DiffMetric()
        metrics_list = [perplexity_logger, coherence_logger, diversity_logger, convergence_logger]  #, diff_logger]
        lda = models.LdaModel(
            train_corpus,
            id2word=training_dict,
            num_topics=Hyperparameters['num_topics'],
            passes=Hyperparameters['passes'],
            iterations=Hyperparameters['iterations'],
            eta=Hyperparameters['ETA'],
            callbacks= metrics_list
        )

    print("Hello")
    print(lda.metrics)
    print("done printing")
    plt.figure(figsize=(10, 10))
    length_ = len(lda.metrics)
    for i, (metric_name, metric_values) in enumerate(lda.metrics.items()):
        plt.subplot(length_ // 2, length_ - length_ // 2, i+1)
        plt.errorbar(x=np.arange(Hyperparameters['passes']),
                     y=metric_values
                     )
        # plt.title(f"{metric_name}")
        plt.xlabel('Pass number')
        plt.ylabel(f"{metric_name}")
    plt.show()

    # print the topics
    topics = []
    for i in range(Hyperparameters['num_topics']):
        topic = lda.show_topic(i, 12)
        topics.append([token for (token, probability) in topic])
        # print(topic)

    # Log the metrics
    # coherence
    cm = CoherenceModel(topics=topics, corpus=train_corpus, dictionary=training_dict, coherence='u_mass')
    metrics['coherence'] = cm.get_coherence()
    # diversity
    tokens = []  # To calculate diversity, obtain the most probable 25 tokens across all topics.
    for i in range(Hyperparameters['num_topics']):
        for item in lda.show_topic(i, topn=50):     # 50 is chosen heuristically to include most probable tokens.
            tokens.append(item)
    # print(tokens)
    sorted_tokens = sorted(tokens, key=lambda x: x[1], reverse=True)
    # print(sorted_tokens)
    metrics['diversity'] = diversity([token for (token, prob) in sorted_tokens][:25])
    # perplexity
    metrics['perplexity'] = lda.log_perplexity(list(train_corpus))

    # display data
    # display_data = pyLDAvis.gensim_models.prepare(lda, corpus=train_corpus, dictionary=training_dict)
    # with open(os.path.join(model_path, 'display_data'), 'wb') as file:
    #     pickle.dump(display_data, file)

    # saving to file system
    os.mkdir(model_path)
    lda.save(os.path.join(model_path, 'saved_model'))
    with open(os.path.join(model_path, 'hyperparameters.txt'), 'a') as f:
        json.dump(Hyperparameters, f)
    with open(os.path.join(model_path, 'metrics.txt'), 'a') as f:
        json.dump(metrics, f)

    print("HDP model")
    hdp = models.HdpModel(train_corpus, id2word=training_dict, callbacks=metrics_list)
    print(hdp.print_topics(num_topics=20, num_words=10))
    tfidf = models.TfidfModel(train_corpus)
    lsi = models.LsiModel(tfidf, id2word=training_dict, num_topics=20)
    # corpus_tfidf = tfidf[test_corpus]
    # for doc in corpus_tfidf:
    #     print(doc)


if __name__ == '__main__':
    main()
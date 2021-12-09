from gensim import corpora, models
from COTM.datasets import MyCorpus
import logging
import time
from datetime import datetime
import os
import json
from COTM.metrics import diversity, show_progress
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric, ConvergenceMetric, DiffMetric
from COTM.learning import DiversityMetric
from gensim.models.coherencemodel import CoherenceModel

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


def main():
    filepath = 'C:\\Users\\Aly\\gensim-data\\20-newsgroups\\20-newsgroups.json'
    # topic, set, data, id
    # 11314 train and 18846 total records.
    t = time.time()
    train_corpus = MyCorpus(filepath)  # generates tokenized documents one at a time
    training_dict = corpora.Dictionary(train_corpus)
    train_corpus = MyCorpus(filepath, my_dictionary=training_dict)  # generate tokenized BOW documents one at a time
    test_corpus = MyCorpus(filepath, my_dictionary=training_dict, my_set='test')
    print(time.time() - t)
    # print("training samples, ", len(train_corpus))
    # print("testing samples, ", len(test_corpus))

    try:
        lda = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # hdp = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # lsi = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # nmf = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # ctm = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # etm = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # ProdLda = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # NeuralLda = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # CATM = models.LdaModel.load(os.path.join(model_path, 'saved_model'))
        # COTM = models.LdaModel.load(os.path.join(model_path, 'saved_model'))

    except FileNotFoundError as error:
        print(error)
        print('Re-training the model...')
        # model
        perplexity_logger = PerplexityMetric(corpus=train_corpus, logger='shell')
        coherence_logger = CoherenceMetric(corpus=train_corpus, dictionary=training_dict, coherence='u_mass')
        diversity_logger = DiversityMetric(num_topics=Hyperparameters['num_topics'])
        convergence_logger = ConvergenceMetric(logger='shell')
        # diff_logger = DiffMetric()
        metrics_list = [perplexity_logger, coherence_logger, diversity_logger, convergence_logger]  # , diff_logger]
        lda = models.LdaModel(
            train_corpus,
            id2word=training_dict,
            num_topics=Hyperparameters['num_topics'],
            passes=Hyperparameters['passes'],
            iterations=Hyperparameters['iterations'],
            eta=Hyperparameters['ETA'],
            callbacks= metrics_list
        )

    show_progress(lda.metrics)
    print("HDP model")
    hdp = models.HdpModel(train_corpus, id2word=training_dict, callbacks=metrics_list)
    show_progress(hdp.metrics)
    # print(hdp.print_topics(num_topics=20, num_words=10))
    tfidf = models.TfidfModel(train_corpus)
    lsi = models.LsiModel(tfidf, id2word=training_dict, num_topics=20)
    show_progress(lsi.metrics)

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

    # saving to file system
    os.mkdir(model_path)
    lda.save(os.path.join(model_path, 'saved_model'))
    with open(os.path.join(model_path, 'hyperparameters.txt'), 'a') as f:
        json.dump(Hyperparameters, f)
    with open(os.path.join(model_path, 'metrics.txt'), 'a') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()
from gensim import corpora, models
from COTM.datasets import MyCorpus
import logging
import time
from datetime import datetime
import os
import json
from COTM.metrics import show_progress, assess_model
from gensim.models.callbacks import PerplexityMetric, CoherenceMetric, ConvergenceMetric, DiffMetric
from COTM.learning import DiversityMetric


# Setup logging and constants
logging.basicConfig(level=logging.INFO)
num_topics = 20
models_dir = "..\\saved_models"
saving_time = datetime.now().strftime('%Y-%m-%d_%H_%M')  # '%Y-%m-%d_%H_%M'
model_path = os.path.join(models_dir, f'lda_{num_topics}_{saving_time}')

# Hyperparameters
Hyperparameters = {
    "passes": 1,
    "iterations": 1,
    "num_topics": 20,
    "ETA": 'auto'
}


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
    metrics = assess_model(lda, corpus=train_corpus, dictionary=training_dict, n_topics=Hyperparameters['num_topics'])
    # print("HDP model")
    # hdp = models.HdpModel(train_corpus, id2word=training_dict)
    # # print(hdp.print_topics(num_topics=20, num_words=10))
    # tfidf = models.TfidfModel(train_corpus)
    # lsi = models.LsiModel(tfidf, id2word=training_dict, num_topics=20)

    # saving to file system
    os.mkdir(model_path)
    lda.save(os.path.join(model_path, 'saved_model'))
    with open(os.path.join(model_path, 'hyperparameters.txt'), 'a') as f:
        json.dump(Hyperparameters, f)
    with open(os.path.join(model_path, 'metrics.txt'), 'a') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    main()
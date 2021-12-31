from gensim.models.callbacks import PerplexityMetric, CoherenceMetric, ConvergenceMetric, DiffMetric
from COTM.learning import DiversityMetric
from gensim import models


def lda_fn(train_corpus, training_dict, params, model_dir, metrics):
    # model
    perplexity_logger = PerplexityMetric(corpus=train_corpus, logger='shell')
    coherence_logger = CoherenceMetric(corpus=train_corpus, dictionary=training_dict, coherence='u_mass')
    diversity_logger = DiversityMetric(num_topics=params['num_topics'])
    convergence_logger = ConvergenceMetric(logger='shell')
    # diff_logger = DiffMetric()
    metrics_list = [perplexity_logger, coherence_logger, diversity_logger, convergence_logger]  # , diff_logger]
    lda = models.LdaModel(
        train_corpus,
        id2word=training_dict,
        num_topics=params['num_topics'],
        passes=params['passes'],
        iterations=params['iterations'],
        eta=params['ETA'],
        callbacks=metrics_list
    )
    return lda


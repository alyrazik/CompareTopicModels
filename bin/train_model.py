""" Train and test one topic model"""
# By: Aly Abdelrazek

# Imports
import logging
import time
import os
import json
import plac

# Import helper functions
from gensim import corpora
from COTM.datasets import Dataset
from COTM.models import MODEL_FACTORY
from COTM.stopwords import STOPWORDS_FACTORY
from datetime import datetime
from COTM.metrics import show_progress, assess_model
import COTM.models

# Constants
num_topics = 20
STOPWORDS = STOPWORDS_FACTORY['gensim'] + STOPWORDS_FACTORY['aly']
# filepath = 'C:\\Users\\Aly\\gensim-data\\20-newsgroups\\20-newsgroups.json'
# model_dir = "..\\saved_models"

# Setup logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO
                    )
logger = logging.getLogger('Train Topic Model')


@plac.annotations(
    train_filepath=plac.Annotation(
        'Path to train data.',
        type=str
    ),
    eval_filepath=plac.Annotation(
        'Path to eval data.',
        type=str
    ),
    model_dir=plac.Annotation(
        'Path where the model is stored.',
        type=str
    ),
    model_name=plac.Annotation(
        'Model specification function. Pick one of the following: {}.'.format(
            list(MODEL_FACTORY.keys())
        ),
        type=str
    ),
    params_path=plac.Annotation(
        'Hyperparameters for the model. Pick according to the model',
        type=str
    ),
    stopwords_groups=plac.Annotation(
        'list of groups of stopwords as in stopwords.py file',
        type=str
    ),
    metrics=plac.Annotation(
        'list of metrics to log',
        type=str
    )
)
def main(
        train_filepath,
        eval_filepath,
        model_dir,
        model_name,
        params_path,
        stopwords_groups=None,
        metrics=None
):
    try:
        saving_time = datetime.now().strftime('%Y-%m-%d_%H_%M')  # '%Y-%m-%d_%H_%M'
        model_path = os.path.join(model_dir, f'lda_{num_topics}_{saving_time}')

        # topic, set, data, id
        # 11314 train and 18846 total records.
        if stopwords_groups is not None:
            stopwords = []
            for group in stopwords_groups:
                for word in STOPWORDS_FACTORY[group]:
                    stopwords.append(word)
        else:
            stopwords = STOPWORDS

        t = time.time()
        train_corpus = Dataset(train_filepath, stopwords=stopwords)  # generates tokenized documents one at a time
        training_dict = corpora.Dictionary(train_corpus)
        train_corpus = Dataset(train_filepath, dictionary=training_dict)  # generate tokenized BOW documents 1 @ a time
        test_corpus = Dataset(eval_filepath, dictionary=training_dict, set='test')
        print('Reading time is:', time.time() - t)
        params = {}
        with open(params_path, 'r') as f:
            params.update(json.load(f))
        if model_name not in MODEL_FACTORY:
            message = (
                '{} not present. Pick one of the following: {}'.format(
                    model_name,
                    list(MODEL_FACTORY.keys())
                )
            )
            raise RuntimeError(message)

        model = COTM.models.model_fn(MODEL_FACTORY.get(model_name),
                                     train_corpus,
                                     training_dict,
                                     params=params,
                                     model_dir=model_dir,
                                     metrics=metrics
                                     )
        fig = show_progress(model.metrics)
        metrics = assess_model(model, corpus=train_corpus, dictionary=training_dict, n_topics=params['num_topics'])
        metrics['perplexity'] = model.log_perplexity(list(train_corpus))

        # saving to file system
        os.mkdir(model_path)
        model.save(os.path.join(model_path, 'saved_model'))
        with open(os.path.join(model_path, 'hyperparameters.txt'), 'a') as f:
            json.dump(params, f)
        with open(os.path.join(model_path, 'metrics.txt'), 'a') as f:
            json.dump(metrics, f)
        fig.savefig(os.path.join(model_path, 'metrics_progression_with_passes.png'), bbox_inches='tight')
        with open(os.path.join(model_path, 'stopwords_list.txt'), 'a') as f:
            f.writelines([stopword + ', ' for stopword in stopwords])

    except RuntimeError:
        logger.exception('Exception occurred in running train_model.py')


if __name__ == '__main__':
    plac.call(main)

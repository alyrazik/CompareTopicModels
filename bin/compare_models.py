from gensim import corpora, models
from COTM.datasets import MyCorpus
import logging
import time
from datetime import datetime
import os
import json
from gensim.models.coherencemodel import CoherenceModel
logging.basicConfig(level=logging.INFO)
num_topics = 20
models_dir = "..\\saved_models"
saving_time = datetime.now().strftime('%Y-%m-%d_%H_%M')  # '%Y-%m-%d_%H_%M'
model_path = os.path.join(models_dir, f'lda_{num_topics}_{saving_time}')
# Hyperparameters

Hyperparameters = {
    "passes": 1,
    "iterations": 1,
    "num_topics": 20
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
        lda = models.LdaModel(
            train_corpus,
            id2word=training_dict,
            num_topics=Hyperparameters['num_topics'],
            passes=Hyperparameters['passes'],
            iterations=Hyperparameters['iterations']
        )

        topics = []
        for i in range(Hyperparameters['num_topics']):
            topic = lda.show_topic(i, 12)
            topics.append([token for (token, probability) in topic])
            print(topic)

        cm = CoherenceModel(topics=topics, corpus=train_corpus, dictionary=training_dict, coherence='u_mass')
        metrics['coherence'] = cm.get_coherence()
        os.mkdir(model_path)
        lda.save(os.path.join(model_path, 'saved_model'))
        with open(os.path.join(model_path, 'hyperparameters.txt'), 'a') as f:
            json.dump(Hyperparameters, f)
        with open(os.path.join(model_path, 'metrics'), 'a') as f:
            json.dump(metrics, f)

    # hdp = models.HdpModel(train_corpus, id2word=training_dict)
    # tfidf = models.TfidfModel(train_corpus)
    # lsi_model = models.LsiModel(tfidf, id2word=training_dict, num_topics=2)
    # corpus_tfidf = tfidf[test_corpus]
    # for doc in corpus_tfidf:
    #     print(doc)


if __name__ == '__main__':
    main()
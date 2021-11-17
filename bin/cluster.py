import pandas as pd
import json
import time
from COTM.cleaning import clean_all
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Constants
keywords = ['topic modeling', 'neural topic model', 'topic model', 'neural topic modelling']
N_TOPICS = 4  # max number of topics to use in your loop starting from 2 to N_topics-1
N_PASSES = 100
ITERATIONS = 400
CHUNKSIZE = 50
max_df = 0.3
min_df = 0.05
BIGRAMS_FREQUENCY = 10
TRIGRAM_FREQUENCY = 10
DataPath = 'C://Users//Aly//PycharmProjects//TopicModel//arXiv_metadata//arxiv-metadata-oai-snapshot.json'
DataPath = '//home//mohamed//PycharmProjects//TopicModel//arXiv_metadata//arxiv-metadata-oai-snapshot.json'

# Read file that contains metadata about papers (no paper text)


def get_metadata():
    """define a generator function since file is too big to handle in memory all at once"""
    with open(DataPath, 'r') as file:
        for line in file:
            yield line


def main():
    metadata = get_metadata()  # now metadata is an iterable

    titles = []
    authors = []
    abstracts = []
    update_dates = []

    print("Reading documents...")
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
    print('Reading time is ', t)

    df = pd.DataFrame(zip(titles, authors, abstracts, update_dates),
                      columns=['title', 'author', 'abstract', 'update_date'])

    # df.dropna(axis=0, subset=['title', 'author', 'abstract', 'update_date'], inplace=True)
    print("Number of documents: ", df.shape[0])

    ##############################################################################################################

    # clean
    print("Cleaning text...")
    t = time.time()
    df['abstract'] = df['abstract'].apply(clean_all)
    print("Cleaning time is ", time.time()-t)

    dictionary = gensim.corpora.Dictionary(df['abstract'])
    corpus = [
        gensim.models.doc2vec.TaggedDocument(dictionary.doc2bow(doc), [i]) for i, doc in enumerate(df['abstract'])
    ]
    # print(dictionary)
    # print(dictionary.token2id)

    ##############################################################################################################

    # model
    model = gensim.models.doc2vec.Doc2Vec(vector_size=150, min_count=2, epochs=40)
    model.build_vocab(corpus)
    # print(f"Word 'neural' appeared {model.wv.get_vecattr('neural', 'count')} times in the training corpus.")
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    vectors = df['abstract'].apply(
            lambda x: model.infer_vector(x)
            ).tolist()
    print(type(vectors))
    print(vectors)

    #############################################################################################################

    # Cluster
    K_value = 4
    kmeans_model = KMeans(n_clusters=K_value, init='k-means++', n_init=2000, max_iter=6000)
    # X = kmeans_model.fit(vectors)
    labels = kmeans_model.labels_.tolist()
    clusters = kmeans_model.fit_predict(vectors)
    centers = kmeans_model.cluster_centers_
    print('centers', type(centers))
    print(centers)

    print('labels', type(labels))
    print(labels)
    print('clusters', type(clusters))
    print(clusters)
    # # PCA
    # l = kmeans_model.fit_predict(model.docvecs.vectors_docs)
    # pca = PCA(n_components=2).fit(model.docvecs.vectors_docs)
    # datapoint = pca.transform(model.docvecs.vectors_docs)
    #
    # # GRAPH
    # # """**Plot the clustering result**"""
    #
    # plt.figure
    # label1 = ["#FFFF00", "#008000", "#0000FF", "#800080"]
    # color = [label1[i] for i in labels]
    # plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

    # centroids = kmeans_model.cluster_centers_
    # centroidpoint = pca.transform(centroids)
    # plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
    # plt.show()

    ##############################################################################################################

    # inspect clusters (visualize).


if __name__ == "__main__":
    main()
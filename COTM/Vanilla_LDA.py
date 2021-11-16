import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from gensim import matutils
import scipy.sparse

def data_to_lda(data,
                text_column, vectorizer, lda_obj,
                keep_non_text=True):
    """Full transformation from dataframe with text column to a
    decomposed (LDirA) data frame
    Params:
    dataframe - (Pandas DataFrame obj) dataframe containing text and other data
    text_column - (str) name of column containing text data
    lda_obj - (obj) pre-instantiated sklearn LatentDirichletAllocation object
    keep_non_text - (bool) keep the additional columns from the original dataframe if True and append them.

    Returns:
    lsa_df - document-topic table (with the option of having the previous non-doc-term columns appended on the left)
    lsa_obj - the fitted LatentDirichletAllocation object
    """
    num_non_dt_cols = len(data.columns)
    doc_term = df_to_dt(data, text_column, vectorizer, keep_non_text=keep_non_text)
    if keep_non_text == False:
        num_non_dt_cols = 0
    lda_data, lda_obj = dt_to_lda(doc_term, num_non_dt_cols, lda_obj, keep_non_text=keep_non_text)
    return lda_data, lda_obj


def dt_to_lda(dataframe, num_non_dt_cols: int, lda_obj, keep_non_text=True, ):
    """Takes document term matrix and returns (dataframe with LDirA topic data,
    LDirA sklearn object). Specify number of non-DocTerm columns, fn will assume
    all the Doc-Term columns are to the left of that.
    Params:
    dataframe - (Pandas DataFrame obj) dataframe containing text and other data
    num_non_dt_cols - (int) this should essentially be the total number of columns in
    your original dataframe, pre-vectorisation. The function will split off the document-term matrix through
    an iloc performed on the basis of the number provided.
    lda_obj - (obj) pre-instantiated sklearn LatentDirichletAllocation object
    keep_non_text - (bool) keep the additional columns from the original dataframe if True and append them.

    Returns:
    new_df / vect_train_df - dataframe with/out non-text columns on the left and document term matrix on the right"""
    data = dataframe

    dt_mat = data.iloc[:, num_non_dt_cols:]
    lda_df = pd.DataFrame(lda_obj.fit_transform(dt_mat), index=data.index,
                          columns=list(range(1, (lda_obj.n_components + 1))))
    lda_df = lda_df.add_prefix('topic_')

    if keep_non_text:
        df = data.iloc[:, :num_non_dt_cols]
        df_out = pd.concat([df, lda_df], join='inner', axis=1)
        return df_out, lda_obj

    else:
        return lda_df, lda_obj


def df_to_dt(dataframe, text_column: str, vectoriser_obj, keep_non_text=True):
    """Takes in dataframe, specifying which columns is the one to vectorise
    + a vectorisation object (of type sklearn.feature_extraction/CountVectoriser
    or TFiDF vectoriser) and returns a new dataframe with the original columns
    and new Document-Term matrix appended to the right
    Params:
    dataframe - (Pandas DataFrame obj) dataframe containing text and other data
    text_column - (str) name of column containing text data
    vectoriser_obj - (obj) pre-instantiated vectorisation object (of type sklearn.feature_extraction/CountVectoriser
    or TFiDF vectoriser)
    keep_non_text - (bool) keep the additional columns from the original dataframe if True and append them.

    Returns:
    new_df / vect_train_df - dataframe with/out non-text columns on the left and document term matrix on the right
    """
    data = dataframe

    text_series = data[text_column]

    vect_train_sparse = vectoriser_obj.fit_transform(text_series.values.astype('U'))
    vect_train_df = pd.DataFrame(vect_train_sparse.toarray(),
                                 columns=vectoriser_obj.get_feature_names(), index=data.index)

    if keep_non_text:
        assert len(data) == len(
            vect_train_df), "Size mismatch: text and non-text dfs have len {} and {} respectively".format(
            len(vect_train_df), len(data))
        new_df = pd.concat([data, vect_train_df], axis=1, join='outer')
        return new_df

    else:
        return vect_train_df


def print_topics(model, vectorizer, n_top_words, topics_to_include = None):
    """Takes in the sklearn decomposition model (LSA/LDA), our DocTerm
    vectorizer (Count/TFiDF), nr of words we'd like to see and num of topics
    to visualise.
    Prints out the top n topics/latent concepts plus their most
    strongly associated terms"""
    words = vectorizer.get_feature_names()
    if topics_to_include==None:
        for topic_idx, topic in enumerate(model.components_):
            print(f"\nTopic #{topic_idx+1}:")
            print("; ".join([words[i]
                            for i in topic.argsort()[:-n_top_words - 1:-1]]))
    else:
        for topic_idx, topic in enumerate(model.components_):
            if topic_idx in topics_to_include:
                print(f"\nTopic ##{topic_idx+1}")
                print("; ".join([words[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]]))


def model_from_text(col, max_df=0.25, min_df=10):
    """
    :param
    col a series of text. Typically a column in pandas dataframe.
    :return:
    corpus is the term-document matrix in the gensim format
    id2word is dictionary of the all terms and their respective location in the term-document matrix
    """
    cv = CountVectorizer(max_df=max_df, min_df=min_df)  # for topic modeling.
    # Create document term matrix
    data_cv = cv.fit_transform(col)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.reset_index()
    tdm = data_dtm.transpose()
    sparse_counts = scipy.sparse.csr_matrix(tdm)
    corpus = matutils.Sparse2Corpus(sparse_counts)
    id2word = dict((v, k) for k, v in cv.vocabulary_.items())  # index to word
    # cvectorizer.vocabulary_ is a dictionary with tokens as keys and values represent their index.
    return corpus, id2word
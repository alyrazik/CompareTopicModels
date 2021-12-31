import tensorflow as tf
import logging
import gensim.downloader as api
import json
from COTM.cleaning import clean_tokenize
from COTM.stopwords import STOPWORDS_FACTORY
logger = logging.getLogger('datasets')


def create_tf_dataset(path):
    dataset = tf.data.TextLineDataset(path)
    return dataset


def load_dataset(name, load_to_memory=True):
    """
    downloads datasets from gensim data repo https://github.com/RaRe-Technologies/gensim-data
    and either saves it to filesystem or loads it to memory.

    Args:
       - name: name of the dataset to download. For example, "20-newsgroups"
       - load_to_memory: if True, saves the dataset to filesystem on ~./gensim-data/20-newsgroups/20-newsgroups.gz
       otherwise, it loads the data to memory
    Returns:
       - the dataset or none (if data is saved to disk)
    Throws:
       - Nothing. Catches ValueError from api.load in case the dataset name is invalid.
    """
    try:
        if load_to_memory is False:
            location = api.load(name, return_path=True)
            print(f" The requested dataset is saved to {location}")
            return location
        else:
            data = api.load(name, return_path=False)
            logger.info(f" The requested dataset is loaded to memory")
            return data
        # return data
    except ValueError:
        print("Dataset not found. Please choose a dataset from ",
              "https://github.com/RaRe-Technologies/gensim-data")


class Dataset:
    def __iter__(self):
        with open(self.filepath, 'r') as f:
            for line in f:
                d = json.loads(line)
                if d.get('set') == self.set:
                    if self.my_dictionary is not None:
                        yield self.my_dictionary.doc2bow(clean_tokenize(d.get('data'), stopwords=self.stopwords))
                    else:
                        yield clean_tokenize(d.get('data'), stopwords=self.stopwords)
                    # yield self.obtain_document_text(d)

    def __init__(self, filepath, dictionary=None, set='train',
                 stopwords=STOPWORDS_FACTORY['gensim'] + STOPWORDS_FACTORY['aly']):
        self.filepath = filepath
        self.my_dictionary = dictionary
        self.set = set
        self.stopwords = stopwords
        self.length = 0

    def __len__(self):
        with open(self.filepath, 'r') as f:
            for line in f:
                d = json.loads(line)
                if d.get('set') == self.set:
                    self.length += 1
        return self.length

    def obtain_document_text(record):
        return record.get('data').split()

# def parse_record(record, feature_names):
#
#
#     # """
#     # Process a record to create input tensors and labels.
#     #
#     # Process a record with DATASETS_METADATA structure.
#     # Args:
#     #     - record: a record (<tf.TFRecord>).
#     #     - feature_names: list of features to yield (<list<string>>).
#     # Returns:
#     #     A dict of tf.Tensor where the keys are
#     #     feature_names
#     # """
#     # keys_to_features = {
#     #     feature_name: DATASETS_METADATA[feature_name]
#     #     for feature_name in feature_names
#     # }
#     # keys_to_features['ic50'] = DATASETS_METADATA['ic50']
#     # features = tf.parse_single_example(record, keys_to_features)
#     # features = tf.parse_single_example(re)
#     # label = features.pop('ic50')
#     # return features, label
#     record = json.loads(record.map(lambda s: tf.py_function()))
#     output = {k: record[k] for k in feature_names}
#     return output

#
# def generate_dataset_from_file(
#         filepath, buffer_size=int(256e+6), num_parallel_reads=None
# ):
#     """
#     Generate a tf.Dataset given a path.
#
#     Args:
#         - filepath: path to a file or a folder containing data (<string>).
#         - buffer_size: size of the buffer in bytes (<int>). Defaults to 256MB.
#     Returns:
#         A tf.Dataset iterator over file/s in .tfrecords format.
#     """
#     if os.path.isdir(filepath):
#         filenames = get_sorted_filelist(filepath)
#     else:
#         filenames = [filepath]
#
#     logger.debug(
#         'Parsing examples from the following files: {}'.format(filenames)
#     )
#
#     return tf.data.TFRecordDataset(
#         filenames,
#         buffer_size=buffer_size,
#         num_parallel_reads=num_parallel_reads
#     )


# def get_sorted_filelist(filepath, pattern='tfrecords'):
#     """
#     Gets a sorted list of all files with suffix pattern in a given filepath.
#
#     Args:
#         - filepath: path to folder to retrieve files (<string>).
#         - pattern: suffix specifying file types to retrieve (<string>).
#     Returns:
#         A list of sorted strings to all matching files in filepath.
#
#     """
#     return sorted([
#         filename for filename in
#         map(lambda entry: os.path.join(filepath, entry), os.listdir(filepath))
#         if os.path.isfile(filename) and pattern in filename
#     ])
#
#

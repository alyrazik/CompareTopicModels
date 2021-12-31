# Compare Topic Models

A package that gives a single interface to running and evaluation of several topic models on your data.
You can run it from the command line or use it in your python scripts. 


## Usage:

Please clone the repo to your local host and run the script 'bin/train_model.py' from the command line. 
During the training, the script displays the perplexity score every epoch of training. After the training
is complete, charts of the evolution of metrics with passes are shown.

The script saves its artefacts in a folder within the specified mode_path (see below argument).
The artefacts are:

* hyperparameters.txt: those used to train the model.
* metics.txt: the final values of the model metrics, such as, perplexity score and coherence score.
* metrics_progression_with_passes.png: the charts that were displayed after the training finishes.
* saved_model: This is the saved_model which can be loaded using gensim library.
* Other files that are saved along with the model by gensim. 

you need to specify the following arguments. Note that currently only lda model is available. 
also optional arguments defaults to None and are not used. 

```
usage: 

positional arguments:
  train_filepath        Path to train data. 
  eval_filepath         Path to eval data.
  model_path            Path where the model and training artefacts are stored. 
  model_name            Topic model name. Pick one of the
                        following: ['lda', 'lsi', 'nmf', 'hdp', 'ctm', 'etm', 'ProdLda', 'NeuralLda', 'BERTopic', 'CATM']
  params_path           Path to model params. Dictionary with parameters for the selected topic model . 
                        See hyperparameters folder for examples. 

optional arguments:
  stopwords_groups      list of groups of stopwords as in stopwords.py file.
  metrics               list of metrics to log
  ```
  

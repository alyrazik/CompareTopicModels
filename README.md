# Compare Topic Models

A package that gives a single interface to running and evaluation of several topic models.
You can run it from the command line or use it in your python scripts. 


## Usage:

Please clone the repo to your local host and run the script 'train_model.py' from the command line. 
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
  

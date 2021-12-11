def model_fn(
        model_specification_fn,
        corpus,
        dictionary,
        params,
        model_dir,
        metrics
):
    return model_specification_fn(corpus, dictionary, params, model_dir, metrics)

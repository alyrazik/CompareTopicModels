import matplotlib.pyplot as plt
import numpy as np


def diversity(tokens):
    """It defines topic diversity as the percentage of the unique words in the top 25 words
of all topics
Args:
    tokens: list of tokens
returns:
    diversity score as a percentage of unique words in top 25 tokens.
    """
    return len(set(tokens)) / len(tokens)


def show_progress(metrics):
    """ charts the metrics that were logged during model training.
    Args:
        metrics: dictionary with keys of metric names and values are lists of measured values.
    returns:
        displays the figure with axes to display trends.
        """

    plt.figure(figsize=(10, 10))
    length_ = len(metrics)
    for i, (metric_name, metric_values) in enumerate(metrics.items()):
        plt.subplot(length_ // 2, length_ - length_ // 2, i+1)
        plt.errorbar(x=np.arange(len(metric_values)),
                     y=metric_values
                     )
        # plt.title(f"{metric_name}")
        plt.xlabel('Pass number')
        plt.ylabel(f"{metric_name}")
    plt.show()

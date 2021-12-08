def diversity(tokens):
    """It defines topic diversity as the percentage of the unique words in the top 25 words
of all topics
Args:
    tokens: list of tokens
returns:
    diversity score as a percentage of unique words in top 25 tokens.
    """
    return len(set(tokens)) / len(tokens)

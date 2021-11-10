Coherence Optimized Topic Model with User Entry

this is an interpretable topic model that has three
features in it. First, it optionally takes a user-
defined topic to regularize the model so as to generate
other topics from the corpus using the user input.
second it optimizes for coherence scores to generate
interpretable topics while maintaining the modeling
quality.
third, it represents topics, sentences, and words in
the same embedding space. the idea here is that humans
use phrases (sentences) to carry over a single semantic
meaning. Some words have a semantic meaning too but not 
all of them. So here, we regularize the model to 
associate semantics to words if possible, or else, to
associate the semantics to sentences (phrases). 

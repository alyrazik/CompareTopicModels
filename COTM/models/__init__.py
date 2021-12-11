"""Initialization for `paccmann.model.model_specifications` submodule."""
from .lda import lda_fn
from .lsi import lsi_fn
from .nmf import nmf_fn
from .hdp import hdp_fn
from .ctm import ctm_fn
from .etm import etm_fn
from .ProdLda import prod_lda_fn
from .NeuralLda import neural_lda_fn
from .BERTopic import BERTopic_fn
from .CATM import catm_fn


MODEL_FACTORY = {
    'lda': lda_fn,
    'lsi': lsi_fn,
    'nmf': nmf_fn,
    'hdp': hdp_fn,
    'ctm': ctm_fn,
    'etm': etm_fn,
    'ProdLda': prod_lda_fn,
    'NeuralLda': neural_lda_fn,
    'BERTopic': BERTopic_fn,
    'CATM': catm_fn
}


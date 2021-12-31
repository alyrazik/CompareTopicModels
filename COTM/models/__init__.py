"""Initialization for `paccmann.model.model_specifications` submodule."""
from COTM.models.model_specs.lda import lda_fn
from COTM.models.model_specs.lsi import lsi_fn
from COTM.models.model_specs.nmf import nmf_fn
from COTM.models.model_specs.hdp import hdp_fn
from COTM.models.model_specs.ctm import ctm_fn
from COTM.models.model_specs.etm import etm_fn
from COTM.models.model_specs.ProdLda import prod_lda_fn
from COTM.models.model_specs.NeuralLda import neural_lda_fn
from COTM.models.model_specs.BERTopic import BERTopic_fn
from COTM.models.model_specs.CATM import catm_fn
from COTM.models.core import model_fn


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


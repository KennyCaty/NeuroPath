
import os
from  sentence_transformers import SentenceTransformer
def init_embedding_model(model_name):
    if 'GritLM/' in model_name:
        from src.lm_wrapper.gritlm import GritWrapper
        return GritWrapper(model_name)
    elif model_name != 'bm25':
        return SentenceTransformer(model_name)

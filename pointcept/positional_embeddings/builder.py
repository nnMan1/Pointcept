"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

POSITIONAL_EMBEDDINGS = Registry("positional_embeddings")


def build_positional_embedding(cfg):
    """Build models."""
    return POSITIONAL_EMBEDDINGS.build(cfg)

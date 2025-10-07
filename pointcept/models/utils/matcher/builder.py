"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

MATCHERS = Registry("matchers")

def build_matcher(cfg):
    """Build models."""
    return MATCHERS.build(cfg)

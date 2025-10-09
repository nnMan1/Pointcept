"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

MATCHERS = Registry("matchers")
COSTS = Registry("costs")

def build_matcher(cfg):
    """Build matchers."""
    return MATCHERS.build(cfg)

def build_cost(cfg):
    """Build costs."""
    return COSTS.build(cfg)

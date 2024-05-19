"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
import sys
sys.path.append('/home')

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    tester.test()

class Args:
    def __init__(self):
        self.config_file = 'configs/abc_dataset/insseg-mask3d-v1m1-0-spunet-base.py'
        self.num_gpus = 1
        self.num_machines = 1
        self.machine_rank = 0 
        self.dist_url = 'auto'
        self.options={'save_path': 'exp/abc_dataset_hungarian_matcher/insseg-mask3d-v1m1-0-spunet-base_delete3'}

def main():
    args = Args()
    # args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()

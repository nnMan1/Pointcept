"""
Main Training Script

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
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

class Args:
    def __init__(self):
        self.config_file = 'configs/abc_dataset/insseg-mask3d-v1m1-0-spunet-base.py'
        self.num_gpus = 1
        self.num_machines = 1
        self.machine_rank = 0 
        self.dist_url = 'auto'
        self.options={'save_path': 'exp/delete_imed/insseg-mask3d-v1m1-0-spunet-base_delete4'}

def main():

    args = default_argument_parser().parse_args()
    # args = Args()
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

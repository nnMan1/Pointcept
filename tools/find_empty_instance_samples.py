"""List samples that have NO valid (non-ignore) instances.

Such a sample makes SuperpointPooling skip it and desync
instance_segment_offset from offset -> IndexError in the loss (only when
use_superpoint_pooling=True). This runs each sample through the deterministic
geometry transforms (CenterShift/Copy/GridSample/InstanceParser -- the same
instance labels the model sees) and flags any where (instance != -1).sum() == 0.

Writes <split>_empty_instances.txt per split. Run in the container:

  sh scripts/find_empty_instances.sh configs/my_real/<config>.py
"""

import argparse
import inspect
import os

import numpy as np

from pointcept.utils.config import Config
from pointcept.datasets import build_dataset
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import Compose

IGNORE = -1
# deterministic, instance-affecting transforms (no augmentation, no feature load)
KEEP = ("CenterShift", "Copy", "GridSample", "InstanceParser", "NormalizeCoord")


def build_ds(data_cfg):
    cfg = dict(data_cfg)
    cfg.pop("transform", None)
    cls = DATASETS.get(cfg["type"])
    accepted = set(inspect.signature(cls.__init__).parameters)
    for k, v in {"test_mode": False, "cache": True,
                 "load_images": True, "load_image_files": True}.items():
        if k in accepted:
            cfg[k] = v
        else:
            cfg.pop(k, None)
    return build_dataset(cfg)


def build_pipeline(data_cfg):
    cfgs = [t for t in (data_cfg.get("transform", []) or []) if t.get("type") in KEEP]
    return Compose(cfgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--out", default="empty_instance_report")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    for tag in ("train", "val", "test"):
        if tag not in cfg.data:
            continue
        ds = build_ds(cfg.data[tag])
        pl = build_pipeline(cfg.data[tag])
        total = len(ds.data_list) if hasattr(ds, "data_list") else len(ds)
        bad = []
        print(f"\n[{tag}] scanning {total} samples ...")
        for i in range(total):
            path = ds.data_list[i] if hasattr(ds, "data_list") and i < len(ds.data_list) \
                else ds.get_data_name(i)
            try:
                d = pl(ds.get_data(i))
                inst = np.asarray(d["instance"]).reshape(-1)
                n_valid = int((inst != IGNORE).sum())
                if n_valid == 0:
                    bad.append(path)
                    print(f"  [{tag}] EMPTY-INSTANCE: {path}")
            except Exception as e:
                print(f"  [{tag}] idx {i} ({path}) FAILED: {type(e).__name__}: {e}")
        with open(os.path.join(out_dir, f"{tag}_empty_instances.txt"), "w") as f:
            f.write("\n".join(bad) + ("\n" if bad else ""))
        print(f"[{tag}] {len(bad)}/{total} samples with NO valid instances "
              f"-> {tag}_empty_instances.txt")

    print(f"\nDone. Reports under: {out_dir}")


if __name__ == "__main__":
    main()

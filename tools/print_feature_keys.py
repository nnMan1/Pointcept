"""Print the sample names each split (train/val) looks up, the feature-cache
keys, and which names are NOT covered by the cache.

Uses the same exact/suffix matching as LoadImageFeatures, so a "MISSING" here is
exactly what would raise KeyError at train time. Run inside the container (so
/home resolves, matching the names the trainer builds):

  sh scripts/print_feature_keys.sh configs/my_real/<config>.py
"""

import argparse
import inspect
import os
import pickle

from pointcept.utils.config import Config
from pointcept.datasets import build_dataset
from pointcept.datasets.builder import DATASETS


def build_ds(data_cfg):
    cfg = dict(data_cfg)
    cfg.pop("transform", None)
    cls = DATASETS.get(cfg["type"])
    if "test_mode" in set(inspect.signature(cls.__init__).parameters):
        cfg["test_mode"] = False
    return build_dataset(cfg)


def feat_root(data_cfg):
    for t in (data_cfg.get("transform", []) or []):
        if t.get("type") == "LoadImageFeatures" and t.get("features_root"):
            return t["features_root"]
    return None


def load_keys(root):
    idx = pickle.load(open(os.path.join(root, "feature_index.pkl"), "rb"))
    return set(e[-1] if isinstance(e, (list, tuple)) else e for e in idx)


def resolve(name, keys):
    if name in keys:
        return "exact"
    cands = [k for k in keys if k.endswith(name) or name.endswith(k)]
    return "suffix" if len(cands) == 1 else ("ambiguous" if cands else "MISSING")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("-o", "--out", default="feature_key_report",
                    help="output dir for the .txt files")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(args.out, name)
    os.makedirs(out_dir, exist_ok=True)

    roots = {}  # features_root -> keys (loaded once)

    for tag in ("train", "val"):
        if tag not in cfg.data:
            continue
        ds = build_ds(cfg.data[tag])
        names = [ds.get_data_name(i) for i in range(len(ds.data_list))]
        root = feat_root(cfg.data[tag])
        keys = roots.get(root)
        if root and keys is None:
            keys = roots[root] = load_keys(root)

        # <tag>_names.txt : "<status>\t<name>" for every sample
        # <tag>_missing.txt : just the names NOT resolvable in the cache
        rows, missing = [], []
        for n in names:
            st = resolve(n, keys) if keys else "no-cache"
            rows.append(f"{st}\t{n}")
            if st not in ("exact", "suffix"):
                missing.append(n)
        with open(os.path.join(out_dir, f"{tag}_names.txt"), "w") as f:
            f.write(f"# {tag}: {len(names)} samples | features_root={root}\n")
            f.write("\n".join(rows) + "\n")
        with open(os.path.join(out_dir, f"{tag}_missing.txt"), "w") as f:
            f.write("\n".join(missing) + ("\n" if missing else ""))
        print(f"[{tag}] {len(names)} samples, {len(missing)} MISSING "
              f"-> {tag}_names.txt / {tag}_missing.txt")

    for root, keys in roots.items():
        fn = "cache_keys.txt"
        with open(os.path.join(out_dir, fn), "w") as f:
            f.write(f"# {root} ({len(keys)} keys)\n")
            f.write("\n".join(sorted(keys)) + "\n")
        print(f"[cache] {len(keys)} keys from {root} -> {fn}")

    print(f"\nWrote files under: {out_dir}")


if __name__ == "__main__":
    main()

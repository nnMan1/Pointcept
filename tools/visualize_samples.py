"""Visualize raw dataset samples (images + point cloud + superpoints + features).

For a given Pointcept config, dumps, per sample, from the train and test splits:
  * a PNG with: a grid of the rendered multi-view images, the input point cloud
    (coord) colored by instance, the superpoint over-segmentation (seg_indices)
    if available, and the DinoV3 point features (PCA->RGB) if a feature cache is
    configured;
  * a binary PLY with the FULL point cloud (xyz + per-point scalar fields:
    instance, segment, superpoint, and the 3 DinoV3-feature PCA channels) for
    deeper offline analysis in CloudCompare / MeshLab.

It calls dataset.get_data(idx) directly, so NO augmentation/transforms are
applied -- you see the raw data the pipeline starts from. Point features are the
cached DinoV3 grids scattered onto points via the dataset mappings (the same
projection the model uses), reduced to 3D by PCA.

Run inside the training container, e.g.:

  sh scripts/run_visualize.sh configs/abc_dataset/<config>.py 10
"""

import argparse
import inspect
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pointcept.utils.config import Config
from pointcept.datasets import build_dataset
from pointcept.datasets.builder import DATASETS
from pointcept.datasets.transform import LoadImageFeatures, Compose

MAX_IMAGES = 12           # cap rendered views per figure (images only)
MAX_MAP_ROWS = 400000     # cap pixel->point matches used for feature scatter
PATCH = 16                # DinoV3 patch size (pixel coord -> feature-grid index)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_numpy(x):
    return np.asarray(x)


def _color_by_id(ids):
    """Map arbitrary integer ids -> RGBA float colors; -1 (ignore) -> gray."""
    ids = np.asarray(ids).reshape(-1)
    uniq = np.unique(ids)
    lut = {v: i for i, v in enumerate(uniq)}
    cmap = plt.get_cmap("tab20")
    colors = np.zeros((len(ids), 4))
    for v in uniq:
        mask = ids == v
        colors[mask] = (0.6, 0.6, 0.6, 1.0) if v < 0 else cmap(lut[v] % 20)
    return colors


# --------------------------------------------------------------------------- #
# point features: scatter cached DinoV3 grids onto points, PCA -> RGB
# --------------------------------------------------------------------------- #
def has_valid_features(data):
    """Check the dict actually carries usable DinoV3 features + mappings.

    Returns (ok: bool, reason: str). Used to skip feature viz cleanly.
    """
    if "image_features" not in data or data["image_features"] is None:
        return False, "no 'image_features' key"
    feats = _to_numpy(data["image_features"])
    if feats.ndim != 4 or feats.size == 0:
        return False, f"unexpected image_features shape {tuple(feats.shape)}"
    if "mappings_src" not in data or "mappings_tgt" not in data:
        return False, "no pixel->point mappings"
    if len(_to_numpy(data["mappings_tgt"]).reshape(-1)) == 0:
        return False, "empty mappings"
    return True, "ok"


def compute_point_features(data):
    """Return (proj Nx3, valid Nx bool) DinoV3 features per point, or None.

    Mirrors Image2PointCLoud: per pixel->point match (view, row, col), index the
    feature grid at (view, row//PATCH, col//PATCH) and mean-pool onto the point.
    PCA is fit on the mapped features then applied, so we never materialize an
    N x 1280 array.
    """
    if "image_features" not in data or "mappings_src" not in data:
        return None
    feats = _to_numpy(data["image_features"]).astype(np.float32)  # (V,h,w,C)
    if feats.ndim != 4:
        return None
    V, h, w, C = feats.shape
    src = _to_numpy(data["mappings_src"]).astype(np.int64)        # (M,3)
    tgt = _to_numpy(data["mappings_tgt"]).reshape(-1).astype(np.int64)
    # After GridSample the cloud is voxelized; 'inverse' maps each original
    # point to its grid point. mappings_tgt indexes original points, so remap
    # to grid indices -- exactly how the model scatters image features.
    if "inverse" in data and data["inverse"] is not None:
        tgt = _to_numpy(data["inverse"]).reshape(-1).astype(np.int64)[tgt]
    n_pts = len(_to_numpy(data["coord"]))

    m = len(tgt)
    if m > MAX_MAP_ROWS:
        sel = np.random.default_rng(0).choice(m, MAX_MAP_ROWS, replace=False)
        src, tgt = src[sel], tgt[sel]

    a = np.clip(src[:, 0], 0, V - 1)
    b = np.clip(src[:, 1] // PATCH, 0, h - 1)
    c = np.clip(src[:, 2] // PATCH, 0, w - 1)
    mapped = feats[a, b, c]                                       # (M,C)

    # PCA (numpy SVD) fit on a subset, then project all mapped vectors to 3D
    fit = mapped
    if len(fit) > 20000:
        fit = fit[np.random.default_rng(1).choice(len(fit), 20000, replace=False)]
    mean = fit.mean(0)
    _, _, Vt = np.linalg.svd(fit - mean, full_matrices=False)
    comps = Vt[:3]                                                # (3,C)
    proj_map = (mapped - mean) @ comps.T                          # (M,3)

    point_proj = np.zeros((n_pts, 3), np.float32)
    cnt = np.zeros(n_pts, np.float32)
    np.add.at(point_proj, tgt, proj_map)
    np.add.at(cnt, tgt, 1.0)
    valid = cnt > 0
    if not valid.any():
        return None  # no point received any feature
    point_proj[valid] /= cnt[valid, None]
    return point_proj, valid


def features_to_rgb(proj, valid):
    """Per-channel percentile-normalized PCA -> uint8 RGB; invalid -> gray."""
    rgb = np.full((len(proj), 3), 153, np.uint8)  # gray for unmapped points
    if valid.any():
        v = proj[valid]
        out = np.zeros_like(v)
        for k in range(3):
            lo, hi = np.percentile(v[:, k], 2), np.percentile(v[:, k], 98)
            out[:, k] = np.clip((v[:, k] - lo) / (hi - lo + 1e-9), 0, 1)
        rgb[valid] = (out * 255).astype(np.uint8)
    return rgb


# --------------------------------------------------------------------------- #
# PLY export (binary little-endian, with scalar fields)
# --------------------------------------------------------------------------- #
def write_ply(path, coord, rgb, scalars):
    n = len(coord)
    dtype = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    for name in scalars:
        dtype.append((name, "<f4"))
    arr = np.empty(n, dtype=dtype)
    coord = np.asarray(coord, np.float32)
    arr["x"], arr["y"], arr["z"] = coord[:, 0], coord[:, 1], coord[:, 2]
    rgb = np.asarray(rgb, np.uint8)
    arr["red"], arr["green"], arr["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    for name, vals in scalars.items():
        arr[name] = np.asarray(vals, np.float32)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue"]
    header += [f"property float {name}" for name in scalars]
    header.append("end_header\n")
    with open(path, "wb") as f:
        f.write(("\n".join(header)).encode())
        arr.tofile(f)


# --------------------------------------------------------------------------- #
# per-sample rendering
# --------------------------------------------------------------------------- #
def visualize_sample(data, png_path, ply_path, feat_loader=None, pipeline=None):
    # Capture raw images before any transform consumes/drops them (the config's
    # LoadImageFeatures drops 'images'); images aren't geometrically transformed.
    images = list(data.get("images", []) or [])

    # raw mode: attach features via a standalone loader (keeps images).
    if feat_loader is not None:
        try:
            feat_loader(data)
        except Exception as e:
            print(f"    [feat] loader failed: {type(e).__name__}: {e}")

    # model mode: run the config's transform pipeline so coord/features are in
    # the exact space the model is trained on (CenterShift, GridSample, ...).
    if pipeline is not None:
        try:
            data = pipeline(data)
        except Exception as e:
            print(f"    [pipeline] failed: {type(e).__name__}: {e}")

    coord = _to_numpy(data["coord"]).astype(np.float32)
    n_pts = len(coord)
    instance = _to_numpy(data["instance"]).reshape(-1) if "instance" in data else np.zeros(n_pts)
    segment = _to_numpy(data["segment"]).reshape(-1) if "segment" in data else np.full(n_pts, -1)
    has_sp = "seg_indices" in data and data["seg_indices"] is not None \
        and len(_to_numpy(data["seg_indices"])) == n_pts
    superpoint = _to_numpy(data["seg_indices"]).reshape(-1) if has_sp else None

    # point features: validate the dict actually carries them, then scatter
    # (uses 'inverse' to land on grid points when the pipeline ran).
    proj = valid = None
    ok, reason = has_valid_features(data)
    if not ok:
        print(f"    [feat] no usable features ({reason}); skipping feature viz")
    else:
        try:
            res = compute_point_features(data)
        except Exception as e:
            res = None
            print(f"    [feat] computation failed: {type(e).__name__}: {e}")
        if res is None:
            print("    [feat] no points received features; skipping feature viz")
        else:
            proj, valid = res

    # ---------------- PLY: full cloud, labels as scalar fields ----------------
    # RGB = instance color (quick view); scalar fields for CloudCompare analysis.
    scalars = {"instance": instance.astype(np.float32),
               "segment": segment.astype(np.float32)}
    if has_sp:
        scalars["superpoint"] = superpoint.astype(np.float32)
    if proj is not None:
        scalars["feat_pca0"] = proj[:, 0]
        scalars["feat_pca1"] = proj[:, 1]
        scalars["feat_pca2"] = proj[:, 2]
    ply_rgb = (_color_by_id(instance)[:, :3] * 255).astype(np.uint8)
    write_ply(ply_path, coord, ply_rgb, scalars)

    # ---------------- PLY: features, RGB = DinoV3 PCA (direct feature view) ----
    if proj is not None:
        feat_rgb = features_to_rgb(proj, valid)
        feat_path = ply_path[:-4] + "_feat.ply"
        write_ply(feat_path, coord, feat_rgb,
                  {"feat_pca0": proj[:, 0], "feat_pca1": proj[:, 1],
                   "feat_pca2": proj[:, 2], "feat_valid": valid.astype(np.float32)})
        print(f"      features: {100.0 * valid.mean():.0f}% of points")

    # ---------------- PNG: rendered images only (no 3D rendering) ----------------
    n_img = min(len(images), MAX_IMAGES)
    if n_img == 0:
        return  # nothing to put in the PNG; the cloud lives in the PLY(s)
    img_cols = min(5, n_img)
    img_rows = int(np.ceil(n_img / img_cols))
    fig = plt.figure(figsize=(3.2 * img_cols, 3.2 * img_rows))
    for k in range(n_img):
        ax = fig.add_subplot(img_rows, img_cols, k + 1)
        ax.imshow(_to_numpy(images[k]))
        ax.set_title(f"view {k}", fontsize=8)
        ax.axis("off")
    fig.suptitle(str(data.get("name", os.path.basename(png_path))), fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# dataset / driver
# --------------------------------------------------------------------------- #
def build_raw_dataset(data_cfg):
    """Instantiate a dataset for raw get_data() access: no transforms, images on.

    Datasets have different __init__ signatures (e.g. HDF5_Dataset has no
    'cache'), so only apply overrides the target class actually accepts.
    """
    cfg = dict(data_cfg)
    cfg.pop("transform", None)
    cls = DATASETS.get(cfg["type"])
    accepted = set(inspect.signature(cls.__init__).parameters)
    # NOTE: do NOT force cache=False -- that makes get_data recompute the whole
    # pipeline (incl. the expensive per-view pixel->point projection) every call.
    # Respect the config's cache so the prebuilt cached.pth is reused (fast, and
    # it already contains images/mappings/superpoints).
    overrides = {"test_mode": False,
                 "load_images": True, "load_image_files": True}
    for k, v in overrides.items():
        if k in accepted:
            cfg[k] = v
        else:
            cfg.pop(k, None)
    return build_dataset(cfg)


def find_feature_loader(data_cfg):
    """If the split's transforms load a feature cache, return a LoadImageFeatures
    that attaches features WITHOUT dropping the raw images (raw mode only)."""
    for t in (data_cfg.get("transform", []) or []):
        if t.get("type") == "LoadImageFeatures" and t.get("features_root"):
            return LoadImageFeatures(features_root=t["features_root"], drop_keys=())
    return None


def build_pipeline(data_cfg):
    """Build the config's transform pipeline MINUS ToTensor/Collect, so the
    output stays numpy but coord/features are in the exact space the model is
    trained on (CenterShift, augmentation, GridSample, InstanceParser, ...).
    LoadImageFeatures stays in, so features are attached the same way as training."""
    cfgs = [t for t in (data_cfg.get("transform", []) or [])
            if t.get("type") not in ("ToTensor", "Collect")]
    return Compose(cfgs) if cfgs else None


def dump_split(dataset, feat_loader, pipeline, n, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    total = len(dataset.data_list) if hasattr(dataset, "data_list") else len(dataset)
    n = min(n, total)
    indices = np.linspace(0, total - 1, n).astype(int) if total else []
    print(f"[{tag}] {total} samples available; dumping {len(indices)} -> {out_dir}")
    for i, idx in enumerate(indices):
        try:
            data = dataset.get_data(int(idx))
            base = f"{tag}_{i:02d}_idx{int(idx)}"
            visualize_sample(data,
                             os.path.join(out_dir, base + ".png"),
                             os.path.join(out_dir, base + ".ply"),
                             feat_loader=feat_loader, pipeline=pipeline)
            print(f"  [{tag}] sample {i} (idx {idx}) -> {base}.png/.ply")
        except Exception as e:
            print(f"  [{tag}] FAILED idx {idx}: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to a Pointcept config")
    ap.add_argument("-n", "--num", type=int, default=10, help="samples per split")
    ap.add_argument("-o", "--out", default="visualizations", help="output root dir")
    ap.add_argument("--space", choices=["model", "raw"], default="model",
                    help="'model': apply the config's transforms (training coord "
                         "space); 'raw': untransformed get_data coords")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0]
    out_root = os.path.join(args.out, name)

    for tag in ("train", "test"):
        if tag not in cfg.data:
            print(f"[{tag}] not in config.data; skipping")
            continue
        ds = build_raw_dataset(cfg.data[tag])
        if args.space == "model":
            pipeline = build_pipeline(cfg.data[tag])
            feat_loader = None  # the pipeline's LoadImageFeatures attaches them
            print(f"[{tag}] space=model: applying config transforms "
                  f"(minus ToTensor/Collect)")
        else:
            pipeline = None
            feat_loader = find_feature_loader(cfg.data[tag])
            if feat_loader is None:
                print(f"[{tag}] no feature cache in transforms; skipping features")
        dump_split(ds, feat_loader, pipeline, args.num,
                   os.path.join(out_root, tag), tag)

    print(f"\nDone. PNG + PLY under: {out_root}")


if __name__ == "__main__":
    main()

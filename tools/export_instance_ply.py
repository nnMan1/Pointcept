"""Izvoz rezultata instance segmentacije kao PLY oblaci tacaka.

Po sceni se pisu tri fajla, da se greske vide bez poredjenja boja u glavi:

  *_gt.ply    GT instance, svaka svoja boja
  *_pred.ply  predikcije, obojene bojom GT instance sa kojom se najvise
              preklapaju — tako ista boja u oba fajla znaci pogodjen dio
  *_err.ply   zelena = tacka je pripisana ispravnoj instanci, crvena = pogresnoj,
              siva = nijedna maska je ne pokriva

Maske iz `--options test.save_predictions=True` su vec KNN-om vracene na
`origin_coord`, pa su poravnate sa sirovim tackama scene.

  python tools/export_instance_ply.py --config <cfg> --pred-dir <viz>/predictions \\
      --out <viz>/ply --n 9
"""

import argparse
import os

import numpy as np
import trimesh

from pointcept.utils.config import Config
from pointcept.datasets import build_dataset
from pointcept.datasets.builder import DATASETS


UNASSIGNED = np.array([70, 70, 74], dtype=np.uint8)      # tamnosivo
UNMATCHED = np.array([235, 235, 235], dtype=np.uint8)    # predikcija bez GT para
OK_COL = np.array([90, 190, 120], dtype=np.uint8)
BAD_COL = np.array([215, 70, 55], dtype=np.uint8)


def palette(n):
    """n vizuelno razdvojenih boja: setnja po nijansi zlatnim presjekom."""
    import colorsys
    cols = []
    for i in range(max(n, 1)):
        h = (i * 0.61803398875) % 1.0
        s = 0.55 + 0.30 * ((i % 3) / 2.0)
        v = 0.72 + 0.25 * ((i % 2))
        r, g, b = colorsys.hsv_to_rgb(h, s, min(v, 1.0))
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(cols, dtype=np.uint8)


def build_raw_dataset(cfg):
    """Dataset bez transformacija — treba nam samo sirovi coord/instance."""
    dc = dict(cfg.data.test)
    dc["transform"] = []
    dc["load_images"] = False
    cls = DATASETS.get(dc["type"])
    return build_dataset(dc)


def assign_points(masks, scores, n_pts):
    """Svaka tacka dobija masku najveceg skora koja je pokriva."""
    pred = np.full(n_pts, -1, dtype=np.int64)
    for k in np.argsort(scores):           # rastuce -> jaci prepisuju slabije
        pred[masks[k]] = k
    return pred


def match_to_gt(pred, gt, ignore=-1):
    """Za svaku predvidjenu instancu, GT instanca sa najvecim IoU."""
    gt_ids = [g for g in np.unique(gt) if g != ignore]
    gt_masks = {g: (gt == g) for g in gt_ids}
    mapping = {}
    for p in np.unique(pred):
        if p < 0:
            continue
        pm = pred == p
        best_g, best_iou = None, 0.0
        for g, gm in gt_masks.items():
            inter = np.count_nonzero(pm & gm)
            if inter == 0:
                continue
            iou = inter / np.count_nonzero(pm | gm)
            if iou > best_iou:
                best_g, best_iou = g, iou
        mapping[p] = (best_g, best_iou)
    return mapping


def gt_miou(pred, gt, ignore=-1):
    """Prosjek preko GT instanci najboljeg IoU sa bilo kojom predikcijom."""
    gt_ids = [g for g in np.unique(gt) if g != ignore]
    if not gt_ids:
        return 0.0
    pred_ids = [p for p in np.unique(pred) if p >= 0]
    pred_masks = {p: (pred == p) for p in pred_ids}
    out = []
    for g in gt_ids:
        gm = gt == g
        best = 0.0
        for pm in pred_masks.values():
            inter = np.count_nonzero(pm & gm)
            if inter:
                best = max(best, inter / np.count_nonzero(pm | gm))
        out.append(best)
    return float(np.mean(out))


def write(path, coord, colors):
    trimesh.PointCloud(vertices=np.asarray(coord, dtype=np.float32),
                       colors=np.asarray(colors, dtype=np.uint8)).export(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=9, help="koliko scena izvesti")
    ap.add_argument("--match-thr", type=float, default=0.25)
    a = ap.parse_args()

    cfg = Config.fromfile(a.config)
    ds = build_raw_dataset(cfg)
    os.makedirs(a.out, exist_ok=True)

    by_name = {}
    for i in range(len(ds)):
        by_name[ds.get_data_name(i)] = i

    # prvi prolaz: ocijeni sve scene, pa izaberi reprezentativne
    scored = []
    for f in sorted(os.listdir(a.pred_dir)):
        if not f.endswith(".npz"):
            continue
        name = f[:-4]
        if name not in by_name:
            print(f"  preskacem {name[:60]} (nema u datasetu)")
            continue
        z = np.load(os.path.join(a.pred_dir, f))
        n_pts = int(z["n_points"])
        raw = ds.get_data(by_name[name])
        gt = np.asarray(raw["instance"])
        if len(gt) != n_pts:
            print(f"  preskacem {name[:60]} (n {len(gt)} != {n_pts})")
            continue
        masks = np.unpackbits(z["masks"], axis=1)[:, :n_pts].astype(bool)
        pred = assign_points(masks, np.asarray(z["scores"]), n_pts)
        scored.append((gt_miou(pred, gt), name, pred, gt, raw))
        print(f"  {scored[-1][0]:.3f}  {name[:70]}", flush=True)

    if not scored:
        print("nema nijedne scene")
        return
    scored.sort(key=lambda t: t[0])
    n = min(a.n, len(scored))
    third = max(1, n // 3)
    picks = ([("najgore", s) for s in scored[:third]]
             + [("srednje", s) for s in scored[len(scored) // 2: len(scored) // 2 + third]]
             + [("najbolje", s) for s in scored[-third:]])

    print(f"\nizvozim {len(picks)} scena u {a.out}")
    for tag, (score, name, pred, gt, raw) in picks:
        coord = np.asarray(raw["coord"])
        short = name.split("_")[-1][:40].strip("-_") or "scene"
        base = os.path.join(a.out, f"{tag}_{score:.3f}_{short}")

        gt_ids = [g for g in np.unique(gt) if g != -1]
        pal = palette(max(len(gt_ids), 1))
        gt_color = {g: pal[i] for i, g in enumerate(gt_ids)}

        cg = np.tile(UNASSIGNED, (len(coord), 1))
        for g, c in gt_color.items():
            cg[gt == g] = c
        write(base + "_gt.ply", coord, cg)

        mapping = match_to_gt(pred, gt)
        cp = np.tile(UNASSIGNED, (len(coord), 1))
        ce = np.tile(UNASSIGNED, (len(coord), 1))
        for p, (g, iou) in mapping.items():
            m = pred == p
            if g is not None and iou >= a.match_thr:
                cp[m] = gt_color[g]
                ce[m] = np.where((gt[m] == g)[:, None], OK_COL, BAD_COL)
            else:
                cp[m] = UNMATCHED
                ce[m] = BAD_COL
        write(base + "_pred.ply", coord, cp)
        write(base + "_err.ply", coord, ce)

        n_pred = len([p for p in np.unique(pred) if p >= 0])
        acc = float((ce == OK_COL).all(1).mean())
        print(f"  {tag:9s} GT_mIoU={score:.3f}  GT instanci={len(gt_ids):3d}  "
              f"predikcija={n_pred:3d}  tacaka ispravno={100*acc:5.1f}%  {short}")


if __name__ == "__main__":
    main()

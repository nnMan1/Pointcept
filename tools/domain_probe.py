"""Domain-invariance probe for the PTv3 encoder.

Question: does the encoder representation still separate synthetic from real?
If a linear probe on pooled encoder features is near chance, feature-level
domain alignment (DANN etc.) has nothing to fix and the whole adversarial
family can be closed. If it is near-perfect, alignment is worth one run.

Three descriptors per scene, so the encoder can be compared against its own
input rather than against chance alone:
  input     - cheap geometry stats of the point cloud the encoder is fed
  backbone  - pooled PT-V3 output ('feat', pre mask_features_head)
  encoder   - pooled Encoder output ('features', what the decoder sees)

Stage 1 (this script, GPU) dumps descriptors to an .npz.
Stage 2 (--analyze, CPU) fits the logistic probes with GroupKFold.

  python tools/domain_probe.py --config <cfg> --weight <ckpt> --out probe.npz
  python tools/domain_probe.py --analyze probe.npz
"""

import argparse
import os

import numpy as np


# ---------------------------------------------------------------- descriptors
def input_stats(coord, n_sub=4096, rng=None):
    """Geometry-only descriptor of the input cloud (the control)."""
    c = coord.astype(np.float64)
    n = len(c)
    ext = c.max(0) - c.min(0)
    ext = np.sort(ext)[::-1]
    diag = float(np.linalg.norm(ext))
    ctr = c - c.mean(0)
    # PCA shape ratios
    ev = np.linalg.eigvalsh(np.cov(ctr.T))[::-1]
    ev = np.maximum(ev, 1e-12)
    # radial spread
    r = np.linalg.norm(ctr, axis=1)
    # local density: nn distance on a subsample
    idx = rng.choice(n, size=min(n_sub, n), replace=False)
    s = c[idx]
    d = np.linalg.norm(s[:, None, :] - s[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    return np.array([
        np.log(n), np.log(diag + 1e-9),
        ext[1] / (ext[0] + 1e-9), ext[2] / (ext[0] + 1e-9),
        np.log(ev[1] / ev[0]), np.log(ev[2] / ev[0]),
        r.mean() / (diag + 1e-9), r.std() / (diag + 1e-9),
        np.percentile(r, 90) / (diag + 1e-9),
        np.log(np.median(nn) + 1e-9),
        np.log(np.percentile(nn, 90) + 1e-9),
        float(np.percentile(nn, 90) / (np.median(nn) + 1e-9)),
    ], dtype=np.float64)


def instance_rows(coord, inst, feat, ignore=-1, min_pts=30):
    """Per-instance descriptors, so features are compared at MATCHED geometry.

    Scene-level mean+std is permutation-invariant but NOT invariant to which
    points got sampled: real scans have holes and uneven density, so the pooled
    mean shifts even when the per-point mapping is identical. Pooling per
    physical part instead, and carrying that part's own geometry alongside,
    lets the analysis regress the geometry out and ask the sharper question:
    for parts of the same shape and size, does the feature depend on domain?
    """
    out = []
    f = feat.float().cpu().numpy()
    for iid in np.unique(inst):
        if iid == ignore:
            continue
        m = inst == iid
        n = int(m.sum())
        if n < min_pts:
            continue
        c = coord[m].astype(np.float64)
        ctr = c - c.mean(0)
        ext = np.sort(c.max(0) - c.min(0))[::-1]
        diag = float(np.linalg.norm(ext)) + 1e-9
        ev = np.maximum(np.linalg.eigvalsh(np.cov(ctr.T))[::-1], 1e-12)
        r = np.linalg.norm(ctr, axis=1)
        geom = np.array([
            np.log(n), np.log(diag),
            ext[1] / (ext[0] + 1e-9), ext[2] / (ext[0] + 1e-9),
            np.log(ev[1] / ev[0]), np.log(ev[2] / ev[0]),
            r.mean() / diag, r.std() / diag,
            n / (diag ** 3),                      # point density per unit volume
        ], dtype=np.float64)
        fi = f[m]
        out.append((np.concatenate([fi.mean(0), fi.std(0)]).astype(np.float64), geom))
    return out


def pool(feat):
    """Permutation-invariant pooling of per-point features -> descriptor."""
    f = feat.float()
    return np.concatenate([
        f.mean(0).cpu().numpy(),
        f.std(0).cpu().numpy(),
    ]).astype(np.float64)


# ---------------------------------------------------------------- domains
FEAT_ROOT = "/leonardo_scratch/large/userexternal/vdosljak/dino_features"


def _set_features_root(transform, root):
    out = []
    for t in transform:
        t = dict(t)
        if t.get("type") == "LoadImageFeatures":
            t["features_root"] = root
        out.append(t)
    return out


def domain_configs(cfg):
    """Three domains sharing the eval-time (non-augmenting) transform.

    real         - CETIM scans, the target domain
    synth_motor  - motor-synthetic, built to mimic those same motors:
                   CONTENT IS MATCHED to a large part of real, so separability
                   here is domain signal rather than "different objects"
    synth_merged - generic merged synthetic: different content, same renderer.
                   Gives the content-only reference scale, without which an
                   AUC of 1.0 cannot be attributed to the domain at all.
    """
    val, test = dict(cfg.data["val"]), dict(cfg.data["test"])
    merged = dict(val)
    merged["split"] = "val"
    merged["data_root"] = "data/segment-assembly-merged-synthetic/data"
    merged["transform"] = _set_features_root(val["transform"], f"{FEAT_ROOT}/mech_synth/features")

    motor = dict(val)
    motor["split"] = "train"  # ftrain+fval, 47 scenes (fval alone is only 5)

    return {"real": test, "synth_motor": motor, "synth_merged": merged}


# ---------------------------------------------------------------- extraction
def extract(cfg_path, weight, out_path, n_per_domain, seed):
    import torch
    from pointcept.utils.config import Config
    from pointcept.datasets import build_dataset, collate_fn
    from pointcept.models import build_model

    cfg = Config.fromfile(cfg_path)
    rng = np.random.default_rng(seed)

    model = build_model(cfg.model).cuda().eval()
    ckpt = torch.load(weight, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {weight}: {len(missing)} missing, {len(unexpected)} unexpected", flush=True)

    rows, irows = [], []
    for domain, dcfg in domain_configs(cfg).items():
        ds = build_dataset(dcfg)
        n = min(n_per_domain, len(ds))
        order = rng.permutation(len(ds))[:n]
        print(f"[{domain}] {dcfg['type']} split={dcfg['split']} "
              f"root={dcfg['data_root']} -> {n}/{len(ds)} scenes", flush=True)

        for c, i in enumerate(order):
            data = collate_fn([ds[int(i)]])
            name = data.get("name", [f"{domain}_{i}"])
            name = name[0] if isinstance(name, (list, tuple)) else str(name)
            coord_np = data["coord"].numpy() if hasattr(data["coord"], "numpy") else np.asarray(data["coord"])

            for k, v in data.items():
                if hasattr(v, "cuda"):
                    data[k] = v.cuda(non_blocking=True)

            with torch.no_grad():  # fp32, kao InstSegTester; amp ruši spconv implicit_gemm
                bb = model.encoder.backbone(data)
                enc = model.encoder.mask_features_head(bb["feat"])

            inst_np = data["instance"].cpu().numpy()
            for fvec, gvec in instance_rows(coord_np, inst_np, enc):
                irows.append(dict(domain=domain, name=name, feat=fvec, geom=gvec))

            rows.append(dict(
                domain=domain, name=name,
                d_input=input_stats(coord_np, rng=rng),
                d_backbone=pool(bb["feat"]),
                d_encoder=pool(enc),
            ))
            if (c + 1) % 20 == 0:
                print(f"  {c + 1}/{n}", flush=True)
            del data, bb, enc
            torch.cuda.empty_cache()

    np.savez_compressed(
        out_path,
        domain=np.array([r["domain"] for r in rows]),
        name=np.array([r["name"] for r in rows]),
        d_input=np.stack([r["d_input"] for r in rows]),
        d_backbone=np.stack([r["d_backbone"] for r in rows]),
        d_encoder=np.stack([r["d_encoder"] for r in rows]),
        i_domain=np.array([r["domain"] for r in irows]),
        i_name=np.array([r["name"] for r in irows]),
        i_feat=np.stack([r["feat"] for r in irows]),
        i_geom=np.stack([r["geom"] for r in irows]),
    )
    print(f"wrote {out_path}: {len(rows)} scenes, {len(irows)} instances", flush=True)


# ---------------------------------------------------------------- analysis
def group_of(name):
    """Object identity, so CV folds never share an object."""
    p = str(name).replace("\\", "/").strip("/").split("/")
    p = [x for x in p if x]
    return "/".join(p[:2]) if len(p) >= 2 else (p[0] if p else "?")


def _auc(X, y, groups, seed, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    ng = min(len(set(groups[y == 0])), len(set(groups[y == 1])))
    n_splits = max(2, min(5, ng))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y, groups):
        if len(set(y[te])) < 2:
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, C=C, class_weight="balanced"),
        ).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def analyze(npz_path, seed=0):
    from sklearn.metrics import roc_auc_score

    z = np.load(npz_path, allow_pickle=True)
    dom = z["domain"]
    groups_all = np.array([f"{d}:{group_of(n)}" for d, n in zip(dom, z["name"])])
    keys = [("d_input", "input"), ("d_backbone", "backbone"), ("d_encoder", "encoder")]

    print("scenes per domain:")
    for d in sorted(set(dom)):
        m = dom == d
        print(f"  {d:<13} {int(m.sum()):>4} scenes, {len(set(groups_all[m])):>3} objects")

    pairs = [
        ("real", "synth_motor", "TARGET: content matched -> domain signal"),
        ("real", "synth_merged", "domain + content differ (upper reference)"),
        ("synth_motor", "synth_merged", "CONTROL: same renderer, content differs"),
    ]

    res = {}
    for a, b, note in pairs:
        m = (dom == a) | (dom == b)
        if m.sum() == 0 or len(set(dom[m])) < 2:
            continue
        y = (dom[m] == b).astype(int)
        g = groups_all[m]
        print(f"\n{a}  vs  {b}    [{note}]")
        print(f"  {'descriptor':<10} {'dim':>5} {'AUC':>17} {'AUC(C=0.01)':>13}")
        for key, label in keys:
            X = z[key][m]
            mu, sd = _auc(X, y, g, seed)
            lo, _ = _auc(X, y, g, seed, C=0.01)
            res[(a, b, label)] = mu
            print(f"  {label:<10} {X.shape[1]:>5} {mu:>10.4f} +-{sd:5.4f} {lo:>13.4f}")

    # null control: permuted labels within real -> validates the machinery
    m = dom == "real"
    if m.sum() >= 10:
        rng = np.random.default_rng(seed)
        y = rng.permutation((np.arange(int(m.sum())) % 2).astype(int))
        mu, sd = _auc(z["d_encoder"][m], y, groups_all[m], seed)
        print(f"\nnull control (permuted labels within real): AUC {mu:.4f} +-{sd:.4f}"
              f"   {'OK' if abs(mu - 0.5) < 0.15 else 'PROBE IS BROKEN'}")

    # where does the signal live: best single feature dimension
    m = (dom == "real") | (dom == "synth_motor")
    if m.sum() and len(set(dom[m])) == 2:
        y = (dom[m] == "synth_motor").astype(int)
        X = z["d_encoder"][m]
        per = np.array([max(roc_auc_score(y, X[:, k]), 1 - roc_auc_score(y, X[:, k]))
                        for k in range(X.shape[1])])
        top = np.sort(per)[::-1][:5]
        print(f"\nencoder, real vs synth_motor - best single dims (in-sample): "
              + ", ".join(f"{v:.3f}" for v in top))
        print(f"  dims with AUC>0.9: {int((per > 0.9).sum())}/{len(per)}")

    # ------------------------------------------------------------- verdict
    tgt = res.get(("real", "synth_motor", "encoder"))
    inp = res.get(("real", "synth_motor", "input"))
    ctl = res.get(("synth_motor", "synth_merged", "encoder"))
    print("\n" + "=" * 70)
    if tgt is None:
        print("verdict: not enough data")
        return res
    print(f"encoder separability real<->synth_motor : {tgt:.4f}")
    print(f"  same at the input                     : {inp:.4f}" if inp is not None else "")
    print(f"  content-only reference (synth<->synth) : {ctl:.4f}" if ctl is not None else "")
    print()
    if tgt < 0.60:
        print("Encoder features are already domain-invariant.")
        print("-> adversarial alignment (DANN/GAN) has nothing to fix. Close that direction.")
    elif ctl is not None and tgt <= ctl + 0.02:
        print("The encoder separates real from synthetic no better than it separates")
        print("two synthetic sets from each other -- i.e. the probe is reading CONTENT,")
        print("not domain. Domain alignment would be chasing an artefact.")
    elif inp is not None and inp > 0.95 and tgt > 0.95:
        print("Already separable at the input; the encoder merely preserves it.")
        print("-> fix the input statistic that leaks before reaching for DANN.")
    elif inp is not None and inp < 0.75 < tgt:
        print("Inputs are NOT linearly separable, yet the encoder makes them so.")
        print("-> the encoder manufactures the gap; feature alignment is the right tool.")
    else:
        print("Partial separability; alignment is a marginal lever at best.")
    return res


def _auc_resid(F, G, y, groups, seed, residualize=True):
    """AUC for domain-from-features, optionally after regressing geometry out.

    The regression is fit on the training fold only, so the geometry that gets
    removed is never fit on the points being scored.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score

    ng = min(len(set(groups[y == 0])), len(set(groups[y == 1])))
    cv = StratifiedGroupKFold(n_splits=max(2, min(5, ng)), shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(F, y, groups):
        if len(set(y[te])) < 2:
            continue
        Ftr, Fte = F[tr], F[te]
        if residualize:
            gs = StandardScaler().fit(G[tr])
            rg = Ridge(alpha=1.0).fit(gs.transform(G[tr]), F[tr])
            Ftr = F[tr] - rg.predict(gs.transform(G[tr]))
            Fte = F[te] - rg.predict(gs.transform(G[te]))
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced"),
        ).fit(Ftr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Fte)[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def _common_support(G, y, cols=(0, 1), lo=10, hi=90):
    """Keep instances inside the geometric range both domains actually cover."""
    keep = np.ones(len(G), dtype=bool)
    for c in cols:
        a, b = G[y == 0][:, c], G[y == 1][:, c]
        lo_v = max(np.percentile(a, lo), np.percentile(b, lo))
        hi_v = min(np.percentile(a, hi), np.percentile(b, hi))
        keep &= (G[:, c] >= lo_v) & (G[:, c] <= hi_v)
    return keep


def analyze_instances(npz_path, seed=0):
    z = np.load(npz_path, allow_pickle=True)
    if "i_domain" not in z:
        print("npz has no instance-level arrays; re-run extraction")
        return
    dom, F, G = z["i_domain"], z["i_feat"], z["i_geom"]
    groups_all = np.array([str(n) for n in z["i_name"]])

    print("instances per domain:")
    for d in sorted(set(dom)):
        print(f"  {d:<13} {int((dom == d).sum()):>5}")

    pairs = [("real", "synth_motor", "TARGET"), ("synth_motor", "synth_merged", "CONTROL: content only")]
    res = {}
    print(f"\n{'pair':<28} {'geom only':>11} {'feat raw':>11} {'feat|geom':>12} {'+matched':>11}")
    print("-" * 78)
    for a, b, note in pairs:
        m = (dom == a) | (dom == b)
        if len(set(dom[m])) < 2:
            continue
        y = (dom[m] == b).astype(int)
        g, Fm, Gm = groups_all[m], F[m], G[m]

        auc_g, _ = _auc_resid(Gm, Gm, y, g, seed, residualize=False)
        auc_f, sf = _auc_resid(Fm, Gm, y, g, seed, residualize=False)
        auc_r, sr = _auc_resid(Fm, Gm, y, g, seed, residualize=True)

        keep = _common_support(Gm, y)
        if keep.sum() > 50 and len(set(y[keep])) == 2:
            auc_m, sm = _auc_resid(Fm[keep], Gm[keep], y[keep], g[keep], seed, residualize=True)
            mtxt = f"{auc_m:.3f}+-{sm:.3f}"
        else:
            auc_m, mtxt = None, "n/a"
        res[(a, b)] = dict(geom=auc_g, feat=auc_f, resid=auc_r, matched=auc_m,
                           n=int(m.sum()), n_matched=int(keep.sum()))
        print(f"{a}/{b:<16} {auc_g:>11.3f} {auc_f:>6.3f}+-{sf:.3f} {auc_r:>6.3f}+-{sr:.3f} {mtxt:>11}")

    m = dom == "real"
    if m.sum() > 100:
        rng = np.random.default_rng(seed)
        y = rng.permutation((np.arange(int(m.sum())) % 2).astype(int))
        mu, _ = _auc_resid(F[m], G[m], y, groups_all[m], seed, residualize=True)
        print(f"\nnull control (permuted labels, real only): {mu:.3f}"
              f"   {'OK' if abs(mu - 0.5) < 0.15 else 'PROBE IS BROKEN'}")

    t, c = res.get(("real", "synth_motor")), res.get(("synth_motor", "synth_merged"))
    print("\n" + "=" * 78)
    if not t:
        return res
    key = t["matched"] if t["matched"] is not None else t["resid"]
    ckey = (c["matched"] if c and c["matched"] is not None else (c["resid"] if c else None))
    print(f"domain-from-features at matched geometry, real vs synth_motor : {key:.3f}"
          f"   (n={t['n_matched']} instances)")
    if ckey is not None:
        print(f"same quantity for two synthetic sets (content-only control)   : {ckey:.3f}")
    print()
    if key < 0.60:
        print("Per-part features do not carry domain once geometry is removed.")
        print("-> encoder is domain-invariant at the level that matters; alignment is pointless.")
    elif ckey is not None and key <= ckey + 0.03:
        print("Domain is no more decodable than content is -> still reading WHICH PART,")
        print("not WHICH DOMAIN. Alignment would chase an artefact.")
    else:
        print("A real per-part domain shift survives geometry removal.")
        print("-> scene-level pooling had hidden it; feature alignment becomes worth one run.")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--weight")
    ap.add_argument("--out", default="domain_probe.npz")
    ap.add_argument("--n-per-domain", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyze", metavar="NPZ")
    ap.add_argument("--analyze-inst", metavar="NPZ")
    a = ap.parse_args()

    if a.analyze_inst:
        analyze_instances(a.analyze_inst, seed=a.seed)
    elif a.analyze:
        analyze(a.analyze, seed=a.seed)
    else:
        assert a.config and a.weight, "--config and --weight required for extraction"
        extract(a.config, a.weight, a.out, a.n_per_domain, a.seed)


if __name__ == "__main__":
    main()

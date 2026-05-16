"""Vizualizacija razlike izmedju implicitnog FN (ScanNet/VOC) i FN sa -inf confidence.

Generise tri figure:
  1. instance_matching.png — greedy matching predikcija na GT instance
  2. pr_curves_compare.png — PR krive za oba pristupa
  3. ap_area_compare.png   — VOC interpolovana kriva sa popunjenom povrsinom (AP)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------- zajednicki primer iz prethodnog odgovora ----------
num_gt = 5
# (confidence, is_TP, gt_match_id ili None)
preds = [
    (0.90, True,  "GT-1"),
    (0.85, False, None),
    (0.70, True,  "GT-3"),
    (0.60, True,  "GT-2"),
    (0.40, False, None),
    (0.30, False, None),
]
fn_ids = ["GT-4", "GT-5"]  # dva nepogodjena GT-a

y_true = np.array([int(p[1]) for p in preds])
y_score = np.array([p[0] for p in preds])

# ====================================================================
# Figure 1: matching predikcija <-> GT
# ====================================================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(-0.5, 7.5)
ax.set_aspect("equal")
ax.axis("off")

# Predikcije sa leve strane (sortirane po confidence desc)
pred_x = 1.5
gt_x = 7.5
y_top = 6.5

ax.text(pred_x, y_top + 0.6, "Predikcije\n(sort. po confidence)", ha="center",
        fontsize=11, fontweight="bold")
ax.text(gt_x, y_top + 0.6, "GT instance", ha="center",
        fontsize=11, fontweight="bold")

pred_positions = {}
for i, (conf, is_tp, match_id) in enumerate(preds):
    y = y_top - i
    color = "#4caf50" if is_tp else "#f44336"
    edge = "#1b5e20" if is_tp else "#b71c1c"
    ax.add_patch(mpatches.FancyBboxPatch(
        (pred_x - 0.9, y - 0.28), 1.8, 0.56,
        boxstyle="round,pad=0.02", facecolor=color, edgecolor=edge, linewidth=1.5, alpha=0.85
    ))
    label = f"P{i+1}  conf={conf:.2f}"
    ax.text(pred_x, y, label, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    pred_positions[i] = (pred_x + 0.9, y)

# GT-ovi sa desne strane (GT-1..GT-5)
gt_positions = {}
gt_ids = ["GT-1", "GT-2", "GT-3", "GT-4", "GT-5"]
for i, gid in enumerate(gt_ids):
    y = y_top - i - 0.5
    is_matched = gid not in fn_ids
    color = "#42a5f5" if is_matched else "#ffa726"
    edge = "#0d47a1" if is_matched else "#e65100"
    ax.add_patch(mpatches.FancyBboxPatch(
        (gt_x - 0.7, y - 0.28), 1.4, 0.56,
        boxstyle="round,pad=0.02", facecolor=color, edgecolor=edge, linewidth=1.5, alpha=0.85
    ))
    ax.text(gt_x, y, gid, ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    gt_positions[gid] = (gt_x - 0.7, y)

# Match linije
gt_y = {gid: gt_positions[gid][1] for gid in gt_ids}
for i, (conf, is_tp, match_id) in enumerate(preds):
    if is_tp:
        px, py = pred_positions[i]
        gx, gy = gt_positions[match_id]
        ax.annotate("", xy=(gx, gy), xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=2, alpha=0.7))
    else:
        px, py = pred_positions[i]
        ax.annotate("", xy=(px + 1.0, py), xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.5, alpha=0.6,
                                    linestyle="--"))
        ax.text(px + 1.2, py, "FP", fontsize=9, color="#b71c1c", va="center", fontweight="bold")

# FN oznaka
for gid in fn_ids:
    gx, gy = gt_positions[gid]
    ax.text(gx - 0.5, gy, "FN", fontsize=9, color="#e65100", va="center",
            ha="right", fontweight="bold")

# Legenda
legend_y = -0.2
legend_elements = [
    mpatches.Patch(facecolor="#4caf50", edgecolor="#1b5e20", label="TP predikcija"),
    mpatches.Patch(facecolor="#f44336", edgecolor="#b71c1c", label="FP predikcija"),
    mpatches.Patch(facecolor="#42a5f5", edgecolor="#0d47a1", label="GT pogodjen"),
    mpatches.Patch(facecolor="#ffa726", edgecolor="#e65100", label="GT propusten (FN)"),
]
ax.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=10,
          bbox_to_anchor=(0.5, -0.05), frameon=False)

ax.set_title(f"Greedy matching pri IoU ≥ 0.5 (num_gt={num_gt}, TP=3, FP=3, FN=2)",
             fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "instance_matching.png"), dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved instance_matching.png")

# ====================================================================
# Figure 2: PR krive za oba pristupa (raw, bez interpolacije + sa VOC)
# ====================================================================
def compute_pr(y_true, y_score, num_gt):
    order = np.argsort(-y_score, kind="stable")
    y_t = y_true[order]
    cum_tp = np.cumsum(y_t == 1)
    cum_fp = np.cumsum(y_t == 0)
    recall = cum_tp / num_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    return recall, precision

def voc_interp(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))
    return mrec, mpre, ap

# Pristup 1: implicitan FN
r1, p1 = compute_pr(y_true, y_score, num_gt)
mrec1, mpre1, ap1 = voc_interp(r1, p1)

# Pristup 2: -inf FN entries
y_true_ext = np.concatenate([y_true, np.ones(num_gt - int(y_true.sum()), dtype=int)])
y_score_ext = np.concatenate([y_score, np.full(num_gt - int(y_true.sum()), -np.inf)])
r2, p2 = compute_pr(y_true_ext, y_score_ext, num_gt)
mrec2, mpre2, ap2 = voc_interp(r2, p2)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

for ax, (mrec, mpre, ap, raw_r, raw_p, title, color) in zip(
    axes,
    [(mrec1, mpre1, ap1, r1, p1, "Pristup 1 — implicitan FN (ScanNet/VOC)", "#1976d2"),
     (mrec2, mpre2, ap2, r2, p2, "Pristup 2 — eksplicitan FN sa −∞", "#d32f2f")]
):
    # Stepenasta (raw) PR kriva
    ax.step(raw_r, raw_p, where="post", color=color, linewidth=1.8, alpha=0.5,
            label="raw PR (cum_tp/(cum_tp+cum_fp))")
    ax.scatter(raw_r, raw_p, color=color, s=55, zorder=5, alpha=0.7)

    # VOC interpolovana kriva (monotono opadajuca)
    # Konstruisemo "step from the left" oblik za interpolovanu krivu
    voc_x = []
    voc_y = []
    for i in range(len(mrec) - 1):
        voc_x.extend([mrec[i], mrec[i + 1]])
        voc_y.extend([mpre[i + 1], mpre[i + 1]])
    ax.plot(voc_x, voc_y, color=color, linewidth=2.8, linestyle="-",
            label="VOC interpolacija")
    ax.fill_between(voc_x, 0, voc_y, color=color, alpha=0.15)

    # Anotacija svake raw tacke
    for x, y in zip(raw_r, raw_p):
        ax.annotate(f"({x:.2f}, {y:.2f})", xy=(x, y),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=8, color=color)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_title(f"{title}\nAP = {ap:.3f}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=10)
    ax.axvline(0.6, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(0.6, 1.02, "max stvarni\nrecall = 0.6", fontsize=8, ha="center", color="gray")

axes[0].set_ylabel("Precision", fontsize=11)

fig.suptitle(f"PR krive: num_gt={num_gt}, TP=3, FP=3, FN=2 (ΔAP = {ap2-ap1:+.3f})",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "pr_curves_compare.png"), dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"Saved pr_curves_compare.png  (AP1={ap1:.3f}, AP2={ap2:.3f})")

# ====================================================================
# Figure 3: AP povrsina sa istaknutom razlikom
# ====================================================================
fig, ax = plt.subplots(figsize=(12, 7))

# Pristup 1 - osnovna povrsina
voc_x1, voc_y1 = [], []
for i in range(len(mrec1) - 1):
    voc_x1.extend([mrec1[i], mrec1[i + 1]])
    voc_y1.extend([mpre1[i + 1], mpre1[i + 1]])
ax.fill_between(voc_x1, 0, voc_y1, color="#1976d2", alpha=0.45,
                label=f"AP pristup 1 (implicit FN) = {ap1:.3f}", step=None)
ax.plot(voc_x1, voc_y1, color="#0d47a1", linewidth=2.5)

# Pristup 2 - dodatna povrsina koja se javlja
voc_x2, voc_y2 = [], []
for i in range(len(mrec2) - 1):
    voc_x2.extend([mrec2[i], mrec2[i + 1]])
    voc_y2.extend([mpre2[i + 1], mpre2[i + 1]])

# Interpoliramo y_1 na x_2 da bismo dobili razliku
y1_on_x2 = np.interp(voc_x2, voc_x1, voc_y1)
y2_arr = np.array(voc_y2)
ax.fill_between(voc_x2, y1_on_x2, y2_arr,
                where=(y2_arr > y1_on_x2),
                color="#d32f2f", alpha=0.5,
                label=f"DODATNA povrsina kod pristupa 2 = {ap2-ap1:.3f}")

# Pristup 2 - obris
ax.plot(voc_x2, voc_y2, color="#b71c1c", linewidth=2.5, linestyle="--",
        label=f"AP pristup 2 (−∞ FN) = {ap2:.3f}")

# Raw tacke
ax.scatter(r1, p1, color="#0d47a1", s=60, zorder=5, label="raw tacke (oba pristupa)")
ax.scatter(r2[len(r1):], p2[len(r1):], color="#b71c1c", s=80, marker="s", zorder=5,
           label="raw tacke (samo pristup 2: −∞ FN)")

# Strelica koja pokazuje "novu" povrsinu
ax.annotate(
    "FN-ovi sa conf=−∞\ngeneraseu dodatne TP\ntacke posle recall=0.6\n→ AP raste za 0.25",
    xy=(0.85, 0.5), xytext=(0.45, 0.85),
    fontsize=11, color="#b71c1c", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=2),
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffebee", edgecolor="#b71c1c")
)

ax.axvline(0.6, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(0.61, 0.05, "max recall pristupa 1", fontsize=9, color="gray")

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.08)
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Vizualizacija razlike u AP-u: implicitan FN vs. eksplicitan FN sa −∞",
             fontsize=13, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left", fontsize=10, framealpha=0.95)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "ap_area_compare.png"), dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"Saved ap_area_compare.png  (diff = {ap2-ap1:+.3f})")

print("\nFinal numbers:")
print(f"  Pristup 1 (implicitan FN): AP = {ap1:.4f}")
print(f"  Pristup 2 (-inf FN):       AP = {ap2:.4f}")
print(f"  Razlika:                   {ap2-ap1:+.4f}")

"""Vizualizacija precision/recall/F1 kao funkcija confidence threshold-a.

Generise:
  1. threshold_pr_curves.png   — precision, recall, F1 vs threshold (oba pristupa)
  2. threshold_step_table.png  — tabelarni prikaz kako se TP/FP/precision/recall menjaju
                                  korak po korak dok spustamo threshold
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))

# Isti primer kao u prethodnim figurama
num_gt = 5
preds = [
    (0.90, 1),  # TP
    (0.85, 0),  # FP
    (0.70, 1),  # TP
    (0.60, 1),  # TP
    (0.40, 0),  # FP
    (0.30, 0),  # FP
]
y_true = np.array([p[1] for p in preds])
y_score = np.array([p[0] for p in preds])

# ---- Threshold sweep ----
# Za svaki threshold t, predikcije sa score >= t su "predicted positive"
# TP(t)  = broj predikcija sa y_true=1 i score >= t
# FP(t)  = broj predikcija sa y_true=0 i score >= t
# precision(t) = TP / (TP + FP)
# recall(t)    = TP / num_gt
# F1(t)        = 2*P*R / (P + R)

# Threshold-i: iznad svih (nista predikcija) -> svaki score -> ispod svih (sve)
unique_scores = sorted(set(y_score.tolist()), reverse=True)
# Dodajemo "iznad svega" (1.01) i "ispod svega" (0.0) za vizualizaciju
thresholds = [1.01] + unique_scores + [0.0]

rows = []
for t in thresholds:
    predicted = y_score >= t
    tp = int(np.sum(predicted & (y_true == 1)))
    fp = int(np.sum(predicted & (y_true == 0)))
    fn = num_gt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # konvencija: nista predikcija -> P=1
    recall = tp / num_gt
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    rows.append({
        "threshold": t,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })

# ====================================================================
# Figure 1: precision / recall / F1 vs threshold
# ====================================================================
fig, ax = plt.subplots(figsize=(12, 7))

t_arr = np.array([r["threshold"] for r in rows])
p_arr = np.array([r["precision"] for r in rows])
r_arr = np.array([r["recall"] for r in rows])
f1_arr = np.array([r["f1"] for r in rows])

# Step plotovi (jer su metrike "step funkcije" threshold-a)
ax.step(t_arr, p_arr, where="post", color="#1976d2", linewidth=2.5, label="Precision")
ax.step(t_arr, r_arr, where="post", color="#d32f2f", linewidth=2.5, label="Recall")
ax.step(t_arr, f1_arr, where="post", color="#388e3c", linewidth=2.5, linestyle="--", label="F1 score")

# Tacke na svakom thresholdu (osim "iznad svega")
ax.scatter(t_arr[1:], p_arr[1:], color="#1976d2", s=60, zorder=5)
ax.scatter(t_arr[1:], r_arr[1:], color="#d32f2f", s=60, zorder=5)
ax.scatter(t_arr[1:], f1_arr[1:], color="#388e3c", s=50, zorder=5, marker="s")

# Anotiraj svaki unique score na x-osi sa TP/FP brojevima ispod
for r in rows[1:-1]:  # preskoci 1.01 i 0.0
    t = r["threshold"]
    ax.axvline(t, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.text(t, -0.10, f"t={t:.2f}", ha="center", fontsize=8.5, color="gray")
    ax.text(t, -0.16, f"TP={r['tp']} FP={r['fp']}", ha="center", fontsize=8, color="gray")

# Maksimalni F1
best_idx = int(np.argmax(f1_arr))
best_t = t_arr[best_idx]
best_f1 = f1_arr[best_idx]
ax.annotate(
    f"Best F1 = {best_f1:.3f}\npri t = {best_t:.2f}\n(P={p_arr[best_idx]:.2f}, R={r_arr[best_idx]:.2f})",
    xy=(best_t, best_f1), xytext=(best_t - 0.25, best_f1 + 0.18),
    fontsize=10, color="#1b5e20", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.5),
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f5e9", edgecolor="#1b5e20")
)

ax.set_xlim(1.05, -0.05)  # threshold ide od visokog ka niskom (sleva nadesno = labavije)
ax.set_ylim(-0.22, 1.08)
ax.set_xlabel("Confidence threshold  t  (predikcija prihvacena ako score ≥ t)", fontsize=12)
ax.set_ylabel("Vrednost metrike", fontsize=12)
ax.set_title(
    "Precision, Recall i F1 kao funkcija confidence threshold-a\n"
    f"(num_gt={num_gt}, 3 TP + 3 FP predikcija, 2 FN GT-a)",
    fontsize=13, fontweight="bold"
)
ax.grid(True, alpha=0.3, axis="y")
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.axhline(0, color="black", linewidth=0.5)
ax.axhline(1, color="black", linewidth=0.5, alpha=0.3)

# Komentar levo (visok threshold) i desno (nizak threshold)
ax.text(1.0, 1.03, "STROG\n(malo predikcija,\nvisok P, nizak R)",
        ha="right", fontsize=9, color="#1976d2", fontweight="bold")
ax.text(0.0, 1.03, "LABAV\n(sve predikcije,\nnizak P, max R)",
        ha="left", fontsize=9, color="#d32f2f", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "threshold_pr_curves.png"), dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved threshold_pr_curves.png")

# ====================================================================
# Figure 2: tabela korak-po-korak (kako se TP/FP/P/R menjaju)
# ====================================================================
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.axis("off")

headers = ["t (prag)", "TP", "FP", "FN", "Predicted", "Precision", "Recall", "F1"]
col_widths = [0.10, 0.07, 0.07, 0.07, 0.11, 0.13, 0.12, 0.11]
n_cols = len(headers)
n_rows = len(rows) + 1  # + header

row_h = 0.085
y_start = 0.9
x_start = 0.06

# Header
x = x_start
for i, h in enumerate(headers):
    ax.add_patch(Rectangle((x, y_start), col_widths[i], row_h,
                            facecolor="#37474f", edgecolor="white", linewidth=1.5))
    ax.text(x + col_widths[i] / 2, y_start + row_h / 2, h,
            ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    x += col_widths[i]

# Najveci F1 indeks (u "rows" listi)
best_row_idx = int(np.argmax([r["f1"] for r in rows]))

for ri, r in enumerate(rows):
    y = y_start - (ri + 1) * row_h
    x = x_start
    is_best = (ri == best_row_idx)
    bg = "#fff9c4" if is_best else ("#fafafa" if ri % 2 == 0 else "white")

    # Threshold display label
    if r["threshold"] > 1.0:
        t_label = "→ ∞ (niko)"
    elif r["threshold"] <= 0.0:
        t_label = "→ 0 (svi)"
    else:
        t_label = f"{r['threshold']:.2f}"

    pred_count = r["tp"] + r["fp"]
    cells = [
        t_label,
        str(r["tp"]),
        str(r["fp"]),
        str(r["fn"]),
        str(pred_count),
        f"{r['precision']:.3f}",
        f"{r['recall']:.3f}",
        f"{r['f1']:.3f}",
    ]
    cell_colors = ["white"] * 8
    cell_colors[5] = "#bbdefb"   # precision col
    cell_colors[6] = "#ffcdd2"   # recall col
    cell_colors[7] = "#c8e6c9"   # f1 col

    for i, c in enumerate(cells):
        face = bg if not is_best else "#fff59d"
        if not is_best and i in (5, 6, 7):
            face = cell_colors[i]
        ax.add_patch(Rectangle((x, y), col_widths[i], row_h,
                                facecolor=face, edgecolor="#cfd8dc", linewidth=0.8))
        weight = "bold" if (is_best and i in (5, 6, 7)) else "normal"
        ax.text(x + col_widths[i] / 2, y + row_h / 2, c,
                ha="center", va="center", fontsize=10.5, fontweight=weight)
        x += col_widths[i]

# Anotacija za best F1
ax.text(x_start + sum(col_widths) + 0.01, y_start - (best_row_idx + 1) * row_h - row_h / 2 + row_h,
        "← Best F1 prag", fontsize=10, color="#827717", fontweight="bold",
        va="center")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title(
    "Korak-po-korak: kako se metrike menjaju dok spustamo threshold\n"
    "(svaki red = jedna predikcija vise je prihvacena)",
    fontsize=13, fontweight="bold", pad=10
)

# Mala legenda dole
ax.text(x_start, 0.05,
        "Precision = TP / (TP+FP)  •  Recall = TP / num_gt  •  F1 = 2·P·R / (P+R)",
        fontsize=10, color="#37474f", style="italic")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "threshold_step_table.png"), dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved threshold_step_table.png")

# Print summary
print("\nThreshold sweep:")
print(f"  {'t':>6} {'TP':>3} {'FP':>3} {'FN':>3} {'P':>7} {'R':>7} {'F1':>7}")
for r in rows:
    t = r["threshold"]
    t_s = f">{1.0:.2f}" if t > 1.0 else f"{t:.2f}"
    print(f"  {t_s:>6} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} "
          f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")

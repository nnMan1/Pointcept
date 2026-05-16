"""Dekompozicija problema sa -inf FN:
Pokazuje kako se precision i recall menjaju korak-po-korak (po sortiranom
y_score) za oba pristupa, da bi se videlo gde tacno -inf trik 'krade' AP.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))

num_gt = 5
y_true = np.array([1, 0, 1, 1, 0, 0])         # po confidence desc
y_score = np.array([0.90, 0.85, 0.70, 0.60, 0.40, 0.30])

# Pristup 1: samo predikcije
cum_tp_1 = np.cumsum(y_true == 1)
cum_fp_1 = np.cumsum(y_true == 0)
precision_1 = cum_tp_1 / np.maximum(cum_tp_1 + cum_fp_1, 1)
recall_1 = cum_tp_1 / num_gt

# Pristup 2: dodajemo -inf FN entries
num_fn = num_gt - int(y_true.sum())
y_true_2 = np.concatenate([y_true, np.ones(num_fn, dtype=int)])
y_score_2 = np.concatenate([y_score, np.full(num_fn, -np.inf)])
cum_tp_2 = np.cumsum(y_true_2 == 1)
cum_fp_2 = np.cumsum(y_true_2 == 0)
precision_2 = cum_tp_2 / np.maximum(cum_tp_2 + cum_fp_2, 1)
recall_2 = cum_tp_2 / num_gt

steps_1 = np.arange(1, len(precision_1) + 1)
steps_2 = np.arange(1, len(precision_2) + 1)

# Labele tipa entry-ja
labels_2 = ["TP", "FP", "TP", "TP", "FP", "FP", "FN(−∞)", "FN(−∞)"]

fig, (ax_p, ax_r) = plt.subplots(1, 2, figsize=(15, 6.5))

# ---------------- Precision ----------------
ax_p.plot(steps_1, precision_1, "-o", color="#1976d2", linewidth=2.5,
          markersize=11, label="Pristup 1 (implicit FN)", zorder=4)
ax_p.plot(steps_2, precision_2, "--s", color="#d32f2f", linewidth=2.5,
          markersize=10, label="Pristup 2 (−∞ FN)", zorder=3, alpha=0.85)

# Istaknuti samo "fake" -inf tacke
fake_x = steps_2[6:]
fake_y = precision_2[6:]
ax_p.scatter(fake_x, fake_y, s=300, facecolor="none", edgecolor="#d32f2f",
             linewidth=2.5, zorder=5, label="−∞ FN entries (\"lazne\")")

# Anotacije na svakom koraku
for i, (s, p, lbl) in enumerate(zip(steps_2, precision_2, labels_2)):
    color = "#d32f2f" if "FN" in lbl else ("#1976d2" if i < 6 else "#d32f2f")
    ax_p.annotate(f"{lbl}\nP={p:.3f}", xy=(s, p),
                  xytext=(0, 14 if i % 2 == 0 else -28),
                  textcoords="offset points", ha="center", fontsize=8.5,
                  color=color, fontweight="bold")

# Osenci region gde se javlja rast (efekat -inf-a)
ax_p.axvspan(6.5, 8.5, alpha=0.15, color="#d32f2f", zorder=1)
ax_p.text(7.5, 0.93, "P RASTE\n(cum_tp +1,\ncum_fp +0)",
          ha="center", fontsize=10, color="#b71c1c", fontweight="bold")

ax_p.set_xlabel("Korak (predikcije sortirane po confidence-u silazno)", fontsize=11)
ax_p.set_ylabel("Precision = cum_tp / (cum_tp + cum_fp)", fontsize=11)
ax_p.set_title("Precision po koraku\n(efekat −∞: P se podiže na kraju)",
               fontsize=12, fontweight="bold")
ax_p.set_xlim(0.5, 8.7)
ax_p.set_ylim(0, 1.08)
ax_p.set_xticks(steps_2)
ax_p.grid(True, alpha=0.3)
ax_p.legend(loc="lower left", fontsize=10)

# ---------------- Recall ----------------
ax_r.plot(steps_1, recall_1, "-o", color="#1976d2", linewidth=2.5,
          markersize=11, label="Pristup 1 (implicit FN)", zorder=4)
ax_r.plot(steps_2, recall_2, "--s", color="#d32f2f", linewidth=2.5,
          markersize=10, label="Pristup 2 (−∞ FN)", zorder=3, alpha=0.85)

ax_r.scatter(fake_x, recall_2[6:], s=300, facecolor="none", edgecolor="#d32f2f",
             linewidth=2.5, zorder=5, label="−∞ FN entries (\"lazne\")")

# Horizontalne linije pokazuju max recall
ax_r.axhline(0.6, color="#1976d2", linestyle=":", linewidth=1.5, alpha=0.6)
ax_r.text(0.6, 0.62, "max stvarni recall = 0.6 (TP/num_gt)", color="#1976d2",
          fontsize=9, fontweight="bold")
ax_r.axhline(1.0, color="#d32f2f", linestyle=":", linewidth=1.5, alpha=0.6)
ax_r.text(0.6, 1.02, "lazni recall = 1.0 (svi GT-ovi 'pogodjeni' na −∞)",
          color="#d32f2f", fontsize=9, fontweight="bold")

# Anotacije na recallima
for i, (s, r, lbl) in enumerate(zip(steps_2, recall_2, labels_2)):
    color = "#d32f2f" if "FN" in lbl else "#1976d2"
    ax_r.annotate(f"{lbl}\nR={r:.2f}", xy=(s, r),
                  xytext=(0, 12 if i in (0, 2, 3, 6, 7) else -25),
                  textcoords="offset points", ha="center", fontsize=8.5,
                  color=color, fontweight="bold")

ax_r.axvspan(6.5, 8.5, alpha=0.15, color="#d32f2f", zorder=1)
ax_r.text(7.5, 0.32, "R RASTE\nIZNAD 0.6\n(svaki −∞\nFN = +1 TP)",
          ha="center", fontsize=10, color="#b71c1c", fontweight="bold")

ax_r.set_xlabel("Korak (predikcije sortirane po confidence-u silazno)", fontsize=11)
ax_r.set_ylabel("Recall = cum_tp / num_gt", fontsize=11)
ax_r.set_title("Recall po koraku\n(efekat −∞: R prelazi 0.6 i ide do 1.0)",
               fontsize=12, fontweight="bold")
ax_r.set_xlim(0.5, 8.7)
ax_r.set_ylim(0, 1.08)
ax_r.set_xticks(steps_2)
ax_r.grid(True, alpha=0.3)
ax_r.legend(loc="lower right", fontsize=10)

fig.suptitle(
    "Sta tacno radi −∞ FN trik: POVECAVA I RECALL I PRECISION u repu krive",
    fontsize=14, fontweight="bold", y=1.00
)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "inf_decomposition.png"), dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved inf_decomposition.png")

# Print numerical breakdown
print("\nKorak | tip      | cum_tp cum_fp | precision recall")
print("-" * 60)
for i in range(len(steps_2)):
    in_p1 = i < len(steps_1)
    src = "real" if i < 6 else "FN(-inf)"
    p = precision_2[i]
    r = recall_2[i]
    print(f"  {i+1}   | {labels_2[i]:8s} | {cum_tp_2[i]:6d} {cum_fp_2[i]:6d} | "
          f"{p:9.4f} {r:7.4f}")

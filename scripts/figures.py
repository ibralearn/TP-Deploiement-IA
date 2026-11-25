"""Génère les figures de comparaison des 4 pipelines à partir de data/resultats.json."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
V = json.load(open(os.path.join(ROOT, "resultats", "resultats.json")))["variants"]

COL = {"reference": "#8a8984", "pipeline1": "#257de0", "pipeline2": "#f47230"}
HATCH = {"FP16": "////", "INT8": "xxx", "FP32": ""}
TXT, TXT2, GRID = "#191919", "#666666", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 200, "hatch.color": "white", "hatch.linewidth": 1.0,
})

def name(v):
    return f"{v['session'].split(' ')[0]} · {v['label']}"

def legend(fig, y=-0.02, extra=()):
    h = [Patch(facecolor=COL["reference"], label="Référence PyTorch eager (même session)"),
         Patch(facecolor=COL["pipeline1"], label="Pipeline n°1 : ONNX → OpenVINO (CPU) / TensorRT (GPU)"),
         Patch(facecolor=COL["pipeline2"], label="Pipeline n°2 : TorchScript / torch.compile")]
    h += [Patch(facecolor="#b5b4ae", hatch=HATCH[p], edgecolor="white", label=f"Variante {p}") for p in extra]
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, y), fontsize=8.5)

def hbars(ax, vs, vals, errs, fmt, xlabel, title, xmax=None):
    y = np.arange(len(vs))[::-1]
    bars = ax.barh(y, vals, xerr=errs, height=0.62, color=[COL[v["family"]] for v in vs],
                   edgecolor="white", linewidth=1.5, error_kw=dict(elinewidth=1, ecolor=TXT, capsize=3))
    for b, v in zip(bars, vs):
        b.set_hatch(HATCH[v["prec"]])
    top = max(m + (e or 0) for m, e in zip(vals, errs if errs is not None else [0] * len(vals)))
    for yi, m, e in zip(y, vals, errs if errs is not None else [0] * len(vals)):
        ax.text(m + (e or 0) + top * 0.012, yi, fmt(m), va="center", fontsize=8, color=TXT)
    ax.set_yticks(y)
    ax.set_yticklabels([name(v) for v in vs], fontsize=8)
    ax.set_xlim(0, xmax or top * 1.22)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=TXT)

def fr(x, d=1):
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s

def two_panel(metric, fname, xlabel, suptitle, fmt, err_idx=1, key=None, titles=None, extra=("FP16", "INT8")):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, plat in zip(axes, ["CPU", "GPU"]):
        vs = [v for v in V if v["platform"] == plat]
        vals = [key(v)[0] for v in vs]
        errs = [key(v)[err_idx] for v in vs]
        hbars(ax, vs, vals, errs, fmt, xlabel, titles[plat] if titles else plat)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TXT)
    legend(fig, extra=extra)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(os.path.join(OUT, fname), bbox_inches="tight")
    plt.close(fig)

# 1. Débit
two_panel("thr", "fig1_debit.png", "images / s (moyenne ± écart-type, 10 passes)",
          "Débit de traitement des quatre pipelines", lambda m: fr(m, 1), key=lambda v: v["thr"],
          titles={"CPU": "CPU (Xeon 2,2 GHz, 2 vCPU)", "GPU": "GPU (Tesla T4)"})
# 2. Latence
two_panel("lat", "fig2_latence.png", "ms par batch de 10 images (moyenne ± écart-type)",
          "Temps d'inférence (latence d'un batch de 10 images)", lambda m: fr(m, 1 if m > 100 else 2),
          key=lambda v: v["lat"], titles={"CPU": "CPU (Xeon 2,2 GHz, 2 vCPU)", "GPU": "GPU (Tesla T4)"})
# 4. Énergie par image + puissance (consommation énergétique)
ST = json.load(open(os.path.join(ROOT, "resultats", "statistiques_detaillees.json")))["variantes"]
fig, axes = plt.subplots(2, 2, figsize=(11, 9.2))
for col, plat in enumerate(["CPU", "GPU"]):
    vs = [v for v in V if v["platform"] == plat]
    hbars(axes[0, col], vs, [v["E"][0] for v in vs], [v["E"][1] for v in vs],
          lambda m: fr(m, 0 if m > 100 else 1), "mJ par image (moyenne ± écart-type, 3 mesures)",
          "CPU — énergie par image, ESTIMÉE" if plat == "CPU" else "GPU — énergie par image, MESURÉE (NVML)")
    pw = [ST[v["id"]]["puissance"] for v in vs]
    hbars(axes[1, col], vs, [p[0] for p in pw], [p[1] or 0 for p in pw], lambda m: fr(m, 1),
          "W (moyenne" + (" ± écart-type, 3 mesures)" if plat == "GPU" else " ; estimation : 10 W par vCPU actif)"),
          "CPU — puissance moyenne, ESTIMÉE" if plat == "CPU" else "GPU — puissance moyenne de la carte, MESURÉE",
          xmax=26 if plat == "CPU" else 90)
    if plat == "GPU":
        axes[1, col].axvline(70, color=TXT2, lw=0.9, ls="--")
        axes[1, col].text(70.6, len(vs) - 0.45, "plafond T4 : 70 W", fontsize=7.5, color=TXT2)
fig.suptitle("Consommation énergétique : énergie par image et puissance moyenne pendant l'inférence", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=TXT)
legend(fig, y=-0.01, extra=("FP16", "INT8"))
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
fig.savefig(os.path.join(OUT, "fig4_energie.png"), bbox_inches="tight")
plt.close(fig)

# 3. Top-1 + accord
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
vs = V
y = np.arange(len(vs))[::-1]
for ax, (k, title, xl) in zip(axes, [("top1", "Précision Top-1 sur les 50 images", "%"),
                                      ("acc", "Accord Top-1 avec la référence eager", "% des 50 prédictions identiques")]):
    vals = [v[k] for v in vs]
    bars = ax.barh(y, vals, height=0.62, color=[COL[v["family"]] for v in vs], edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, vs):
        b.set_hatch(HATCH[v["prec"]])
    for yi, m in zip(y, vals):
        ax.text(m + 1, yi, f"{m:.0f} %", va="center", fontsize=8, color=TXT)
    ax.set_xlim(0, 112)
    ax.set_yticks(y)
    ax.set_yticklabels([name(v) if ax is axes[0] else "" for v in vs], fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xl)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=TXT)
fig.suptitle("Qualité des prédictions", x=0.01, ha="left", fontsize=12, fontweight="bold", color=TXT)
legend(fig, extra=("FP16", "INT8"))
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(os.path.join(OUT, "fig3_top1.png"), bbox_inches="tight")
plt.close(fig)

# 5. Accélération vs référence de la même session (sans unité -> un seul axe commun)
fig, ax = plt.subplots(figsize=(11, 5.4))
vs = [v for v in V if v["family"] != "reference"]
refs = {v["session"]: v for v in V if v["family"] == "reference"}
sp = [v["thr"][0] / refs[v["session"]]["thr"][0] for v in vs]
# incertitude relative combinée (σ/μ de la variante et de la référence)
spe = [s * np.hypot(v["thr"][1] / v["thr"][0], refs[v["session"]]["thr"][1] / refs[v["session"]]["thr"][0]) for s, v in zip(sp, vs)]
hbars(ax, vs, sp, spe, lambda m: "×" + fr(m, 2), "accélération du débit (× par rapport à la référence eager de la même session)",
      "Accélération apportée par chaque pipeline", xmax=5.0)
ax.axvline(1, color=TXT2, lw=1, ls="--")
ax.text(1.02, len(vs) - 0.4, "référence = ×1", fontsize=7.5, color=TXT2)
legend(fig, extra=("FP16", "INT8"))
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(os.path.join(OUT, "fig5_acceleration.png"), bbox_inches="tight")
plt.close(fig)

# 6. Variabilité : CV du débit + plage [min ; max] normalisée par la moyenne
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
vs = V
y = np.arange(len(vs))[::-1]
ax = axes[0]
cv = [v["thr"][4] for v in vs]
bars = ax.barh(y, cv, height=0.62, color=[COL[v["family"]] for v in vs], edgecolor="white", linewidth=1.5)
for b, v in zip(bars, vs):
    b.set_hatch(HATCH[v["prec"]])
for yi, m in zip(y, cv):
    ax.text(m + 0.4, yi, fr(m, 1) + " %", va="center", fontsize=8, color=TXT)
for x, lab in [(2, "2 %"), (5, "5 %")]:
    ax.axvline(x, color=TXT2, lw=0.9, ls="--")
ax.text(0.3, len(vs) - 0.35, "très stable", fontsize=7, color=TXT2)
ax.text(5.3, len(vs) - 0.35, "variable (> 5 %)", fontsize=7, color=TXT2)
ax.set_xlim(0, 28)
ax.set_yticks(y)
ax.set_yticklabels([name(v) for v in vs], fontsize=8)
ax.grid(axis="y", visible=False)
ax.set_xlabel("coefficient de variation du débit (σ / moyenne, %)")
ax.set_title("Dispersion sur 10 passes", loc="left", fontsize=10.5, fontweight="bold", color=TXT)
ax = axes[1]
for yi, v in zip(y, vs):
    m, s, lo, hi = v["thr"][0], v["thr"][1], v["thr"][2], v["thr"][3]
    c = COL[v["family"]]
    ax.plot([100 * lo / m, 100 * hi / m], [yi, yi], color=c, lw=2, solid_capstyle="round")
    ax.plot([100 * (m - s) / m, 100 * (m + s) / m], [yi, yi], color=c, lw=6, alpha=0.35, solid_capstyle="butt")
    ax.plot([100], [yi], "o", ms=6, color=c, mec="white", mew=1.5)
ax.axvline(100, color=TXT2, lw=0.9, ls="--")
ax.set_yticks(y)
ax.set_yticklabels([])
ax.set_xlim(55, 130)
ax.grid(axis="y", visible=False)
ax.set_xlabel("débit en % de la moyenne de la variante (trait fin : min–max ; bande : ± 1 σ)")
ax.set_title("Plage des 10 passes mesurées", loc="left", fontsize=10.5, fontweight="bold", color=TXT)
fig.suptitle("Stabilité des mesures de débit", x=0.01, ha="left", fontsize=12, fontweight="bold", color=TXT)
legend(fig, extra=("FP16", "INT8"))
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig(os.path.join(OUT, "fig6_variabilite.png"), bbox_inches="tight")
plt.close(fig)

# 7. Étude des tailles de batch (GPU, hors protocole commun)
BS = [1, 10, 50]
S1 = {"Réf. eager FP32": [279.74, 892.84, 1079.75], "TensorRT FP32": [663.21, 1129.24, 1157.17], "TensorRT FP16": [1494.55, 4053.62, 5294.08]}
S2 = {"Réf. eager FP32": [291.7, 892.2, 1086.3], "TorchScript FP32": [473.8, 1070.0, 1379.0],
      "compile + CUDA Graphs FP32": [431.6, 904.3, 1017.3], "compile + CUDA Graphs FP16": [1019.0, 2581.2, 4004.5]}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
for ax, S, title, fam in [(axes[0], S1, "GPU-1 : TensorRT", "pipeline1"), (axes[1], S2, "GPU-2 : TorchScript / torch.compile", "pipeline2")]:
    for lab, vals in S.items():
        c = COL["reference"] if lab.startswith("Réf") else COL[fam]
        ls = "--" if "FP16" in lab else (":" if "TorchScript" in lab else "-")
        mk = "s" if "TorchScript" in lab else "o"
        ax.plot(BS, vals, ls=ls, marker=mk, color=c, lw=2, ms=7, mec="white", mew=1.5,
                label=f"{lab} ({fr(vals[0], 0)} → {fr(vals[-1], 0)} img/s)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(BS); ax.set_xticklabels(BS)
    ax.set_xlim(0.8, 65)
    ax.set_xlabel("taille de batch")
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=TXT)
    ax.set_yticks([300, 500, 1000, 2000, 5000])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fr(v, 0)))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylim(220, 8000)
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, handlelength=3.6)
axes[0].set_ylabel("images / s (échelle log)")
fig.suptitle("GPU : effet de la taille de batch sur le débit (étude complémentaire, 5 passes)", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=TXT)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(os.path.join(OUT, "fig7_taille_batch_gpu.png"), bbox_inches="tight")
plt.close(fig)
print("OK", sorted(os.listdir(OUT)))

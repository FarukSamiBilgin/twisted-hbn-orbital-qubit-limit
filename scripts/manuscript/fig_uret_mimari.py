# -*- coding: utf-8 -*-
"""Mimari makale sekilleri: data/mimari_hesap.json -> figs/ (PDF+PNG).

fig1: mimari semasi (AA ucgen orgusu, DFS ciftleri, AB ancilla, koruma katmanlari)
fig2: H1 seviye/kacak semasi (olculen kuyu, alpha, mu matris yapisi)
fig3: J(theta) vs W1(theta) yarisi + calisma pencereleri
fig4: Gamma(V0) derinlestirme zinciri (hedef cizgileri, E21/lambda yan ekseni)
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, RegularPolygon

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
F = os.path.join(HERE, "figs")
os.makedirs(F, exist_ok=True)

plt.rcParams.update({
    "font.size": 8.5, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "figure.dpi": 150,
    "savefig.bbox": "tight", "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
})

M = json.load(open(os.path.join(D, "mimari_hesap.json"), encoding="utf-8"))

# ================================================================== FIG 1
fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                               gridspec_kw={"width_ratios": [1.5, 1.0]},
                               constrained_layout=True)
# (a) orgu: AA ucgen orgusu + DFS ciftleri + AB ancilla
a1 = np.array([1.0, 0.0]); a2 = np.array([0.5, math.sqrt(3) / 2])
AA, AB = [], []
for i in range(-1, 5):
    for j in range(-1, 4):
        p = i * a1 + j * a2
        AA.append(p)
        AB.append(p + (a1 + a2) / 3.0)
AA = np.array(AA); AB = np.array(AB)
axL.set_xlim(-0.6, 4.1); axL.set_ylim(-0.75, 2.8)
# DFS ciftleri: yatay komsu ciftler, aralarinda >=2aM seyreltme
pairs = [((0, 0), (1, 0)), ((3, 0), (4, 0)), ((0.5, 1), (1.5, 1)),
         ((3.5, 1), (4.5, 1)), ((1, 2), (2, 2))]


def pos(c):
    return c[0] * a1 + c[1] * (a2 - a1 * 0.5) * 0 + np.array([c[0], c[1] * math.sqrt(3) / 2])


for (c1, c2) in pairs:
    p1 = np.array([c1[0], c1[1] * math.sqrt(3) / 2])
    p2 = np.array([c2[0], c2[1] * math.sqrt(3) / 2])
    axL.plot(*zip(p1, p2), color="#4c72b0", lw=2.2, zorder=2)
for p in AA:
    axL.plot(*p, "o", ms=7, mfc="#aec7e8", mec="#4c72b0", mew=1.0, zorder=3)
for p in AB:
    if -0.5 < p[0] < 4.0 and -0.6 < p[1] < 2.6:
        axL.plot(*p, "^", ms=5, mfc="#f2c894", mec="#c44e52", mew=0.8, zorder=3)
# etiketler
axL.annotate("DFS pair = 1 logical ($J_{XY}$)",
             (0.5, 0.0), (-0.55, -0.68), fontsize=7,
             arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
axL.annotate("AA site (K valley):\ndata qubit $\\{|1\\rangle,|2\\rangle\\}$",
             (2.0, math.sqrt(3)), (2.6, 2.45), fontsize=7,
             arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))
axL.annotate("AB pocket (M valley):\nancilla / erasure flag / reset",
             tuple(AB[9]), (2.3, -0.62), fontsize=7, color="#9e3d3d",
             arrowprops=dict(arrowstyle="-", lw=0.6, color="#9e3d3d"))
axL.set_aspect("equal"); axL.axis("off")
axL.set_title("(a) KILIM-1 lattice: two site species", fontsize=9)

# (b) koruma katmanlari
layers = [
    ("K0  sweet spot $d{=}0$ + gate-embedded XY8", "#f0ecd2"),
    ("K1  DFS pair (collective dephasing 8–38$\\times$)", "#d8f0dc"),
    ("K2  erasure flags (E$_3$ / AB charge / photon)", "#d2ecf0"),
    ("K3  autonomous $\\Lambda$-pump ($10^2$–$10^3$)", "#dce7f5"),
    ("K4  measured stabilizer (late, readout-gated)", "#e8e8f5"),
]
axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.axis("off")
for i, (txt, c) in enumerate(layers):
    y = 0.03 + i * 0.19
    axR.add_patch(FancyBboxPatch((0.02, y), 0.96, 0.16,
                                 boxstyle="round,pad=0.01", fc=c, ec="0.5", lw=0.7))
    axR.text(0.5, y + 0.08, txt, ha="center", va="center", fontsize=7.3)
axR.set_title("(b) protection stack (bottom-up)", fontsize=9)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(F, "figm1_mimari.%s" % ext))
plt.close(fig)

# ================================================================== FIG 2
h1 = M["H1"]
lv = h1["levels_meV"]
labs = [l["label"] for l in h1["labels"]]
fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.0, 2.8),
                              gridspec_kw={"width_ratios": [1.1, 1.0]},
                              constrained_layout=True)
# seviye semasi
uniq = []
for l in h1["labels"]:
    if l["E_meV"] < -20:
        uniq.append((l["E_meV"], l["label"], l["deg"]))
for (E, lab, deg) in uniq:
    ax.hlines(E, 0.15, 0.85, color="#4c72b0", lw=1.6)
    ax.annotate("%s%s" % ("$%s$" % lab.replace("A1", "A_1").replace("A2", "A_2")
                          if lab in ("A1", "A2", "E") else lab,
                          "  ($\\times$%d)" % deg if deg > 1 else ""),
                (0.87, E), fontsize=8, va="center", color="#4c72b0")
ax.annotate("", xy=(0.3, lv[1]), xytext=(0.3, lv[0]),
            arrowprops=dict(arrowstyle="<->", color="k", lw=0.9))
ax.text(0.31, (lv[0] + lv[1]) / 2, "$E_{21}=%.1f$" % h1["E21_meV"], fontsize=7.5)
ax.annotate("", xy=(0.55, lv[3]), xytext=(0.55, lv[1]),
            arrowprops=dict(arrowstyle="<->", color="#c44e52", lw=0.9))
ax.text(0.56, (lv[1] + lv[3]) / 2,
        "$E_{32}=%.1f$\n$\\alpha=+%.1f$" % (lv[3] - lv[1], h1["alpha_E_meV"]),
        fontsize=7.5, color="#c44e52")
ax.set_xlim(0, 1.45); ax.set_xticks([])
ax.set_ylabel("$E$ (meV, cell mean = 0)")
ax.set_title("(a) measured-well spectrum at $1.09^\\circ$", fontsize=9)
# dipol cubugu
mus = [("$\\mu_{12}$ elem.", h1["mu_eleman_a0e"]),
       ("$\\mu_{12}$ summed", h1["mu_toplam_a0e"]),
       ("$\\mu_{23}\\to E'$", h1["mu23_E_a0e"]),
       ("$\\mu_{23}\\to A_1'$", h1["mu23_A1_a0e"])]
axr.barh(range(len(mus)), [m[1] for m in mus], color=["#4c72b0", "#4c72b0",
                                                      "#c44e52", "#c44e52"],
         alpha=0.8, height=0.55)
axr.set_yticks(range(len(mus)), [m[0] for m in mus])
axr.invert_yaxis()
for i, (nm, v) in enumerate(mus):
    axr.text(v + 0.8, i, "%.1f" % v, va="center", fontsize=7.5)
axr.set_xlabel("dipole matrix element ($a_0 e$)")
axr.set_title("(b) transition dipoles", fontsize=9)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(F, "figm2_seviye.%s" % ext))
plt.close(fig)

# ================================================================== FIG 3
race = M["yarisma"]
th = np.array([r["theta"] for r in race])
Jel = np.array([r["J_eleman_meV"] for r in race])
Jto = np.array([r["J_toplam_meV"] for r in race])
W1 = np.array([r["W1_meV"] for r in race])
al = np.array([r["alpha_E_meV"] for r in race])
fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.fill_between(th, Jel, Jto, color="#4c72b0", alpha=0.25,
                label=r"$J_{XY}$ ($\mu$ element–summed band)")
ax.semilogy(th, Jto, "-", color="#4c72b0", lw=1.4)
ax.semilogy(th, Jel, "-", color="#4c72b0", lw=1.0, alpha=0.7)
ax.semilogy(th, np.maximum(W1, 1e-13), "-", color="#c44e52", lw=1.4,
            label="$W_1$ (lowest bandwidth)")
ax.semilogy(th, al, "--", color="#55a868", lw=1.2,
            label=r"$\alpha=E_{32}-E_{21}$ (leakage gap)")
ax.axvspan(1.09, 1.33, color="#4c72b0", alpha=0.08)
ax.text(1.21, 2e-4, "point A\nwindow", ha="center", fontsize=7, color="#33517e")
ax.axvspan(0.4, 0.6, color="#55a868", alpha=0.08)
ax.text(0.5, 2e-4, "point B\n(array)", ha="center", fontsize=7, color="#3d7048")
ax.set_xlabel(r"twist angle $\theta$ (deg)")
ax.set_ylabel("energy (meV)")
ax.set_xlim(0.35, 1.45)
ax.legend(loc="lower right", fontsize=6.6)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(F, "figm3_yarisma.%s" % ext))
plt.close(fig)

# ================================================================== FIG 4
deep = M["derinlestirme"]
dk = json.load(open(os.path.join(D, "dugum_kontrol.json"), encoding="utf-8"))
pts = [(r["V0_esdeger_meV"], r["hbarG1ph_ueV"], r["lambda_um"]) for r in deep]
pts += [(r["V0eq"], r["hbarG_ueV"], 1239.842 / r["E21"]) for r in dk["ince_grid"]]
pts.sort()
V0 = np.array([p[0] for p in pts])
hg = np.array([p[1] for p in pts])
lam = np.array([p[2] for p in pts])
fig, ax = plt.subplots(figsize=(3.6, 2.9))
ax.semilogy(V0, hg, "o-", color="#4c72b0", lw=1.6, ms=4,
            label=r"$\hbar\Gamma_{1\mathrm{ph}}(T{=}0)$, scaled measured well")
for y, lab, yt in ((39.6, "measured well", 26), (1.0, r"1 $\mu$eV interim", 1.35),
                   (0.0411, "10 MHz target", 0.055)):
    ax.axhline(y, color="0.5", ls=":", lw=0.8)
    ax.text(318, yt, lab, fontsize=6.6, color="0.35", ha="right")
ax.annotate("node of the truncation-model ring state\n(ruled out by direct $\\kappa_{AA}$ measurement)",
            (120, 0.3), (130, 0.045), fontsize=6.4, color="#9aa6bd",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#9aa6bd"))
# olculen basinc zinciri (P1 kampanyasi, bicim kaymasiyla cozuldu)
p1 = json.load(open(os.path.join(D, "p1_seviyeler.json"), encoding="utf-8"))
pV = [p1["taban"].get("V0eq_meV", 40.04)] + \
     [p1["dz_%s" % z]["V0eq_meV"] for z in (0.1, 0.2, 0.3, 0.4)]
pG = [p1["taban"]["hbarG1ph_ueV"]] + \
     [p1["dz_%s" % z]["hbarG1ph_ueV"] for z in (0.1, 0.2, 0.3, 0.4)]
ax.semilogy(pV, pG, "D-", color="#2e7d32", lw=1.6, ms=5, zorder=6,
            label="measured pressure chain (P1, shape-resolved)")
ax.annotate(r"$\approx$2.5 GPa", (pV[-1], pG[-1]),
            (pV[-1] + 14, pG[-1] * 1.6), fontsize=6.6, color="#2e7d32")
ax.set_xlabel(r"equivalent well amplitude $V_0^{\rm eq}$ (meV)")
ax.set_ylabel(r"one-phonon linewidth $\hbar\Gamma$ ($\mu$eV)")
ax.set_ylim(0.02, 120)
ax2 = ax.twinx()
ax2.plot(V0, lam, "s--", color="#dd8452", lw=1.0, ms=3, alpha=0.9)
ax2.set_ylabel(r"$\lambda_{21}$ ($\mu$m)", color="#b05f2e")
ax2.tick_params(axis="y", colors="#b05f2e")
ax.legend(loc="upper right", fontsize=6.8)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(F, "figm4_derinlestirme.%s" % ext))
plt.close(fig)

print("figs ->", F)

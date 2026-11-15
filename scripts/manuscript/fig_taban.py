# -*- coding: utf-8 -*-
"""Negatif-sonuc makalesinin ana figuru: T1 kanallari vs E21 (iki-fonon tabani).
Girdi: data/taban_kanallar.json  ->  figs/figm5_taban.(pdf|png)"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "data", "taban_kanallar.json"), encoding="utf-8"))
E = d["E21_meV"]

plt.rcParams.update({"font.size": 8.5})
fig, ax = plt.subplots(figsize=(3.4, 2.75))


def bant(key, dx, color, marker, label):
    xs, lo, hi = [], [], []
    for e, b in zip(E, d[key]):
        if b is None:
            continue
        xs.append(e + dx); lo.append(b[0]); hi.append(b[1])
    mid = [(a * b) ** 0.5 for a, b in zip(lo, hi)]
    ax.errorbar(xs, mid, yerr=[[m - a for m, a in zip(mid, lo)],
                               [b - m for m, b in zip(mid, hi)]],
                fmt=marker, color=color, ms=4, capsize=2, lw=1, label=label)


bant("LA_1f", -1.2, "#4a6fa5", "o", "one-phonon LA")
bant("LA_LA", 0.0, "#c0504d", "s", "LA+LA (full resolvent)")
bant("ZA_ZA", 1.2, "#8064a2", "^", "ZA+ZA (flexural pair)")
te = d["toplam_E21_meV"]; tm = d["toplam_merkez"]; tb = d["toplam_bant"]
ax.errorbar(te, tm, yerr=[[m - b[0] for m, b in zip(tm, tb)],
                          [b[1] - m for m, b in zip(tm, tb)]],
            fmt="D", color="k", ms=4.5, capsize=2.5, lw=1.3,
            label="all channels combined")
ax.axhline(d["hedef_T1_ns"], color="#2e7d32", ls="--", lw=1)
ax.text(45, d["hedef_T1_ns"] * 1.6, "logical-qubit target 30 ns",
        color="#2e7d32", fontsize=7.5)
o = d["optik_yasak_meV"]
ax.axvspan(o[0], o[1], color="0.85", lw=0)
ax.text(o[0] + 0.8, 2e-3, "optical\nforbidden", fontsize=6.5, color="0.35")
ax.axvline(d["ZO_1f_esik_meV"], color="0.5", ls=":", lw=0.9)
ax.text(d["ZO_1f_esik_meV"] + 0.6, 1.5e-3, "ZO 1f onset", rotation=90,
        fontsize=6.3, color="0.4", ha="left", va="bottom")
ax.set_yscale("log")
ax.set_xlim(28, 104)
ax.set_ylim(1e-3, 1e5)
ax.set_xlabel(r"$E_{21}$ (meV)")
ax.set_ylabel(r"$T_1$ (ns), $T=0$")
ax.legend(fontsize=6.4, loc="upper left", frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, "figs", "figm5_taban." + ext), dpi=200)
print("-> figs/figm5_taban.(pdf|png)")

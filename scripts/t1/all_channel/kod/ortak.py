# -*- coding: utf-8 -*-
"""Ortak hazirlik: kuyu + F tablolari + parametre setleri."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from moire import Kuyu, FTablo, noktalar, KOLE
import fonon as fn

DATA = os.path.join(HERE, "..", "data")
hbar = fn.hbar; eC = fn.eC; meV = fn.meV; RHO1 = fn.RHO1

# Kol E faz3 (z_AA(P), yuzey fiti) — yalnizca okunur
_fz = json.load(open(os.path.join(KOLE, "results", "data", "faz3_analiz.json")))
_fh = json.load(open(os.path.join(KOLE, "mantiksal_qubit", "data", "faz3_hassas.json"), encoding="utf-8"))
ZAA = {"ortam_1.09": _fz["bilayer"]["z_opt"]["s000"]["0"],
       "3GPa_1.09": _fz["bilayer"]["z_opt"]["s000"]["3"],
       "5GPa_1.20": _fz["bilayer"]["z_opt"]["s000"]["5"]}
ZAB = {"ortam_1.09": _fz["bilayer"]["z_opt"]["s100"]["0"],
       "3GPa_1.09": _fz["bilayer"]["z_opt"]["s100"]["3"],
       "5GPa_1.20": _fz["bilayer"]["z_opt"]["s100"]["5"]}
YUZ = _fh["yuzey"]


def Ez_turev(nokta):
    """dE_c/dz ve d2E_c/dz2 (eV/A, eV/A^2) AA ve AB'de, Kol E yuzey fiti E=B+A exp(-z/lam)."""
    out = {}
    for tag, s, z in (("AA", "0.0", ZAA[nokta]), ("AB", "1.0", ZAB[nokta]), ("BA", "2.0", ZAB[nokta])):
        y = YUZ[s]; A, lam = y["A_eV"], y["lam_A"]
        ex = A * math.exp(-z / lam)
        out[tag] = {"z": z, "E1": -ex / lam, "E2": ex / lam ** 2}
    return out


def hazirla(nokta, nst=10, kmax=1.6, gcut_b=16.0, h=0.6, pad=1024, gauge=True):
    P4, th = noktalar()[nokta]
    k = Kuyu(P4, th, gcut_b=gcut_b); k.coz(nst); k.gercek_uzay(h=h)
    pairs = [(a, b) for a in range(nst) for b in range(nst)]
    FT = FTablo(k, pairs, kmax=kmax, pad=pad)
    GT = None
    if gauge:
        pr = [(0, 1), (0, 2)]
        GT = {"gx": FTablo(k, pr, kmax=kmax, pad=pad, op="gx"),
              "gy": FTablo(k, pr, kmax=kmax, pad=pad, op="gy")}
    return k, FT, GT


def gapli(dal, gap):
    return fn.Dal(dal.ad + "_a", lambda q: np.sqrt(dal.w(q) ** 2 + gap ** 2),
                  lambda q: dal.w(q) * dal.vg(q) / np.sqrt(dal.w(q) ** 2 + gap ** 2))


# --- parametreler (kaynaklar fonon.py ve rapor) ---
D_DFT = None
_dp = os.path.join(DATA, "dp_sonuc.json")


def dp_parametreleri():
    """Kendi PBE DFT'mden D (vakum-hizali E_c(K)), ikinci-mertebe ve gauge beta."""
    d = json.load(open(_dp))
    E = {k: v["Ec_corner_eV"] for k, v in d.items()}
    D = (E["bi+1"] - E["bi-1"]) / 0.04
    D05 = (E["bi+0.5"] - E["bi-0.5"]) / 0.02
    c2_bi = (E["bi+1"] + E["bi-1"] - 2 * E["e0"]) / 0.01 ** 2       # d2E/de^2 (u_xx=u_yy=e)
    out = {"D_eV": D, "D_eV_0.5": D05, "d2E_biax": c2_bi}
    if "ux+1" in d and "ux-1" in d:
        out["dE_ux"] = (E["ux+1"] - E["ux-1"]) / 0.02
        out["d2E_ux"] = (E["ux+1"] + E["ux-1"] - 2 * E["e0"]) / 0.01 ** 2
    # gauge: K-kosesi enerjisi kayma altinda ~ degismez; minimum konumu kaymasi
    return out, d

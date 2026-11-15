# -*- coding: utf-8 -*-
"""Gamma(V0) dugumunun dogasi: F^2(q) taramasi (s=1 ve s=3) + dugum civari ince s-grid.
Hipotez: duz-dipli/halkamsi durumun gecis form faktoru Bessel-sifiri tasir
(q0*R ~ 3.83); s~3'teki minimum bu geometrik dugumdur."""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KOLE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(KOLE, "preprint"))
from pw_cozucu import Solver, moire_L, KIN

fit = json.load(open(os.path.join(KOLE, "results", "data", "v0_fit_2d_K.json")))["cbK_abs"]
P = {1: fit["yildiz1_P_meV"] * 1e-3, 2: fit["yildiz2_P_meV"] * 1e-3,
     3: fit["yildiz3_P_meV"] * 1e-3}
hbar = 1.054571817e-34; eC = 1.602176634e-19
vLA = 19.9e3; rho = 7.6e-7; D_eV = 4.5


def gf(theta, pp):
    b = 4 * math.pi / (math.sqrt(3.0) * moire_L(theta))
    return max(10.0, 2.6 * math.sqrt(pp / KIN) / b)


def paket(scale, n=241):
    Ps = {k: v * scale for k, v in P.items()}
    s = Solver(1.09, ("measured", Ps), gcut_factor=gf(1.09, 0.2407 * scale))
    w, labels, X, Y, psi, dA, c0 = s.real_space_states(nst=2, n=n)
    E21 = (w[1] - w[0]) * 1e3
    return s, E21, X, Y, psi, dA


def F2(X, Y, psi, dA, q_A, nang=24):
    p0, p1 = psi[..., 0], psi[..., 1]
    tot = 0.0
    for a in range(nang):
        ph = math.pi * a / nang
        q = q_A * np.array([math.cos(ph), math.sin(ph)])
        tot += abs(np.sum(p0.conj() * np.exp(1j * (q[0] * X + q[1] * Y)) * p1) * dA) ** 2
    return tot / nang


def q0A(E21_meV):
    return E21_meV * 1e-3 * eC / hbar / vLA * 1e-10


def hG_ueV(E21_meV, F2v):
    q0 = q0A(E21_meV) * 1e10
    G = (D_eV * eC) ** 2 * q0 ** 2 / (2 * rho * hbar * vLA ** 2) * F2v
    return G * hbar / eC * 1e6


out = {}
for sc in (1.0, 3.0):
    s, E21, X, Y, psi, dA = paket(sc)
    q0 = q0A(E21)
    qs = np.linspace(0.3, 2.5, 23) * q0
    curve = [{"q_over_q0": float(q / q0), "F2": F2(X, Y, psi, dA, float(q))}
             for q in qs]
    out["F2_sweep_s%g" % sc] = {"E21_meV": E21, "q0_A": q0, "curve": curve}
    print("s=%g E21=%.1f q0=%.3f 1/A" % (sc, E21, q0))
    for c in curve:
        print("   q/q0=%.2f F2=%.3e" % (c["q_over_q0"], c["F2"]))

# grid/pencere yakinsakligi (s=3, q=q0)
s, E21, X, Y, psi, dA = paket(3.0, n=241)
f_241 = F2(X, Y, psi, dA, q0A(E21))
s, E21b, X, Y, psi, dA = paket(3.0, n=321)
f_321 = F2(X, Y, psi, dA, q0A(E21b))
print("yakinsaklik s=3: n=241 F2=%.3e ; n=321 F2=%.3e" % (f_241, f_321))
out["yakinsaklik_s3"] = {"n241": f_241, "n321": f_321}

# dugum civari ince s-grid
ince = []
for sc in (2.25, 2.5, 2.75, 3.25, 3.5):
    s, E21, X, Y, psi, dA = paket(sc)
    f = F2(X, Y, psi, dA, q0A(E21))
    hg = hG_ueV(E21, f)
    ince.append({"scale": sc, "V0eq": 40.04 * sc, "E21": E21, "F2": f,
                 "hbarG_ueV": hg})
    print("ince s=%.2f V0eq=%.0f E21=%.1f F2=%.3e hbarG=%.3g ueV"
          % (sc, 40.04 * sc, E21, f, hg))
out["ince_grid"] = ince

json.dump(out, open(os.path.join(HERE, "data", "dugum_kontrol.json"), "w",
                    encoding="utf-8"), indent=1, ensure_ascii=False)
print("-> data/dugum_kontrol.json")

# -*- coding: utf-8 -*-
"""
S0i+ : OLCULEN P1 (basinc) biciminin seviye/koherans zinciri.

Yildiz-basina oran yontemi: P_n(dz) = P_n^{2D taban} x [P_n^{yol}(dz)/P_n^{yol}(taban)]
(yol fitleri ayni aliasing'i tasir -> oranda sadelesir; bicim kaymasi
P2/P1'in dusmesi olarak OLCULDU: -0.309 -> -0.224 @ dz=0.4).

Cikti: her dz icin E21, alpha, mu, uzam, hbar*Gamma_1ph (sayisal form faktor),
J, lambda; data/p1_seviyeler.json; figm4'e olculen-basinc noktalari.
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KOLE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(KOLE, "preprint"))
from pw_cozucu import Solver, moire_L, KIN, A0

fit2d = json.load(open(os.path.join(KOLE, "results", "data", "v0_fit_2d_K.json")))["cbK_abs"]
P2D = {1: fit2d["yildiz1_P_meV"] * 1e-3, 2: fit2d["yildiz2_P_meV"] * 1e-3,
       3: fit2d["yildiz3_P_meV"] * 1e-3}
df = json.load(open(os.path.join(KOLE, "results", "data", "derin_fit.json")))
Ptaban_yol = df["taban_yol"]["P"]

hbar = 1.054571817e-34; eC = 1.602176634e-19; eps0 = 8.8541878128e-12
c_l = 2.99792458e8
rho = 7.6e-7; vLA = 19.9e3; D_eV = 4.5
E2NM = 1.44; A0_NM = 0.0529177


def gf(pp):
    b = 4 * math.pi / (math.sqrt(3.0) * moire_L(1.09))
    return max(10.0, 2.6 * math.sqrt(pp / KIN) / b)


def coz(P):
    pp = 6.0 * abs(P[1])   # kaba pp olcegi
    s = Solver(1.09, ("measured", P), gcut_factor=gf(pp))
    w, labels, D, extent = s.dipoles_extent(nst=6, n=201)
    lv = (np.array(w[:6]) * 1e3).tolist()
    E21 = lv[1] - lv[0]
    alpha = (lv[3] - lv[1]) - E21
    mu_tot = math.hypot(float(D[0, 1]), float(D[0, 2]))
    # sayisal form faktoru
    q0_A = E21 * 1e-3 * eC / hbar / vLA * 1e-10
    w2, lb2, X, Y, psi, dA, c0 = s.real_space_states(nst=2, n=201)
    tot = 0.0; nang = 24
    for a in range(nang):
        ph = math.pi * a / nang
        q = q0_A * np.array([math.cos(ph), math.sin(ph)])
        tot += abs(np.sum(psi[..., 0].conj() *
                          np.exp(1j * (q[0] * X + q[1] * Y)) * psi[..., 1]) * dA) ** 2
    F2 = tot / nang
    q0 = q0_A * 1e10
    G1 = (D_eV * eC) ** 2 * q0 ** 2 / (2 * rho * hbar * vLA ** 2) * F2
    om = E21 * 1e-3 * eC / hbar
    d_Cm = mu_tot * A0 * 1e-10 * eC
    Gr = d_Cm ** 2 * om ** 3 / (3 * math.pi * eps0 * hbar * c_l ** 3)
    J = E2NM * 1e3 * (mu_tot * A0_NM) ** 2 / (5.0 * (moire_L(1.09) / 10) ** 3)
    return {"levels_meV": lv, "E21_meV": E21, "alpha_E_meV": alpha,
            "mu_toplam_a0e": mu_tot, "extent_a0": extent,
            "F2": F2, "hbarG1ph_ueV": G1 * hbar / eC * 1e6,
            "tau1ph_ns": 1e9 / G1, "taurad_us": 1e6 / Gr,
            "J_toplam_meV": J, "lambda_um": 1239.842 / E21}


out = {"taban": coz(P2D)}
print("taban: E21=%.1f alpha=%.1f hbarG=%.3g ueV" %
      (out["taban"]["E21_meV"], out["taban"]["alpha_E_meV"],
       out["taban"]["hbarG1ph_ueV"]))
for dz in (0.1, 0.2, 0.3, 0.4):
    f = df["kampanyalar"]["P1_%s" % dz]["fit"]["P"]
    P = {n + 1: P2D[n + 1] * (f[n] / Ptaban_yol[n]) for n in range(3)}
    r = coz(P)
    r["dz_A"] = dz
    r["P_meV"] = {k: v * 1e3 for k, v in P.items()}
    r["V0eq_meV"] = 40.04 * abs(P[1]) / abs(P2D[1])
    out["dz_%s" % dz] = r
    print("dz=%.1f: V0eq=%.1f P2/P1=%+.3f E21=%.1f alpha=%.1f lam=%.1fum "
          "mu=%.0f hbarG=%.3g ueV tau=%.3g ns J=%.2f" %
          (dz, r["V0eq_meV"], P[2] / P[1], r["E21_meV"], r["alpha_E_meV"],
           r["lambda_um"], r["mu_toplam_a0e"], r["hbarG1ph_ueV"],
           r["tau1ph_ns"], r["J_toplam_meV"]))

json.dump(out, open(os.path.join(HERE, "data", "p1_seviyeler.json"), "w",
                    encoding="utf-8"), indent=1, ensure_ascii=False)
print("-> data/p1_seviyeler.json")

# -*- coding: utf-8 -*-
"""
Mimari icin 4-yildiz (T0a-kisitli) model-bagimliligi kapanisi.

Ortam + basinc varyantlari (rijit P1 Dz=0.2/0.4 ve sabit-P 3 GPa):
P1..P3 olculen yol-fit oranlariyla, P4 yildiz-1 oraniyla olceklenir
(hibritlesme-ayni-mekanizma varsayimi; metinde belirtilir).

Cikti: seviye etiketleri, E21, alpha kanallari (tekli/E'), mu23'ler,
kacak (J*mu2f/mu12/ alpha)^2, blokaj V_B, sayisal-F2 hbar*Gamma.
-> data/t0a_mimari.json
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KOLE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(KOLE, "preprint"))
from pw_cozucu import Solver, STARS, moire_L, KIN, A0

STAR4 = [(2, 1), (-3, 2), (1, -3), (1, 2), (-3, 1), (2, -3)]
t0a = json.load(open(os.path.join(KOLE, "preprint", "data",
                                  "t0a_potansiyel.json"), encoding="utf-8"))
P4S = [p * 1e-3 for p in t0a["fit"]["P_meV"]]      # eV, [P1..P4]
df = json.load(open(os.path.join(KOLE, "results", "data", "derin_fit.json")))
f2a = json.load(open(os.path.join(KOLE, "results", "data", "faz2_analiz.json"),
                     encoding="utf-8"))
TABAN_YOL = df["taban_yol"]["P"]                   # eV [P1..P3]

hbar = 1.054571817e-34; eC = 1.602176634e-19; eps0 = 8.8541878128e-12
vLA = 19.9e3; rho = 7.6e-7; D_eV = 4.5
E2NM = 1.44; A0_NM = 0.0529177; HBARPS = 0.658211951


def pmap_of(P4list):
    m = {}
    for trip, Pv in zip([STARS[1], STARS[2], STARS[3], STAR4], P4list):
        for (mm, nn) in trip:
            m[(mm, nn)] = m.get((mm, nn), 0.0) + Pv
    return m


def coz(P4list, ad):
    b = 4 * math.pi / (math.sqrt(3.0) * moire_L(1.09))
    gf = max(10.0, 2.6 * math.sqrt(0.3 * abs(P4list[0]) / 0.0566 / KIN) / b)
    s = Solver(1.09, ("genel", pmap_of(P4list)), gcut_factor=gf)
    w, labels, D, extent = s.dipoles_extent(nst=8, n=201)
    lv = (np.array(w[:8]) * 1e3).tolist()
    E21 = lv[1] - lv[0]
    labs = [l["label"] for l in labels]
    # kanallar: |e> (ilk E ciftinin uyesi, idx1-2) -> ust seviyeler idx 3..6
    mu12 = math.hypot(float(D[0, 1]), float(D[0, 2]))
    kanallar = []
    i = 3
    for l in labels[2:]:
        deg = l["deg"]
        idxs = list(range(i, i + deg))
        if i >= 8:
            break
        mu2f = math.sqrt(sum(float(D[a, j]) ** 2 for a in (1, 2)
                             for j in idxs if j < 8) / 2.0)
        alpha = (l["E_meV"] - lv[1]) - E21
        kanallar.append({"hedef": l["label"], "E_meV": l["E_meV"],
                         "alpha_meV": alpha, "mu2f_a0e": mu2f,
                         "mu2f_over_mu12": mu2f / mu12})
        i += deg
    # J, kacak, blokaj (en guclu dipolle bagli kanal)
    J = E2NM * 1e3 * (mu12 * A0_NM) ** 2 / (5.0 * (moire_L(1.09) / 10) ** 3)
    en_iyi = max(kanallar, key=lambda k: (k["mu2f_over_mu12"] /
                                          max(abs(k["alpha_meV"]), 0.3)))
    kacaklar = {k["hedef"] + "@%.1f" % k["E_meV"]:
                (J * k["mu2f_over_mu12"] / abs(k["alpha_meV"])) ** 2
                for k in kanallar if abs(k["alpha_meV"]) > 0.05}
    VB = 4 * (J * en_iyi["mu2f_over_mu12"]) ** 2 / abs(en_iyi["alpha_meV"])
    # sayisal form faktor
    q0A = E21 * 1e-3 * eC / hbar / vLA * 1e-10
    w2, lb2, X, Y, psi, dA, c0 = s.real_space_states(nst=2, n=201)
    tot = 0.0
    for a in range(24):
        ph = math.pi * a / 24
        q = q0A * np.array([math.cos(ph), math.sin(ph)])
        tot += abs(np.sum(psi[..., 0].conj() *
                          np.exp(1j * (q[0] * X + q[1] * Y)) * psi[..., 1]) * dA) ** 2
    F2 = tot / 24
    q0 = q0A * 1e10
    G1 = (D_eV * eC) ** 2 * q0 ** 2 / (2 * rho * hbar * vLA ** 2) * F2
    hG = G1 * hbar / eC * 1e6
    rec = {"ad": ad, "P_meV": [p * 1e3 for p in P4list],
           "levels_meV": lv, "labels": labs, "E21_meV": E21,
           "mu12_a0e": mu12, "J_meV": J, "kanallar": kanallar,
           "kacaklar": kacaklar, "V_B_meV": VB,
           "t_CZ_ps": math.pi * HBARPS / VB if VB > 0 else None,
           "hbarG1ph_ueV": hG, "tau1ph_ns": 1e9 / G1,
           "lambda_um": 1239.84 / E21}
    print("%s: E21=%.1f  etiket=%s" % (ad, E21, labs[:5]))
    for k in kanallar:
        print("    -> %s @%.1f: alpha=%+.1f mu2f=%.1f (oran %.2f)"
              % (k["hedef"], k["E_meV"], k["alpha_meV"], k["mu2f_a0e"],
                 k["mu2f_over_mu12"]))
    print("    J=%.2f; kacaklar=%s; V_B=%.3f meV (t_CZ=%.0f ps); hbarG=%.3g ueV tau=%.3g ns"
          % (J, {kk: "%.1e" % vv for kk, vv in kacaklar.items()}, VB,
             rec["t_CZ_ps"], hG, 1e9 / G1))
    return rec


out = {"varyantlar": []}
out["varyantlar"].append(coz(P4S, "4y ortam"))

# rijit P1 oranlari (yol fitlerinden, yildiz-basina)
for dz in (0.2, 0.4):
    f = df["kampanyalar"]["P1_%s" % dz]["fit"]["P"]      # eV
    r = [f[i] / TABAN_YOL[i] for i in range(3)]
    P = [P4S[0] * r[0], P4S[1] * r[1], P4S[2] * r[2], P4S[3] * r[0]]
    out["varyantlar"].append(coz(P, "4y P1 dz=%.1f" % dz))

# sabit-P 3 GPa
f = f2a["P1B"]["3.0"]["fit"]["P"]                        # meV!
r = [f[i] * 1e-3 / TABAN_YOL[i] for i in range(3)]
P = [P4S[0] * r[0], P4S[1] * r[1], P4S[2] * r[2], P4S[3] * r[0]]
out["varyantlar"].append(coz(P, "4y P1b 3GPa"))

json.dump(out, open(os.path.join(HERE, "data", "t0a_mimari.json"), "w",
                    encoding="utf-8"), indent=1, ensure_ascii=False)
print("-> data/t0a_mimari.json")

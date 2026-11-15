# -*- coding: utf-8 -*-
"""
TEK-FONON emisyon hizlari, E -> A1, T=0, tum dogrusal kanallar.
Gamma = q*/(2 pi hbar^2 v_g) int dphi |M|^2 ,  |M|^2 = hbar/(4 rho1 omega) |V|^2  (her s/a dali; m_mod = 2 rho1)
Kanallar: LA/TA x {DP, PZ, gauge, registry(moire-umklapp)} x {s, a}; ZO' (nefes) dogrusal dE/dz;
ZA dogrusal (simetri), ZO, TO/LO, Frohlich, K-vadiler-arasi: enerji/simetri kontrolu.
Cikti: data/kanal_1f.json
"""
import json, math, os, sys, time, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ortak import *
from kuplaj import Tepe
import fonon as fn

NPHI = 96
PHI = np.linspace(0, 2 * np.pi, NPHI, endpoint=False)


def oran(tepe, E21, dal, pol, br, ter, regsign=1.0):
    """Gamma [1/s], E-dubleti ortalamasi."""
    om = E21 / hbar
    q = float(dal.q_of_w(om))
    if not np.isfinite(q):
        return 0.0, None
    vg = float(dal.vg(q)); qA = q * 1e-10
    G = 0.0
    for i in (1, 2):
        V = tepe.V(0, i, qA, PHI, pol, br, ter, regsign) * eC * 1e10      # J/m
        M2 = hbar / (4 * RHO1 * om) * np.abs(V) ** 2
        G += 0.5 * q / (2 * np.pi * hbar ** 2 * vg) * np.mean(M2) * 2 * np.pi
    return G, qA


def main():
    dpp, _ = dp_parametreleri()
    D0 = abs(dpp["D_eV"])
    print("DFT D = %.3f eV" % dpp["D_eV"])
    beta = json.load(open(os.path.join(DATA, "beta.json")))["beta"] if os.path.exists(os.path.join(DATA, "beta.json")) else 2.0
    out = {"D_DFT_eV": dpp["D_eV"], "beta": beta, "noktalar": {}}
    for nokta in ("ortam_1.09", "3GPa_1.09", "5GPa_1.20"):
        t0 = time.time()
        k, FT, GT = hazirla(nokta)
        E21 = (k.E[1] - k.E[0]) * eC
        d_int = ZAA[nokta]
        rec = {"E21_meV": E21 / eC * 1e3, "d_int_A": d_int, "kenar": float(k.kenar_agirligi()[1])}
        # log-egim n_eff
        qs = np.linspace(0.15, 1.2, 43)
        F2 = np.array([np.mean(0.5 * (np.abs(FT(0, 1, q * np.cos(PHI), q * np.sin(PHI))) ** 2 +
                                      np.abs(FT(0, 2, q * np.cos(PHI), q * np.sin(PHI))) ** 2)) for q in qs])
        rec["F2_tablo"] = {"q_invA": qs.tolist(), "F2": F2.tolist(),
                           "n_eff": (-np.gradient(np.log(F2), np.log(qs))).tolist()}
        tablo = {}
        disp = {"lin": fn.dogrusal, "sin": fn.sinus}
        for (vL, vT), dn, D, e11, eps, dw in itertools.product(
                [(21.0e3, 13.0e3), (18.86e3, 11.87e3)], ("lin", "sin"), (D0, 4.5),
                (1.38e-10, 2.91e-10), (5.10, 2.45), (1.0, 0.0)):
            par = {"D": D, "e11": e11, "eps": eps, "dw": dw, "d_int": d_int, "mstar": 0.77, "beta": beta}
            tp = Tepe(k, FT, par, GT)
            LA = disp[dn]("LA", vL); TA = disp[dn]("TA", vT)
            dal = {("L", "s"): LA, ("L", "a"): gapli(LA, fn.W_SH), ("T", "s"): TA, ("T", "a"): gapli(TA, fn.W_SH)}
            key = "vL%.2f_%s_D%.2f_e%.2f_eps%.2f_dw%.0f" % (vL / 1e3, dn, D, e11 * 1e10, eps, dw)
            r = {}
            for pol, br in itertools.product("LT", "sa"):
                for ter in ("dp", "pz", "g", "reg"):
                    if ter == "dp" and pol == "T":
                        continue
                    if ter == "reg" and br == "s":
                        continue
                    G, qA = oran(tp, E21, dal[(pol, br)], pol, br, (ter,))
                    r["%s%s_%s" % (pol, br, ter)] = G
                Gp, qA = oran(tp, E21, dal[(pol, br)], pol, br, ("dp", "pz", "g", "reg"), +1.0)
                Gm, _ = oran(tp, E21, dal[(pol, br)], pol, br, ("dp", "pz", "g", "reg"), -1.0)
                r["%s%s_TOPLAM+" % (pol, br)] = Gp; r["%s%s_TOPLAM-" % (pol, br)] = Gm
                r["q_%s%s" % (pol, br)] = qA
            r["TOPLAM_max"] = sum(max(r["%s%s_TOPLAM+" % (p, b)], r["%s%s_TOPLAM-" % (p, b)]) for p in "LT" for b in "sa")
            r["TOPLAM_min"] = sum(min(r["%s%s_TOPLAM+" % (p, b)], r["%s%s_TOPLAM-" % (p, b)]) for p in "LT" for b in "sa")
            tablo[key] = r
        rec["tablo"] = tablo
        # Kol E tekrar (D=4.5, v=19.9 lin, tek dal rho1, sadece DP): s+a dw=1 ile esdeger
        par = {"D": 4.5, "e11": 0.0, "eps": 1.0, "dw": 1.0, "d_int": d_int, "mstar": 0.77, "beta": 0.0}
        tp = Tepe(k, FT, par, GT); L19 = fn.dogrusal("LA", 19.9e3)
        Gs, _ = oran(tp, E21, L19, "L", "s", ("dp",)); Ga, _ = oran(tp, E21, L19, "L", "a", ("dp",))
        rec["KolE_tekrar_T1_ps"] = 1e12 / (Gs + Ga)
        # --- ZO' (nefes) dogrusal: V = E'(r) * dz ;  dz = -2a_a ; m_mod=2rho1 -> |M|^2 = hbar/(4 rho1 w) * 4 E'^2 |F|^2
        Ez = Ez_turev(nokta)
        E1_0 = (Ez["AA"]["E1"] + 2 * Ez["AB"]["E1"]) / 3.0      # duz ortalama (AA + AB + BA)/3 kaba
        E1_1 = (Ez["AA"]["E1"] - Ez["AB"]["E1"]) / 4.5           # yildiz-1 genligi (AA:+3, AB/BA:-1.5)
        rec["Ez"] = Ez
        zop = fn.za_dal(fn.W_B)
        om = E21 / hbar; q = float(zop.q_of_w(om))
        if np.isfinite(q):
            qA = q * 1e-10; vg = float(zop.vg(q)); G = 0.0
            for i in (1, 2):
                kx, ky = qA * np.cos(PHI), qA * np.sin(PHI)
                V = E1_0 * FT(0, i, kx, ky)
                for Gv in k.Glist[:3]:
                    V = V + 0.5 * E1_1 * (FT(0, i, kx + Gv[0], ky + Gv[1]) + FT(0, i, kx - Gv[0], ky - Gv[1]))
                V = V * eC * 1e10 * 2.0
                M2 = hbar / (4 * RHO1 * om) * np.abs(V) ** 2
                G += 0.5 * q / (2 * np.pi * hbar ** 2 * vg) * np.mean(M2) * 2 * np.pi
            rec["ZOp_nefes_1f"] = {"q_invA": qA, "Gamma": G, "T1_s": 1 / G if G > 0 else None,
                                   "E1_0": E1_0, "E1_1": E1_1}
        else:
            rec["ZOp_nefes_1f"] = {"q_invA": None, "Gamma": 0.0,
                                   "neden": "E21=%.1f meV > ZO' dal tepesi %.1f meV" % (E21 / eC * 1e3, zop.w(fn.Q_M) / meV)}
        za = fn.za_dal()
        rec["enerji_kontrol"] = {"ZA_max_meV": float(za.w(fn.Q_M) / meV), "ZOp_max_meV": float(zop.w(fn.Q_M) / meV),
                                 "ZO_min_meV": 77.0, "TO_meV": 169.5, "LO_meV": 199.6,
                                 "K_dallari_meV": {"ZA": 42, "ZO": 73, "TA": 110.6, "LA": 131.5, "LO": 158.4}}
        out["noktalar"][nokta] = rec
        print("%s E21=%.2f  KolE tekrar %.0f ps  [%.0fs]" % (nokta, rec["E21_meV"], rec["KolE_tekrar_T1_ps"], time.time() - t0), flush=True)
        c = tablo["vL21.00_lin_D%.2f_e1.38_eps5.10_dw1" % D0]
        for kk, vv in c.items():
            if not kk.startswith("q_"):
                print("   %-14s G=%.3e  T1=%s" % (kk, vv, ("%.3g ns" % (1e9 / vv)) if vv > 0 else "-"))
        print("   ZO':", rec["ZOp_nefes_1f"])
    json.dump(out, open(os.path.join(DATA, "kanal_1f.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

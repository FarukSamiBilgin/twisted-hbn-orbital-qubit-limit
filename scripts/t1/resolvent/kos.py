# -*- coding: utf-8 -*-
"""Uretim surucusu: python kos.py <nokta> <vkombo 0/1/2> <Ne> <nphi> <h> <eta_meV> <lams: LT|L> [sos]
Cikti: data/run_<etiket>.json"""
import json, math, os, sys, time
import numpy as np
from iki_fonon_hakem import (Hesap, noktalar, birlestir, tek_birlestir, VLA_BAND, VTA_BAND,
                             GA_MERKEZ, GA_UST, D_BAND, HBAR_eVs, ETA)
from kuyu import _j

HERE = os.path.dirname(os.path.abspath(__file__))


def dogrulayici_noktalari():
    """Dogrulayici fd_cozucu.katsayilar(capa, xi) tarifinin KENDI yeniden yazimi (dosyalar sadece okunur)."""
    t0a = _j("preprint", "data", "t0a_potansiyel.json")
    P4T = [p * 1e-3 for p in t0a["fit"]["P_meV"]]
    df = _j("results", "data", "derin_fit.json"); f2a = _j("results", "data", "faz2_analiz.json")
    TAB3 = df["taban_yol"]["P"]
    def kat(dz, xi):
        rA = [df["kampanyalar"]["P1_%s" % dz]["fit"]["P"][i] / TAB3[i] for i in range(3)]
        rC = [f2a["P1C"][str(dz)]["fit"]["P"][i] * 1e-3 / TAB3[i] for i in range(3)]
        r = [(1 - xi) * rA[i] + xi * rC[i] for i in range(3)]
        return [P4T[0] * r[0], P4T[1] * r[1], P4T[2] * r[2], P4T[3] * r[0]]
    return {"dz04_xi1.00_1.20": (kat(0.4, 1.0), 1.20), "dz04_xi0.75_1.09": (kat(0.4, 0.75), 1.09)}


def sos_blk(H, vby, Ne, nst, eta=ETA):
    """Kol E tipi sonlu ara-durum toplami (yalniz LA deformasyon), kiyas icin."""
    H.k.coz(nst=nst)
    psi, w = H.k.psi, H.k.w
    E21 = H.E21
    eps = E21 * (np.arange(Ne) + 0.5) / Ne
    g = psi[H.ig]; kay = []
    v = vby["LA"]
    for ie in H.ie:
        Ee = w[ie]
        A = {}; B = {}
        for j in range(Ne):
            A[j] = np.array([H.bralar("LA", eps[j], v, psi[m])["d"] @ psi[ie] for m in range(nst)])  # [m, phi]
            B[j] = H.bralar("LA", eps[j], v, g)["d"] @ psi.T                                        # [phi, m]
        for i in range(Ne // 2):
            i2 = Ne - 1 - i
            d1 = 1.0 / (Ee - w[:nst] - eps[i] + 1j * eta)
            d2 = 1.0 / (Ee - w[:nst] - eps[i2] + 1j * eta)
            T = B[i2] @ (d1[:, None] * A[i]) + (B[i] @ (d2[:, None] * A[i2])).T
            kay.append((ie, {"i": i, "eps": (eps[i], eps[i2]), "LL": {("d", "d"): T}}))
    return {"kayit": kay, "eps": eps, "de": E21 / Ne, "Ne": Ne}


if __name__ == "__main__":
    ad, vk, Ne, nphi, h, eta = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), \
        float(sys.argv[5]), float(sys.argv[6]) * 1e-3
    lams = ("LA", "TA") if sys.argv[7] == "LT" else ("LA",)
    sos = len(sys.argv) > 8 and sys.argv[8] == "sos"
    P = noktalar(); P.update(dogrulayici_noktalari())
    P4, th = P[ad]
    vby = {"LA": VLA_BAND[vk], "TA": VTA_BAND[vk]}
    etiket = "%s_v%d_Ne%d_nphi%d_h%.2f_eta%.2f_%s%s" % (ad, vk, Ne, nphi, h, eta * 1e3, sys.argv[7], "_sos" if sos else "")
    t0 = time.time()
    H = Hesap(P4, th, h=h, nphi=nphi)
    out = {"nokta": ad, "theta": th, "P4_meV": [p * 1e3 for p in P4], "E21_meV": H.E21 * 1e3,
           "vLA": vby["LA"], "vTA": vby["TA"], "Ne": Ne, "nphi": nphi, "h": H.k.h, "eta_meV": eta * 1e3,
           "Npt": H.Npt, "tek": {}, "iki": {}}
    tf = H.tek_fonon(vby)
    for D in D_BAND:
        for gA in (0.0, GA_MERKEZ, GA_UST):
            for off in (0, 1):
                key = "D%.2f_gA%.3f_off%d" % (D, gA, off)
                gl = tek_birlestir(tf, "LA", D, gA, off); gt = tek_birlestir(tf, "TA", D, gA, off)
                out["tek"][key] = {"G_LA": gl, "G_TA": gt}
    blk = H.iki_fonon(vby, Ne=Ne, eta=eta, lams=lams, log=False)
    for D in D_BAND:
        for gA in (0.0, GA_MERKEZ, GA_UST):
            for off in (0, 1):
                key = "D%.2f_gA%.3f_off%d" % (D, gA, off)
                r = {"G_LL": birlestir(blk, vby, D, gA, off, nphi, "LL")}
                if "TA" in lams:
                    r["G_LT"] = birlestir(blk, vby, D, gA, off, nphi, "LT")
                    r["G_TT"] = birlestir(blk, vby, D, gA, off, nphi, "TT")
                out["iki"][key] = r
    # enerji-dagilimi (LL, D=4.5, gA=0): hangi e bolgesi tasiyor
    dist = []
    for ie, rec in blk["kayit"]:
        M = rec["LL"][("d", "d")]
        dist.append((ie, rec["eps"][0] * 1e3, float(np.sum(np.abs(M) ** 2))))
    out["LL_dagilim"] = dist
    if sos:
        out["sos"] = {}
        for nst in (10, 20, 40, 80):
            b = sos_blk(H, vby, Ne, nst, eta)
            out["sos"]["nst%d" % nst] = {"G_LL_D4.50": birlestir(b, vby, 4.5, 0.0, 0, nphi, "LL")}
    out["sure_s"] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, "data", "run_%s.json" % etiket), "w"), indent=1)
    g = out["iki"]["D4.50_gA0.000_off0"]["G_LL"]
    print("%s E21=%.2f T1_LL(D4.5)=%.4g ns  T1_1f=%.4g ns  %.0fs" % (etiket, H.E21 * 1e3, 1e9 / g,
          1e9 / out["tek"]["D4.50_gA0.000_off0"]["G_LA"], out["sure_s"]))
    if sos:
        print("  sos:", {k: 1e9 / v["G_LL_D4.50"] for k, v in out["sos"].items()})

# -*- coding: utf-8 -*-
"""Cozucu kontrolleri: E21, yakinsama (gcut, h), FFT-F vs dogrudan, Kol E LA tekrari."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moire import Kuyu, FTablo, F_dogrudan, noktalar

hbar = 1.054571817e-34; eC = 1.602176634e-19
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
res = {}
for ad, (P4, th) in noktalar().items():
    rec = {"P4_meV": [p * 1e3 for p in P4], "theta": th}
    for gb in (12.0, 16.0, 20.0):
        t = time.time()
        k = Kuyu(P4, th, gcut_b=gb); w = k.coz(10)
        lv = (w - w[0]) * 1e3
        k.gercek_uzay(h=0.6)
        FT = FTablo(k, [(0, 1), (0, 2)], kmax=1.0)
        qs = np.array([0.2, 0.35, 0.5, 0.65, 0.8])
        F2 = []
        for q in qs:
            ph = np.linspace(0, 2 * np.pi, 48, endpoint=False)
            v = 0.5 * (np.abs(FT(0, 1, q * np.cos(ph), q * np.sin(ph))) ** 2
                       + np.abs(FT(0, 2, q * np.cos(ph), q * np.sin(ph))) ** 2)
            F2.append(float(v.mean()))
        # dogrudan kontrol
        fd = F_dogrudan(k, 0, 1, 0.5 * 0.6, 0.5 * 0.8)
        ff = FT(0, 1, np.array([0.3]), np.array([0.4]))[0]
        rec["gcut%.0f" % gb] = {"N": k.N, "lv_meV": lv.tolist(), "F2_q": dict(zip(map(str, qs), F2)),
                                "F_fft_vs_dir_relerr": float(abs(ff - fd) / abs(fd)),
                                "kenar": k.kenar_agirligi()[:3].tolist()}
        print("%s gcut=%g N=%d E21=%.3f lv=%s F2=%s fft/dir=%.2e kenar=%.1e [%.0fs]" % (
            ad, gb, k.N, lv[1], np.round(lv[:6], 2), ["%.3e" % f for f in F2],
            abs(ff - fd) / abs(fd), k.kenar_agirligi()[1], time.time() - t), flush=True)
    # h yakinsamasi
    k = Kuyu(P4, th, gcut_b=16.0); k.coz(10); k.gercek_uzay(h=0.45)
    FT = FTablo(k, [(0, 1), (0, 2)], kmax=1.0, pad=1536)
    q = 0.5; ph = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    v = 0.5 * (np.abs(FT(0, 1, q * np.cos(ph), q * np.sin(ph))) ** 2 + np.abs(FT(0, 2, q * np.cos(ph), q * np.sin(ph))) ** 2)
    rec["h045_F2_q0.5"] = float(v.mean())
    print("   h=0.45: F2(0.5)=%.3e" % v.mean())
    # Kol E LA tekrari: D=4.5, v=19.9, rho=7.6e-7, q0=E21/hbar v
    k = Kuyu(P4, th, gcut_b=16.0); w = k.coz(10); k.gercek_uzay(h=0.6)
    FT = FTablo(k, [(0, 1), (0, 2)], kmax=1.0)
    E21 = (w[1] - w[0]); vLA = 19.9e3; rho = 7.6e-7; D = 4.5 * eC
    q0 = E21 * eC / hbar / vLA
    ph = np.linspace(0, 2 * np.pi, 96, endpoint=False); qA = q0 * 1e-10
    F2 = float(np.mean(0.5 * (np.abs(FT(0, 1, qA * np.cos(ph), qA * np.sin(ph))) ** 2
                              + np.abs(FT(0, 2, qA * np.cos(ph), qA * np.sin(ph))) ** 2)))
    G = D ** 2 * q0 ** 2 / (2 * rho * hbar * vLA ** 2) * F2
    rec["KolE_LA_tekrar"] = {"E21_meV": E21 * 1e3, "q0_invA": qA, "F2": F2, "hG_ueV": G * hbar / eC * 1e6,
                             "T1_ps": 1e12 / G}
    print("   Kol E LA tekrari: E21=%.2f q0=%.3f F2=%.3e hG=%.3g ueV T1=%.4g ps" % (
        E21 * 1e3, qA, F2, G * hbar / eC * 1e6, 1e12 / G))
    res[ad] = rec
json.dump(res, open(os.path.join(OUT, "test_cozucu.json"), "w"), indent=1)

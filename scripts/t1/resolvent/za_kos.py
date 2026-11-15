# -*- coding: utf-8 -*-
"""ZA+ZA uretim: 3 ana nokta + 2 dogrulayici noktasi. Cikti data/za_za.json"""
import json, math, os, time
import numpy as np
from iki_fonon_hakem import Hesap, noktalar, VLA_BAND, VTA_BAND, D_BAND, GA_MERKEZ, za_disp, Q_M
from za_tablo import FTablo, za_za_tab
from kos import dogrulayici_noktalari

HERE = os.path.dirname(os.path.abspath(__file__))
P = noktalar(); P.update(dogrulayici_noktalari())
SIRA = ["ortam_0GPa_1.09", "calisma_3GPa_1.09", "5GPa_1.20_uzatma", "dz04_xi0.75_1.09", "dz04_xi1.00_1.20"]
MODELLER = [("sin2", 1.0), ("kq2", 1.29), ("kq2", 0.86)]
out = {}
for ad in SIRA:
    P4, th = P[ad]
    t0 = time.time()
    H = Hesap(P4, th, h=0.75, nphi=24)
    tab = {ie: FTablo(H, ie) for ie in H.ie}
    rec = {"E21_meV": H.E21 * 1e3, "sonuc": {}}
    for model, kap in MODELLER:
        mk = "%s_k%.2f" % (model, kap)
        for vk in range(3):
            vL, vT = VLA_BAND[vk], VTA_BAND[vk]
            for ekran, kw in (("yok", {}), ("statik", {}),
                              ("dinamik", {"gamLA": 0.5e-3, "kutup_haric": 0.1}),
                              ("dinamik_kutuplu", {"gamLA": 0.5e-3})):
                ek = "dinamik" if ekran.startswith("dinamik") else ekran
                for D in D_BAND:
                    for gA in (0.0, GA_MERKEZ):
                        if gA and not (ekran == "dinamik" and vk == 1):
                            continue
                        G, info = za_za_tab(H, tab, model, kap, D, vL, vT, ekran=ek, gA=gA, **kw)
                        key = "%s|v%d|%s|D%.2f|gA%.3f" % (mk, vk, ekran, D, gA)
                        rec["sonuc"][key] = {"G": G, "T1_ns": 1e9 / G if G > 0 else None}
                        if D == 4.5 and gA == 0 and vk == 1 and ekran == "dinamik":
                            rec.setdefault("bilgi", {})[mk] = info
    rec["sure_s"] = time.time() - t0
    out[ad] = rec
    k = "sin2_k1.00|v1|dinamik|D4.50|gA0.000"
    print("%s E21=%.2f  T1_ZZ(sin2,din,D4.5)=%s ns  %.0fs" % (ad, H.E21 * 1e3, rec["sonuc"][k]["T1_ns"], rec["sure_s"]), flush=True)
    json.dump(out, open(os.path.join(HERE, "data", "za_za.json"), "w"), indent=1)
print("BITTI")

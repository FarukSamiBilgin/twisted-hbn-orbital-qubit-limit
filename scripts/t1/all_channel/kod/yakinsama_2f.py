# -*- coding: utf-8 -*-
"""Ardisik 2-fonon yakinsama: nst, nE, nphi, eta (3 GPa, DP-LA + PZ, registry acik, v=21 lin)."""
import sys, json, os, numpy as np
sys.path.insert(0, '.')
from ortak import *
import fonon as fn, kanal_2f as K2
out = {}
for nst in (10, 14):
    k, FT, _ = hazirla("3GPa_1.09", gauge=False, nst=nst)
    E21 = (k.E[1]-k.E[0])*eC
    par = {"D": 5.945, "e11": 1.38e-10, "eps": 5.1, "dw": 1.0, "d_int": ZAA["3GPa_1.09"], "mstar": 0.77, "beta": 0.0}
    for nE, nphi, eta in ((40, 24, 0.3), (60, 32, 0.3), (40, 24, 0.1)):
        K2.ETA = eta*1e-3*eC
        r = K2.ardisik(k, FT, par, E21, K2.dallar_ip(21e3, 13e3, "lin"), nE=nE, nphi=nphi)
        key = "nst%d_nE%d_nphi%d_eta%.1f" % (nst, nE, nphi, eta)
        out[key] = 1e12/r["TOPLAM"]; print(key, "T1=%.2f ps" % out[key], flush=True)
json.dump(out, open(os.path.join(DATA, "yakinsama_2f.json"), "w"), indent=1)

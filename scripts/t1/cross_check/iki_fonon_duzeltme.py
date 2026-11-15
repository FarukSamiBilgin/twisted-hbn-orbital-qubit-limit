# -*- coding: utf-8 -*-
"""
Iki-LA-fonon duzeltmesi: 1/2 ozdes-bozon carpani (artik gamma2 icinde) +
ara-durum toplaminin yakinsamasi (NST = 10 / 20 / 40).

Merkez parametreler D=4.5 eV, v=18.86 km/s; E-dubletinin iki uyesi ortalamasi.
Kiyas: hakemin tam-resolvent LA+LA degerleri (MIMARI/t1_hakem_hukmu.md):
  48.2 meV: SOS10 0.0651, SOS40 0.0511, tam 0.0504 ns (onun makinesinde)
  88.4 meV: SOS10 0.867, 20 0.514, 40 0.241, 80 0.138, tam 0.101 ns
Not: bu cozucunun ust durumlari Gamma-noktasi Bloch durumlari (pencere
+-0.55 L) — kutu-surekliligini hakemin seyrek-LU resolventi kadar temiz
temsil etmez; burada yalniz YON ve buyukluk teyidi aranir.
Cikti: data/iki_fonon_duzeltme.json
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import iki_fonon_bagimsiz as K

out = {}
for ad, (P, th) in K.NOKTALAR.items():
    rec = {}
    for nst in (10, 20, 40):
        E, X, Y, psi, dA = K.hazirla(P, th, nst=nst)
        g = 0.5 * (K.gamma2(E, X, Y, psi, dA, 4.5, 18.86e3, 1)
                   + K.gamma2(E, X, Y, psi, dA, 4.5, 18.86e3, 2))
        E21 = (E[1] - E[0]) / K.eC * 1e3
        rec["NST%d" % nst] = 1e9 / g
        print("%s (E21=%.1f) NST=%d: T1_LA+LA = %.4g ns" % (ad, E21, nst, 1e9 / g),
              flush=True)
    rec["E21_meV"] = E21
    out[ad] = rec
    json.dump(out, open(os.path.join(HERE, "data", "iki_fonon_duzeltme.json"), "w"),
              indent=1)
print("-> data/iki_fonon_duzeltme.json")

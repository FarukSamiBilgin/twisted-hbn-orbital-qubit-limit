# -*- coding: utf-8 -*-
"""
Dipol konvansiyonu kontrolu (dogrulama ekibinin uyarisi uzerine).

Soru: qubit = tek E-bileseni (gerinim-yarilmis Ex ya da kiral E+) ise
flip-flop J hangi dipolle olceklenir?

Hesap: d_i = (<A1|x|e_i>, <A1|y|e_i>) kompleks vektorleri (i = dublet uyeleri),
M = [[dx1,dx2],[dy1,dy2]] matrisinin SVD'si -> tekil degerler s1,s2.
C3 simetrisinde s1 = s2 = mu_tot/sqrt2 olmali: dubletin HER dogrusal/kiral
bileseni ayni mu_el ile baglanir; "toplam" mu_tot = sqrt(s1^2+s2^2) yalniz
dublet-toplami spektroskopik buyukluktur. Kiral kontrol: |d·eps_-| / |d·eps_+|.
Noktalar: ortam 4-yildiz (1.09) ve 3 GPa zinciri (1.09).
"""
import json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KOLE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(KOLE, "preprint"))
from pw_cozucu import Solver, STARS, moire_L, KIN

STAR4 = [(2, 1), (-3, 2), (1, -3), (1, 2), (-3, 1), (2, -3)]
t0a = json.load(open(os.path.join(KOLE, "preprint", "data", "t0a_potansiyel.json"),
                     encoding="utf-8"))
P4TAB = [p * 1e-3 for p in t0a["fit"]["P_meV"]]
df = json.load(open(os.path.join(KOLE, "results", "data", "derin_fit.json")))
f2a = json.load(open(os.path.join(KOLE, "results", "data", "faz2_analiz.json"),
                     encoding="utf-8"))
TAB3 = df["taban_yol"]["P"]


def pmap_of(P4):
    m = {}
    for trip, Pv in zip([STARS[1], STARS[2], STARS[3], STAR4], P4):
        for t in trip:
            m[t] = m.get(t, 0.0) + Pv
    return m


def analiz(ad, P4, theta):
    b = 4 * math.pi / (math.sqrt(3.0) * moire_L(theta))
    gf = max(10.0, 2.6 * math.sqrt(0.3 * abs(P4[0]) / 0.0566 / KIN) / b)
    s = Solver(theta, ("genel", pmap_of(P4)), gcut_factor=gf)
    w, labels, X, Y, psi, dA, c0 = s.real_space_states(nst=3, n=241)
    p0 = psi[..., 0]
    d = []
    for i in (1, 2):
        cx = np.sum(p0.conj() * (X - c0[0]) * psi[..., i]) * dA
        cy = np.sum(p0.conj() * (Y - c0[1]) * psi[..., i]) * dA
        d.append([cx, cy])
    M = np.array([[d[0][0], d[1][0]], [d[0][1], d[1][1]]])  # [kartezyen, durum]
    sv = np.linalg.svd(M, compute_uv=False)
    mu_tot = math.sqrt(float(np.sum(np.abs(M) ** 2)))
    # kiral: en iyi E+ kombinasyonu icin d·eps-+ oranlari
    # (SVD sag-tekil vektorleri dublet-ici en iyi baz; kiral baz = (e1 -+ i e2)/sqrt2)
    epl = (M[:, 0] + 1j * M[:, 1]) / math.sqrt(2.0)   # |E+> adayi dipol vektoru
    emi = (M[:, 0] - 1j * M[:, 1]) / math.sqrt(2.0)
    def pol(v):
        ep = np.array([1, 1j]) / math.sqrt(2.0)
        em = np.array([1, -1j]) / math.sqrt(2.0)
        return abs(np.vdot(ep, v)), abs(np.vdot(em, v)), np.linalg.norm(v)
    print("%s: mu_tot=%.2f  SVD tekil=(%.2f, %.2f)  mu_tot/sqrt2=%.2f" %
          (ad, mu_tot, sv[0], sv[1], mu_tot / math.sqrt(2.0)))
    for nm, v in (("E+aday", epl), ("E-aday", emi)):
        a, bb, n = pol(v)
        print("   %s: |d|=%.2f  |d.eps+|=%.2f |d.eps-|=%.2f" % (nm, n, a, bb))
    return {"mu_tot": mu_tot, "sv": sv.tolist()}


out = {}
out["ortam_4y_1.09"] = analiz("ortam 4-yildiz 1.09", P4TAB, 1.09)
r3 = [p * 1e-3 / TAB3[i] for i, p in enumerate(f2a["P1B"]["3.0"]["fit"]["P"])]
P3g = [P4TAB[0] * r3[0], P4TAB[1] * r3[1], P4TAB[2] * r3[2], P4TAB[3] * r3[0]]
out["3GPa_1.09"] = analiz("3 GPa 1.09", P3g, 1.09)
json.dump(out, open(os.path.join(HERE, "data", "dipol_konvansiyon.json"), "w"),
          indent=1)
print("-> data/dipol_konvansiyon.json")

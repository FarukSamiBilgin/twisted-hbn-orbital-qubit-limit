# -*- coding: utf-8 -*-
"""
IKI-LA-FONON emisyonu — BAGIMSIZ hesap (dogrulama ekibinin betiklerinden
bagimsiz; kendi 4-yildiz cozucum + tam 2. mertebe altin kurali).

Gamma_2 = (1/2) (2pi/hbar) (2pi)^-4 (hbar v)^-1 \\int_0^{q0} dq1 q1 q2*
          <|T|^2>_{phi1,phi2}

DUZELTME (2026-11-08, hakem T1 kampanyasi): (i) ozdes bozonlar (ayni LA
dali) icin 1/2 carpani ilk surumde EKSIKTI (hiz 2x fazlaydi) — eklendi;
(ii) ara-durum toplami NST=10'da kesiliyordu (48 meV'de ~1.3x, 88 meV'de
~8.6x eksik sayim, hakemin tam-resolvent hesabina gore) — NST artik
parametre; yakinsama taramasi: iki_fonon_duzeltme.py.
T = sum_m [ G(q2)G(q1) F_gm(q2) F_me(q1) / (dEm - hbar v q1 + i eta)
          + G(q1)G(q2) F_gm(q1) F_me(q2) / (dEm - hbar v q2 + i eta) ]
G(q)^2 = D^2 hbar q / (2 rho v);  dEm = E_e - E_m;  eta = 0.3 meV.

- |e>: E-dubletinin iki uyesi uzerinden ortalama.
- Ara durumlar: ilk NST=10 seviye (g ve e dahil; m=g'nin q1->q0 polu
  ardisik gercek emisyondur, eta ile duzgunlestirilir, kenar katkisi
  q2*->0 vertexiyle zaten soner).
- Parametre bandi: D in {3.66, 4.5} eV; v in {18.86, 21.0} km/s.
- Kontroller: (i) ayni F-makinesiyle tek-fonon Gamma_1 (onceki sayilarla
  kiyas), (ii) S*Gamma_1(q0/2) kisayolu, (iii) q-grid yakinsamasi.

Noktalar: mimari 3 GPa (P1b, E21~48), 5 GPa 1.2 der (uzatma), kutu
dz04/xi1/1.2 der (E21~88; dogrulama ekibi 0.30-1.94 ns dedi).
Cikti: data/iki_fonon_bagimsiz.json
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

hbar = 1.054571817e-34; eC = 1.602176634e-19
rho = 7.6e-7
NST = 10
ETA_J = 0.3e-3 * eC


def pmap_of(P4list):
    m = {}
    for trip, Pv in zip([STARS[1], STARS[2], STARS[3], STAR4], P4list):
        for (mm, nn) in trip:
            m[(mm, nn)] = m.get((mm, nn), 0.0) + Pv
    return m


def hazirla(P4list, theta, ngrid=181, nst=NST):
    b = 4 * math.pi / (math.sqrt(3.0) * moire_L(theta))
    gf = max(10.0, 2.6 * math.sqrt(0.3 * abs(P4list[0]) / 0.0566 / KIN) / b)
    s = Solver(theta, ("genel", pmap_of(P4list)), gcut_factor=gf)
    w, labels, X, Y, psi, dA, c0 = s.real_space_states(nst=nst, n=ngrid)
    E = np.array(w[:nst]) * eC               # J
    return E, X, Y, psi, dA


def F_tablosu(X, Y, psi, dA, qgrid_A, nphi):
    """F[a,b,iq,iphi] = <a|e^{iq r}|b>; q [1/A]."""
    nst = psi.shape[-1]
    F = np.zeros((nst, nst, len(qgrid_A), nphi), dtype=complex)
    P = psi.reshape(-1, nst)
    Xf, Yf = X.ravel(), Y.ravel()
    for iq, q in enumerate(qgrid_A):
        for ip in range(nphi):
            phi = 2 * math.pi * ip / nphi
            ph = np.exp(1j * (q * math.cos(phi) * Xf + q * math.sin(phi) * Yf))
            M = (P.conj() * ph[:, None]).T @ P * dA   # [a,b]
            F[:, :, iq, ip] = M
    return F


def gamma2(E, X, Y, psi, dA, D_eV, v, e_idx, nq=28, nphi=8):
    """Iki-LA-fonon emisyon hizi [1/s], |e> = e_idx."""
    D = D_eV * eC
    E21 = E[e_idx] - E[0]
    q0 = E21 / hbar / v                      # 1/m
    q0A = q0 * 1e-10                         # 1/A
    xs = np.linspace(0.03, 0.97, nq)         # q1/q0
    qgrid = np.unique(np.concatenate([xs, 1 - xs])) * q0A
    F = F_tablosu(X, Y, psi, dA, qgrid, nphi)
    qlut = {round(float(q), 12): i for i, q in enumerate(qgrid)}

    def Fv(a, b, qA):
        return F[a, b, qlut[round(float(qA), 12)], :]   # [phi]

    dE = E[e_idx] - E[:len(E)]               # J, [m]
    tot = 0.0
    for x in xs:
        q1 = x * q0; q2 = (1 - x) * q0
        q1A, q2A = q1 * 1e-10, q2 * 1e-10
        G1 = D * math.sqrt(hbar * q1 / (2 * rho * v))
        G2 = D * math.sqrt(hbar * q2 / (2 * rho * v))
        # T[phi1, phi2]
        T = np.zeros((nphi, nphi), dtype=complex)
        for m in range(len(E)):
            d1 = dE[m] - hbar * v * q1 + 1j * ETA_J
            d2 = dE[m] - hbar * v * q2 + 1j * ETA_J
            Fgm_q2 = Fv(0, m, q2A)           # [phi2]
            Fme_q1 = Fv(m, e_idx, q1A)       # [phi1]
            Fgm_q1 = Fv(0, m, q1A)           # [phi1]
            Fme_q2 = Fv(m, e_idx, q2A)       # [phi2]
            T += (G2 * G1 / d1) * np.outer(Fme_q1, Fgm_q2) \
               + (G1 * G2 / d2) * np.outer(Fgm_q1, Fme_q2)
        ort = float(np.mean(np.abs(T) ** 2))  # <|T|^2>_{phi1,phi2}
        tot += (x * q0) * ((1 - x) * q0) * ort
    dq1 = (xs[1] - xs[0]) * q0
    integ = tot * dq1 * (2 * math.pi) ** 2   # phi integralleri = (2pi)^2 * ort
    G2rate = 0.5 * (2 * math.pi / hbar) * integ / (2 * math.pi) ** 4 / (hbar * v)
    return G2rate


def gamma1_kontrol(E, X, Y, psi, dA, D_eV, v, e_idx, nphi=24):
    D = D_eV * eC
    E21 = E[e_idx] - E[0]
    q0 = E21 / hbar / v
    q0A = q0 * 1e-10
    F = F_tablosu(X, Y, psi, dA, np.array([q0A]), nphi)
    F2 = float(np.mean(np.abs(F[0, e_idx, 0, :]) ** 2))
    return D ** 2 * q0 ** 2 / (2 * rho * hbar * v ** 2) * F2, F2


NOKTALAR = {}
r3 = [p * 1e-3 / TAB3[i] for i, p in enumerate(f2a["P1B"]["3.0"]["fit"]["P"])]
NOKTALAR["calisma_3GPa_1.09"] = ([P4TAB[0]*r3[0], P4TAB[1]*r3[1],
                                  P4TAB[2]*r3[2], P4TAB[3]*r3[0]], 1.09)
par = json.load(open(os.path.join(HERE, "data", "pareto.json"), encoding="utf-8"))
sl = par["lnslope_p"]
r5 = [math.exp(np.polyval(sl[i], 5.0)) for i in range(3)]
NOKTALAR["5GPa_1.20_uzatma"] = ([P4TAB[0]*r5[0], P4TAB[1]*r5[1],
                                 P4TAB[2]*r5[2], P4TAB[3]*r5[0]], 1.20)
rc = [p * 1e-3 / TAB3[i] for i, p in enumerate(f2a["P1C"]["0.4"]["fit"]["P"])]
NOKTALAR["kutu_dz04_xi1_1.20"] = ([P4TAB[0]*rc[0], P4TAB[1]*rc[1],
                                   P4TAB[2]*rc[2], P4TAB[3]*rc[0]], 1.20)

def main():
  out = {}
  for ad, (P, th) in NOKTALAR.items():
    E, X, Y, psi, dA = hazirla(P, th)
    E21 = (E[1] - E[0]) / eC * 1e3
    print("== %s: E21=%.1f meV ==" % (ad, E21))
    rec = {"E21_meV": E21, "band": {}}
    for D_eV in (3.66, 4.5):
        for v in (18.86e3, 21.0e3):
            g2a = gamma2(E, X, Y, psi, dA, D_eV, v, 1)
            g2b = gamma2(E, X, Y, psi, dA, D_eV, v, 2)
            g2 = 0.5 * (g2a + g2b)
            g1, F2q0 = gamma1_kontrol(E, X, Y, psi, dA, D_eV, v, 1)
            key = "D%.2f_v%.0f" % (D_eV, v / 1e3)
            rec["band"][key] = {"T1_2f_ns": 1e9 / g2, "T1_1f_ns": 1e9 / g1,
                                "F2_q0": F2q0}
            print("  D=%.2f v=%.1f: T1_2f=%.3g ns  (T1_1f=%.3g ns, F2(q0)=%.2g)"
                  % (D_eV, v / 1e3, 1e9 / g2, 1e9 / g1, F2q0))
    # yakinsama (orta parametre, nq arttir)
    g2c = gamma2(E, X, Y, psi, dA, 4.5, 18.86e3, 1, nq=40, nphi=10)
    rec["yakinsama_nq40_T1ns"] = 1e9 / g2c
    print("  yakinsama nq40/nphi10: T1_2f=%.3g ns" % (1e9 / g2c))
    # (S*Gamma1(q0/2) kisayol blogu birim hatasi tasidigi icin kaldirildi.)
    out[ad] = rec
  json.dump(out, open(os.path.join(HERE, "data", "iki_fonon_bagimsiz.json"), "w",
                      encoding="utf-8"), indent=1, ensure_ascii=False)
  print("-> data/iki_fonon_bagimsiz.json")


if __name__ == "__main__":
    main()

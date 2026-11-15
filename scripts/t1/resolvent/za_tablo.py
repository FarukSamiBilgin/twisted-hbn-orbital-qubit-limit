# -*- coding: utf-8 -*-
"""
ZA+ZA (kuadratik egilme kuplaji) — form faktoru TABLOSU + ince bagil-aci izgarasi.
F(K) = <g|e^{-iK.r}|e>, G_x,y(K) = <g|e^{-iK.r} d_x,y|e> kutupsal tabloda
(K in [0,Kmax], phi_K 0..2pi) -> iki-dogrusal interpolasyon.
Integral: q_h (yuksek-enerjili fonon) x phi_1 x psi=phi_2-phi_1 (psi ince, pi civari yogun
cunku K=|k1+k2| kucuk bolge F'yi tasir).
"""
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from iki_fonon_hakem import (za_disp, q_of, u_amp, HBAR_eVs, KIN, Q_M)


class FTablo:
    def __init__(self, H, ie, Kmax=3.2, NK=321, NphK=96):
        self.Ks = np.linspace(0, Kmax, NK)
        self.phK = 2 * np.pi * np.arange(NphK + 1) / NphK
        g = H.psi[H.ig]; e = H.psi[ie]
        gx = H.Dx @ e; gy = H.Dy @ e
        F = np.zeros((NK, NphK + 1), complex); Gx = F.copy(); Gy = F.copy()
        w0 = np.conj(g) * H.dA
        for j, ph in enumerate(self.phK[:-1]):
            Kv = np.outer(self.Ks, [math.cos(ph), math.sin(ph)])
            for s0 in range(0, NK, 64):
                sl = slice(s0, s0 + 64)
                E = np.exp(-1j * (np.outer(Kv[sl, 0], H.X) + np.outer(Kv[sl, 1], H.Y))) * w0[None, :]
                F[sl, j] = E @ e; Gx[sl, j] = E @ gx; Gy[sl, j] = E @ gy
        F[:, -1] = F[:, 0]; Gx[:, -1] = Gx[:, 0]; Gy[:, -1] = Gy[:, 0]
        self.F, self.Gx, self.Gy = F, Gx, Gy
        mk = lambda A: (RegularGridInterpolator((self.Ks, self.phK), A.real, bounds_error=False, fill_value=0.0),
                        RegularGridInterpolator((self.Ks, self.phK), A.imag, bounds_error=False, fill_value=0.0))
        self.iF, self.iGx, self.iGy = mk(F), mk(Gx), mk(Gy)

    def __call__(self, Kn, phK, which="F"):
        it = {"F": self.iF, "Gx": self.iGx, "Gy": self.iGy}[which]
        pts = np.stack([Kn, np.mod(phK, 2 * np.pi)], -1)
        return it[0](pts) + 1j * it[1](pts)


def za_za_tab(H, tablolar, model, kappa, D, vL, vT, ekran="dinamik", gamLA=0.5e-3,
              gA=0.0, off=0.0, Nk=32, nphi1=24, Npsi=2048, kutup_haric=0.0):
    w, vg = za_disp(model, kappa)
    E21 = H.E21
    wmax = float(w(Q_M))
    if E21 / 2 > wmax:
        return 0.0, {}
    eh_lo, eh_hi = E21 / 2, min(wmax, E21)
    qh_lo, qh_hi = q_of(w, eh_lo), q_of(w, eh_hi)
    qhs = qh_lo + (qh_hi - qh_lo) * (np.arange(Nk) + 0.5) / Nk
    dq = (qh_hi - qh_lo) / Nk
    phi1 = 2 * np.pi * (np.arange(nphi1) + 0.25) / nphi1
    # psi = pi + s ; s yogun: s = pi * t^3 , t in (-1,1)
    t = -1 + 2 * (np.arange(Npsi) + 0.5) / Npsi
    s = np.pi * t ** 3; ds = np.pi * 3 * t ** 2 * (2 / Npsi)
    psi = np.pi + s
    Om = E21 / HBAR_eVs
    rvL2, rmu, rlam = vL ** 2, vT ** 2, vL ** 2 - 2 * vT ** 2
    q0 = E21 / (HBAR_eVs * vL * 1e10)
    pref = (2 * math.pi / HBAR_eVs) * 0.5 * (2 * math.pi) ** -4
    tot = {ie: 0.0 for ie in H.ie}
    dK_hist = np.zeros(60); Kb = np.linspace(0, 1.5, 61)
    P1, PS = np.meshgrid(phi1, psi, indexing="ij")
    P1 = P1.ravel(); PS = PS.ravel()
    W = np.repeat(np.full(nphi1, 2 * np.pi / nphi1), Npsi) * np.tile(ds, nphi1)
    for qh in qhs:
        eh = float(w(qh)); el = E21 - eh
        if el <= 0:
            continue
        ql = q_of(w, el)
        uh, ul = u_amp(eh), u_amp(el)
        dens = 2 * qh * dq * ql / float(vg(ql))
        k1 = np.stack([qh * np.cos(P1), qh * np.sin(P1)], 1)
        k2 = np.stack([ql * np.cos(P1 + PS), ql * np.sin(P1 + PS)], 1)
        Kv = k1 + k2
        Kn = np.linalg.norm(Kv, axis=1); phK = np.arctan2(Kv[:, 1], Kv[:, 0])
        Kh = Kv / np.maximum(Kn, 1e-12)[:, None]
        fkk = -np.sum(k1 * k2, 1)
        fLL = -(np.sum(Kh * k1, 1) * np.sum(Kh * k2, 1))
        if ekran == "yok":
            ukk = fkk.astype(complex)
        elif ekran == "statik":
            ukk = (fkk - (rlam * fkk + 2 * rmu * fLL) / rvL2).astype(complex)
        else:
            K2 = (Kn * 1e10) ** 2
            den = rvL2 * K2 - Om ** 2 - 1j * Om * (gamLA / HBAR_eVs)
            ukk = fkk - K2 * (rlam * fkk + 2 * rmu * fLL) / den
        maske = np.ones_like(Kn)
        if kutup_haric > 0:
            maske[np.abs(Kn - q0) < kutup_haric * q0] = 0.0
        if gA != 0.0:
            c, sn = math.cos(off), math.sin(off)
            R = np.array([[c, sn], [-sn, c]])
            k1l, k2l = k1 @ R.T, k2 @ R.T
            Sxx = -k1l[:, 0] * k2l[:, 0]; Syy = -k1l[:, 1] * k2l[:, 1]
            Sxy = -0.5 * (k1l[:, 0] * k2l[:, 1] + k1l[:, 1] * k2l[:, 0])
            Al = np.stack([Sxx - Syy, -2 * Sxy], 1)
            A = Al @ R
            A = A - Kh * np.sum(Kh * A, 1)[:, None]
        for ie in H.ie:
            T = tablolar[ie]
            F = T(Kn, phK, "F")
            amp = D * ukk * F
            if gA != 0.0:
                amp = amp + gA * (-2j * KIN) * (A[:, 0] * T(Kn, phK, "Gx") + A[:, 1] * T(Kn, phK, "Gy"))
            amp = amp * uh * ul * maske
            a2 = np.abs(amp) ** 2 * W
            tot[ie] += dens * float(np.sum(a2))
            dK_hist += dens * np.histogram(Kn, bins=Kb, weights=a2)[0]
    vals = [pref * t for t in tot.values()]
    G = float(np.mean(vals))
    hist = (pref * dK_hist / len(H.ie) / max(G, 1e-300)).tolist()
    return G, {"qh_range": [qh_lo, qh_hi], "ql_range": [q_of(w, E21 - eh_hi) if E21 - eh_hi > 0 else 0.0, q_of(w, eh_lo)],
               "wmax_meV": wmax * 1e3, "K_dagilimi_pay": hist, "K_kenar": Kb.tolist(), "q0_LA": q0}

# -*- coding: utf-8 -*-
"""
Dogrusal (tek-fonon) kuplaj tepe noktalari, cift katman (s: es-faz, a: ters-faz) duzlem-ici dallar.
Her tepe: V [eV/A] (birim katman-yer-degistirme genligi basina), a_zp^2 = hbar/(2*(2 rho1)*omega).

  DP      : D * i q (e.n) * F(q)                      [s: 1, a: dw]
  PZ      : -(e*e11/(2 eps0 eps)) * q * T(phi,e) * f * F(q)
            T = rho_pol/(e11 q^2 a) atomik eksende (zigzag) ; f_s = 1+exp(-qd), f_a = dw(1-exp(-qd))
  GAUGE   : (hbar^2/m*) [k_A . <e^{iqr}(-i grad)> + (q.k_A)/2 F],  k_A = (beta/2a_cc) i q S(phi,e)  [s:1, a:dw]
  REGISTRY: (yalniz a)  2 <(W.e) e^{iqr}> = i sum_G P_G (g_G.e) [F(q+G) - F(q-G)],
            g_G = zhat x G / (2 sin(theta/2))   (Kol E V(s) kayit potansiyeli; d = u1-u2 = 2a)
Atomik zigzag ekseni: moire B1 (30 derece) <-> atomik g = zhat x G/theta ; atomik a1 || y.
"""
import math
import numpy as np

eC = 1.602176634e-19; eps0 = 8.8541878128e-12
HB2_ME = 2 * 3.80998212          # eV A^2  (hbar^2/m_e)
A_CC = 1.446                     # A (a/sqrt3)
ZIGZAG = math.pi / 2             # atomik a1 yonu (moire cercevesinde)


def S_vec(phi, pol):
    """(u_xx-u_yy, -2u_xy)/(i q a) atomik cercevede, moire cercevesine geri donduruldu.
    pol 'L': e=n, 'T': e = zhat x n."""
    n = np.stack([np.cos(phi), np.sin(phi)], -1)
    e = n if pol == "L" else np.stack([-np.sin(phi), np.cos(phi)], -1)
    c, s = math.cos(-ZIGZAG), math.sin(-ZIGZAG)
    R = np.array([[c, -s], [s, c]])
    na = n @ R.T; ea = e @ R.T
    Sx = na[..., 0] * ea[..., 0] - na[..., 1] * ea[..., 1]
    Sy = -(na[..., 0] * ea[..., 1] + na[..., 1] * ea[..., 0])
    Sa = np.stack([Sx, Sy], -1)
    return Sa @ R, n, e      # geri dondur: R^T ters -> (R^-1)^T = R


def T_pz(phi, pol):
    """rho_pol/(e11 q^2 a) = n . S  (skaler, donmez)."""
    S, n, e = S_vec(phi, pol)
    return np.sum(n * S, -1)


class Tepe:
    def __init__(self, kuyu, FT, par, GT=None):
        """FT: FTablo (tum (a,b) ciftleri), GT: {'gx','gy'} FTablo (gauge icin, opsiyonel)."""
        self.k, self.FT, self.GT, self.p = kuyu, FT, GT, par
        th = math.radians(kuyu.theta)
        G = kuyu.Glist
        self.g = np.stack([-G[:, 1], G[:, 0]], -1) / (2 * math.sin(th / 2))   # zhat x G
        self.P = kuyu.Plist
        # cpz [eV] = e*e11/(2 eps0 eps) / e ;  V_pz = cpz * q[1/A] * T * f * F  -> eV/A
        self.cpz = par["e11"] / (2 * eps0 * par["eps"])
        self.cg = HB2_ME / par["mstar"] * par["beta"] / (2 * A_CC)          # eV A

    def V(self, a, b, qA, phi, pol, br, terimler=("dp", "pz", "g", "reg"), regsign=1.0):
        """V_ab(q,phi) [eV/A]; qA [1/A] skaler, phi dizi."""
        p = self.p; dw = p["dw"]
        kx, ky = qA * np.cos(phi), qA * np.sin(phi)
        F = self.FT(a, b, kx, ky)
        out = np.zeros_like(F)
        lay = 1.0 if br == "s" else dw
        if "dp" in terimler and pol == "L":
            out = out + p["D"] * 1j * qA * F * lay
        if "pz" in terimler:
            ex = np.exp(-qA * p["d_int"])
            f = (1 + ex) if br == "s" else dw * (1 - ex)
            out = out - self.cpz * qA * T_pz(phi, pol) * f * F
        if "g" in terimler and self.GT is not None and lay != 0:
            S, n, e = S_vec(phi, pol)
            kAx, kAy = 1j * qA * S[..., 0], 1j * qA * S[..., 1]
            Gx = -1j * self.GT["gx"](a, b, kx, ky); Gy = -1j * self.GT["gy"](a, b, kx, ky)
            qdotk = qA * (np.cos(phi) * kAx + np.sin(phi) * kAy)
            out = out + self.cg * (kAx * Gx + kAy * Gy + 0.5 * qdotk * F) * lay
        if "reg" in terimler and br == "a":
            n = np.stack([np.cos(phi), np.sin(phi)], -1)
            e = n if pol == "L" else np.stack([-np.sin(phi), np.cos(phi)], -1)
            acc = np.zeros_like(F)
            for Gv, gv, Pv in zip(self.k.Glist, self.g, self.P):
                Fp = self.FT(a, b, kx + Gv[0], ky + Gv[1])
                Fm = self.FT(a, b, kx - Gv[0], ky - Gv[1])
                acc = acc + Pv * (e @ gv) * (Fp - Fm)
            out = out + regsign * 1j * acc
        return out

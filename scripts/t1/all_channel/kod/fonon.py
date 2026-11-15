# -*- coding: utf-8 -*-
"""
hBN tek-katman fonon dallari (SI). Kaynaklar (her sayi):
 - v_LA = 21.0, v_TA = 13.0 km/s : Sanchez-Portal & Hernandez, PRB 66, 235415 (2002),
   arXiv:cond-mat/0211173 (DFT tek-katman egimleri).
 - v_LA = 18.86, v_TA = 11.87 km/s : Bosak+ PRB 73, 041402(R) (2006) C11=811, C12=169 GPa,
   rho3D=2279 kg/m^3 (dogrulayicinin t1_yeniden.py tablosundan).
 - ZA: omega = delta q^2 (kucuk q), delta = 5.5e-7 m^2/s : Sanchez-Portal (ayni kaynak).
   ZA(M) = 39.4 meV : Serrano+ PRL 98, 095503 (2007), Tablo II 2ZA(M)=78.8 meV (IXS ve teori).
 - Katmanlar-arasi nefes (breathing) bulk B1g(ZA) teori 15.0 meV (Serrano Tablo I);
   cift katman lineer-zincir: omega_B = 15.0/sqrt2 = 10.6 meV.
 - Kayma (shear) E2g 6.4 meV (Serrano Tablo I, Raman); cift katman 6.4/sqrt2 = 4.5 meV.
 - ZO: TA-ZO halkasi 77-83 meV, LA-ZO halkasi 90.5 meV (Tao+ arXiv:2502.18764, HREELS);
   ZO(Gamma) A2u 97.1 meV (Serrano). -> ZO dalinin tamami >= ~77 meV.
 - TO/LO: E1u(TO) 169.5, E1u(LO) 199.6 meV (Serrano Tablo I).
 - K noktasi (Serrano Tablo II IXS kombinasyonlari TA+LA(K)=242.1, TA+LO(K)=269.0,
   LA+LO(K)=289.9 -> TA(K)=110.6, LA(K)=131.5, LO(K)=158.4 meV; ZA+ZO(K)=115.2 meV ve
   ZO-ZA yarilmasi ~250 cm^-1=31 meV (Sanchez-Portal) -> ZA(K)~42, ZO(K)~73 meV).
 - rho_1 = 7.59e-7 kg/m^2 (2 atom / hucre, a=2.504 A; Bruzzone&Fiori arXiv:1111.1953).
"""
import math
import numpy as np

hbar = 1.054571817e-34; eC = 1.602176634e-19
RHO1 = 7.59e-7
A_HBN = 2.504e-10
Q_M = 2 * math.pi / (math.sqrt(3) * A_HBN)        # Gamma-M, 1/m
meV = 1e-3 * eC / hbar                               # rad/s per meV

ZA_M = 39.4 * meV
DELTA_ZA = 5.5e-7
W_B = 15.0 / math.sqrt(2) * meV
W_SH = 6.4 / math.sqrt(2) * meV


class Dal:
    """omega(q) [rad/s], vg(q) [m/s] ; q [1/m] (izotrop)."""

    def __init__(self, ad, w, vg, qmax=Q_M):
        self.ad, self._w, self._vg, self.qmax = ad, w, vg, qmax

    def w(self, q):
        return self._w(np.asarray(q, float))

    def vg(self, q):
        return self._vg(np.asarray(q, float))

    def q_of_w(self, om):
        """omega(q)=om cozumu (monoton dal), yoksa nan."""
        om = np.asarray(om, float)
        lo = np.zeros_like(om) + 1e3; hi = np.zeros_like(om) + self.qmax
        # dal araliginda olmali: w(q~0) <= om <= w(qmax)  (gapli dallarda alt sinir!)
        ok = (self.w(hi) >= om) & (self.w(lo) <= om)
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            m = self.w(mid) < om
            lo = np.where(m, mid, lo); hi = np.where(m, hi, mid)
        return np.where(ok, 0.5 * (lo + hi), np.nan)


def dogrusal(ad, v):
    return Dal(ad, lambda q: v * q, lambda q: v + 0 * q)


def sinus(ad, v):
    wM = 2 * v * Q_M / math.pi
    return Dal(ad, lambda q: wM * np.sin(np.pi * q / (2 * Q_M)),
               lambda q: v * np.cos(np.pi * q / (2 * Q_M)))


def za_dal(gap=0.0):
    """omega = delta q^2 + c3 q^3 + c4 q^4 ; omega(Q_M)=ZA_M, omega'(Q_M)=0 kosullari.
    gap>0: sqrt(omega_ZA^2 + gap^2) (cift-katman ters-faz/nefes dali)."""
    x = Q_M; d = DELTA_ZA
    A = np.array([[x ** 3, x ** 4], [3 * x ** 2, 4 * x ** 3]])
    bb = np.array([ZA_M - d * x ** 2, -2 * d * x])
    c3, c4 = np.linalg.solve(A, bb)

    def w0(q):
        return d * q ** 2 + c3 * q ** 3 + c4 * q ** 4

    def v0(q):
        return 2 * d * q + 3 * c3 * q ** 2 + 4 * c4 * q ** 3
    if gap == 0:
        return Dal("ZA", w0, v0)
    return Dal("ZOp", lambda q: np.sqrt(w0(q) ** 2 + gap ** 2),
               lambda q: w0(q) * v0(q) / np.sqrt(w0(q) ** 2 + gap ** 2))


if __name__ == "__main__":
    za = za_dal(); zb = za_dal(W_B)
    for qA in (0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.45):
        q = qA * 1e10
        print("q=%.2f/A  ZA=%.2f meV vg=%.0f m/s | ZO'=%.2f | LA(sin,21)=%.1f" % (
            qA, za.w(q) / meV, za.vg(q), zb.w(q) / meV, sinus("LA", 21e3).w(q) / meV))
    for E in (16.2, 24.1, 32.2):
        print("ZA q(E=%.1f meV) = %.3f /A" % (E, za.q_of_w(E * meV) * 1e-10))

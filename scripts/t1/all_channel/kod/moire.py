# -*- coding: utf-8 -*-
"""
BAGIMSIZ duzlem-dalga cozucu + form-faktor tablolari (Kol E kodu import EDILMEZ;
yalnizca katsayi JSON'lari okunur).

V(r) = sum_{(m,n) listeli} P_mn cos(G_mn . r),  G = m B1 + n B2,
B1 = (2pi/L)(1, 1/sqrt3), B2 = (2pi/L)(0, 2/sqrt3)  (Kol E 'genel' konvansiyonu).
H = (hbar^2/2m*) |k+G|^2 + V ;  Gamma noktasi; bantlar ~ueV duz -> Gamma durumlari
tek-kuyu durumlarinin (Wannier) iyi temsilcisi; gercek-uzay penceresi |r|<0.5 L.

F_ab(k) = int_pencere psi_a^* e^{i k.r} psi_b d^2r  -> 2B FFT (sifir dolgulu), k-izgarasi.
Gauss YOK; zarf kuyrugu dogrudan PW cozumunden.
"""
import json, math, os
import numpy as np
from scipy.linalg import eigh
from scipy.ndimage import map_coordinates

KOLE = r"C:\Users\prof.dr.f.sami\Desktop\Moire Qubit\.claude\worktrees\great-wozniak-614fd7\Desktop\Moire Qubit\kolE_hBN"
HB2_2ME = 3.80998212          # eV A^2
A_HBN = 2.504                 # A
MSTAR = 0.77                  # Kol E
STAR1 = [(1, 0), (-1, 1), (0, -1)]
STAR2 = [(1, 1), (-2, 1), (1, -2)]
STAR3 = [(2, 0), (-2, 2), (0, -2)]
STAR4 = [(2, 1), (-3, 2), (1, -3), (1, 2), (-3, 1), (2, -3)]
STARS = [STAR1, STAR2, STAR3, STAR4]


def moire_L(th):
    return A_HBN / (2.0 * math.sin(math.radians(th) / 2.0))


def noktalar():
    """Uc calisma noktasinin 4-yildiz katsayilari (eV) ve aci — Kol E JSON'larindan."""
    t0a = json.load(open(os.path.join(KOLE, "preprint", "data", "t0a_potansiyel.json"), encoding="utf-8"))
    P4 = [p * 1e-3 for p in t0a["fit"]["P_meV"]]
    df = json.load(open(os.path.join(KOLE, "results", "data", "derin_fit.json")))
    f2a = json.load(open(os.path.join(KOLE, "results", "data", "faz2_analiz.json"), encoding="utf-8"))
    par = json.load(open(os.path.join(KOLE, "mantiksal_qubit", "data", "pareto.json"), encoding="utf-8"))
    TAB3 = df["taban_yol"]["P"]
    out = {"ortam_1.09": (list(P4), 1.09)}
    r3 = [p * 1e-3 / TAB3[i] for i, p in enumerate(f2a["P1B"]["3.0"]["fit"]["P"])]
    out["3GPa_1.09"] = ([P4[0] * r3[0], P4[1] * r3[1], P4[2] * r3[2], P4[3] * r3[0]], 1.09)
    sl = par["lnslope_p"]
    r5 = [math.exp(np.polyval(sl[i], 5.0)) for i in range(3)]
    out["5GPa_1.20"] = ([P4[0] * r5[0], P4[1] * r5[1], P4[2] * r5[2], P4[3] * r5[0]], 1.20)
    return out


class Kuyu:
    def __init__(self, P4, theta, gcut_b=16.0, mstar=MSTAR):
        self.P4, self.theta = P4, theta
        self.L = L = moire_L(theta)
        self.B1 = (2 * np.pi / L) * np.array([1.0, 1.0 / math.sqrt(3.0)])
        self.B2 = (2 * np.pi / L) * np.array([0.0, 2.0 / math.sqrt(3.0)])
        self.b = np.linalg.norm(self.B1)
        self.kin = HB2_2ME / mstar
        # listeli Fourier vektorleri ve genlikleri (cos genligi)
        self.Glist, self.Plist = [], []
        for P, st in zip(P4, STARS):
            for (m, n) in st:
                self.Glist.append(m * self.B1 + n * self.B2); self.Plist.append(P)
        self.Glist = np.array(self.Glist); self.Plist = np.array(self.Plist)
        gc = gcut_b * self.b
        nmax = int(gc / (self.b * 0.8)) + 2
        idx, vec = [], []
        for m in range(-nmax, nmax + 1):
            for n in range(-nmax, nmax + 1):
                G = m * self.B1 + n * self.B2
                if np.linalg.norm(G) <= gc:
                    idx.append((m, n)); vec.append(G)
        self.idx = idx; self.Gs = np.array(vec); self.N = len(idx)
        lut = {t: i for i, t in enumerate(idx)}
        H = np.diag(self.kin * np.sum(self.Gs ** 2, axis=1)).astype(complex)
        allv = {}
        for P, st in zip(P4, STARS):
            for (m, n) in st:
                allv[(m, n)] = allv.get((m, n), 0.0) + P / 2
                allv[(-m, -n)] = allv.get((-m, -n), 0.0) + P / 2
        for (tm, tn), u in allv.items():
            for i, (m, n) in enumerate(idx):
                j = lut.get((m - tm, n - tn))
                if j is not None:
                    H[i, j] += u
        self.H = H

    def coz(self, nst=10):
        w, v = eigh(self.H, subset_by_index=[0, nst - 1])
        self.E = w; self.c = v
        return w

    def V(self, X, Y):
        V = np.zeros_like(X)
        for G, P in zip(self.Glist, self.Plist):
            V += P * np.cos(G[0] * X + G[1] * Y)
        return V

    def gercek_uzay(self, h=0.6, rfrac=0.5):
        """psi ve grad psi pencere |r|<rfrac*L (kuyu merkezi r=0, AA)."""
        R = rfrac * self.L
        n = int(2 * R / h) + 1
        x = (np.arange(n) - (n - 1) / 2) * h
        X, Y = np.meshgrid(x, x, indexing="ij")
        rr = np.sqrt(X ** 2 + Y ** 2)
        # yumusak pencere: 0.8R..R arasinda cos^2 sondurme (keskin kenarin sahte
        # yuksek-q icerigini onler; kenar agirligi derin kuyuda ~1e-5)
        t = np.clip((rr - 0.8 * R) / (0.2 * R), 0.0, 1.0)
        mask = np.where(rr <= R, np.cos(0.5 * np.pi * t) ** 2, 0.0)
        ph = np.exp(1j * (np.outer(X.ravel(), self.Gs[:, 0]) + np.outer(Y.ravel(), self.Gs[:, 1])))
        psi = (ph @ self.c).reshape(n, n, -1)
        gx = (ph @ (1j * self.Gs[:, 0:1] * self.c)).reshape(n, n, -1)
        gy = (ph @ (1j * self.Gs[:, 1:2] * self.c)).reshape(n, n, -1)
        psi *= mask[..., None]; gx *= mask[..., None]; gy *= mask[..., None]
        dA = h * h
        nrm = np.sqrt((np.abs(psi) ** 2).sum(axis=(0, 1)) * dA)
        psi /= nrm; gx /= nrm; gy /= nrm
        # E dubletini gercel-simetri tabanina dondurmeye gerek yok (oranlar toplamla alinir)
        self.x, self.X, self.Y, self.h, self.dA = x, X, Y, h, dA
        self.psi, self.gx, self.gy = psi, gx, gy
        return psi

    def kenar_agirligi(self):
        """|r|>0.45L halkasindaki olasilik (pencere kesme hatasi gostergesi)."""
        r = np.sqrt(self.X ** 2 + self.Y ** 2)
        m = r > 0.45 * self.L
        return (np.abs(self.psi[m]) ** 2).sum(axis=0) * self.dA


class FTablo:
    """F_ab(k) k-izgarasinda (FFT, sifir dolgu). k [1/A]."""

    def __init__(self, kuyu, pairs, kmax=1.2, pad=1024, op="1"):
        psi = kuyu.psi; n = psi.shape[0]; h = kuyu.h
        self.dk = 2 * np.pi / (pad * h)
        nk = int(kmax / self.dk) + 4
        self.nk = nk
        self.kax = np.arange(-nk, nk + 1) * self.dk
        x0 = kuyu.x[0]
        self.T = {}
        for (a, b) in pairs:
            if op == "1":
                f = np.conj(psi[..., a]) * psi[..., b]
            elif op == "gx":
                f = np.conj(psi[..., a]) * kuyu.gx[..., b]
            elif op == "gy":
                f = np.conj(psi[..., a]) * kuyu.gy[..., b]
            buf = np.zeros((pad, pad), complex)
            buf[:n, :n] = f
            # F(k) = sum f(r) e^{+ik r} dA ;  r = x0 + j h  -> ifft * pad^2
            Fk = np.fft.ifft2(buf) * pad * pad * kuyu.dA
            Fk = np.fft.fftshift(Fk)
            c = pad // 2
            sub = Fk[c - nk:c + nk + 1, c - nk:c + nk + 1]
            kk = self.kax
            KX, KY = np.meshgrid(kk, kk, indexing="ij")
            sub = sub * np.exp(1j * (KX + KY) * x0)  # pencere ofseti fazi
            self.T[(a, b)] = sub

    def __call__(self, a, b, kx, ky):
        kx = np.asarray(kx, float); ky = np.asarray(ky, float)
        ix = kx / self.dk + self.nk; iy = ky / self.dk + self.nk
        T = self.T[(a, b)]
        re = map_coordinates(T.real, [ix.ravel(), iy.ravel()], order=3, mode="nearest")
        im = map_coordinates(T.imag, [ix.ravel(), iy.ravel()], order=3, mode="nearest")
        return (re + 1j * im).reshape(kx.shape)


def F_dogrudan(kuyu, a, b, kx, ky, op="1"):
    """Kontrol: dogrudan gercek-uzay toplami (tek k)."""
    psi = kuyu.psi
    g = {"1": psi, "gx": kuyu.gx, "gy": kuyu.gy}[op]
    ph = np.exp(1j * (kx * kuyu.X + ky * kuyu.Y))
    return np.sum(np.conj(psi[..., a]) * ph * g[..., b]) * kuyu.dA

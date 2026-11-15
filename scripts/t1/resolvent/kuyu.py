# -*- coding: utf-8 -*-
"""
BAGIMSIZ kuyu cozucu (hakem hatti 'iki_fonon').
- Potansiyel katsayilari Kol E JSON'larindan (SADECE OKUNUR) ayni tarifle kurulur.
- Cozum: kendi 4. mertebe sonlu-fark (FD) kare-izgara cozucum, Dirichlet kutu
  (yari-genislik W = 0.5 L, AA kuyusu merkezde). Kol E pw_cozucu IMPORT EDILMEZ.
- Ayrica kendi duzlem-dalga (PW) Gamma cozucum E21 capraz kontrolu icin.
Birimler: uzunluk A, enerji eV.
"""
import json, math, os
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

KOLE = (r"C:\Users\prof.dr.f.sami\Desktop\Moire Qubit\.claude\worktrees"
        r"\great-wozniak-614fd7\Desktop\Moire Qubit\kolE_hBN")
HBAR2_2ME = 3.8099821      # eV A^2
MSTAR = 0.77               # Kol E (pw_cozucu.py) K-vadisi kutle
KIN = HBAR2_2ME / MSTAR
A_HBN = 2.504
STARS = [[(1, 0), (-1, 1), (0, -1)], [(1, 1), (-2, 1), (1, -2)],
         [(2, 0), (-2, 2), (0, -2)],
         [(2, 1), (-3, 2), (1, -3), (1, 2), (-3, 1), (2, -3)]]


def moire_L(th):
    return A_HBN / (2 * math.sin(math.radians(th) / 2))


def _j(*p):
    return json.load(open(os.path.join(KOLE, *p), encoding="utf-8"))


def noktalar():
    """Kol E iki_fonon_bagimsiz.py / ortam_2f.py ile ayni katsayi tarifi."""
    t0a = _j("preprint", "data", "t0a_potansiyel.json")
    P4 = [p * 1e-3 for p in t0a["fit"]["P_meV"]]
    TAB3 = _j("results", "data", "derin_fit.json")["taban_yol"]["P"]
    f2a = _j("results", "data", "faz2_analiz.json")
    par = _j("mantiksal_qubit", "data", "pareto.json")
    out = {}
    out["ortam_0GPa_1.09"] = (list(P4), 1.09)
    r3 = [p * 1e-3 / TAB3[i] for i, p in enumerate(f2a["P1B"]["3.0"]["fit"]["P"])]
    out["calisma_3GPa_1.09"] = ([P4[0]*r3[0], P4[1]*r3[1], P4[2]*r3[2], P4[3]*r3[0]], 1.09)
    sl = par["lnslope_p"]
    r5 = [math.exp(np.polyval(sl[i], 5.0)) for i in range(3)]
    out["5GPa_1.20_uzatma"] = ([P4[0]*r5[0], P4[1]*r5[1], P4[2]*r5[2], P4[3]*r5[0]], 1.20)
    return out


class Kuyu:
    def __init__(self, P4, th):
        self.P4, self.th = P4, th
        L = moire_L(th); self.L = L
        self.B1 = (2*np.pi/L)*np.array([1.0, 1/math.sqrt(3)])
        self.B2 = (2*np.pi/L)*np.array([0.0, 2/math.sqrt(3)])
        self.G, self.P = [], []
        for Pv, st in zip(P4, STARS):
            for (m, n) in st:
                self.G.append(m*self.B1 + n*self.B2); self.P.append(Pv)
        self.G = np.array(self.G); self.P = np.array(self.P)

    def V(self, x, y):
        out = np.zeros_like(x)
        for g, p in zip(self.G, self.P):
            out += p*np.cos(g[0]*x + g[1]*y)
        return out

    # ---------------- kendi PW cozucum (Gamma) -------------------
    def pw(self, nst=12, gcut_b=12.0):
        b = np.linalg.norm(self.B1); gc = gcut_b*b
        nm = int(gc/b/0.8)+2
        idx = [(m, n) for m in range(-nm, nm+1) for n in range(-nm, nm+1)
               if np.linalg.norm(m*self.B1+n*self.B2) <= gc]
        lut = {t: i for i, t in enumerate(idx)}
        N = len(idx); H = np.zeros((N, N))
        for i, (m, n) in enumerate(idx):
            g = m*self.B1+n*self.B2; H[i, i] = KIN*(g @ g)
        for Pv, st in zip(self.P4, STARS):
            for (a, c) in st:
                for s in (1, -1):
                    for i, (m, n) in enumerate(idx):
                        j = lut.get((m-s*a, n-s*c))
                        if j is not None:
                            H[i, j] += Pv/2
        return eigh(H, eigvals_only=True, subset_by_index=[0, nst-1])

    # ---------------- FD cozucu ------------------------------------
    def kur(self, h=0.6, Wfrac=0.5):
        # kuyu merkezi = V minimumu (AA). cos-only -> r=0 aday; sayisal kontrol
        u = np.linspace(-0.5, 0.5, 201)*self.L
        Xs, Ys = np.meshgrid(u, u, indexing="ij")
        Vs = self.V(Xs, Ys); i = np.unravel_index(np.argmin(Vs), Vs.shape)
        self.c0 = (Xs[i], Ys[i])
        W = Wfrac*self.L
        n = int(round(2*W/h)) - 1
        x = -W + h*np.arange(1, n+1)
        self.h, self.n = (2*W)/(n+1), n
        x = -W + self.h*np.arange(1, n+1)
        self.x = x
        X, Y = np.meshgrid(x, x, indexing="ij")
        self.X, self.Y = X.ravel(), Y.ravel()
        Vg = self.V(self.X + self.c0[0], self.Y + self.c0[1])
        self.Vg = Vg
        # 4. mertebe 1B Laplasyen (Dirichlet)
        c = np.array([-1/12, 4/3, -5/2, 4/3, -1/12])/self.h**2
        D2 = sp.diags([c[0]*np.ones(n-2), c[1]*np.ones(n-1), c[2]*np.ones(n),
                       c[3]*np.ones(n-1), c[4]*np.ones(n-2)], [-2, -1, 0, 1, 2])
        I = sp.identity(n)
        Lap = sp.kron(D2, I) + sp.kron(I, D2)
        self.H = (-KIN*Lap + sp.diags(Vg)).tocsc()
        # 4. mertebe birinci turevler (gauge kuplaji icin)
        d = np.array([1/12, -2/3, 0, 2/3, -1/12])/self.h
        D1 = sp.diags([d[0]*np.ones(n-2), d[1]*np.ones(n-1),
                       d[3]*np.ones(n-1), d[4]*np.ones(n-2)], [-2, -1, 1, 2])
        self.Dx = sp.kron(D1, I).tocsr(); self.Dy = sp.kron(I, D1).tocsr()
        self.dA = self.h**2
        return self

    def coz(self, nst=12):
        sig = float(self.Vg.min()) - 0.02
        w, v = eigsh(self.H, k=nst, sigma=sig, which="LM")
        o = np.argsort(w); w, v = w[o], v[:, o]
        v = v/np.sqrt(self.dA)          # sum |psi|^2 dA = 1
        self.w, self.psi = w, v.T.copy()  # psi[state, pt] (gercel)
        return w

    def gruplar(self, tol=2e-4):
        g, i = [], 0
        while i < len(self.w):
            j = i+1
            while j < len(self.w) and abs(self.w[j]-self.w[i]) < tol:
                j += 1
            g.append(list(range(i, j))); i = j
        return g


if __name__ == "__main__":
    import time
    for ad, (P4, th) in noktalar().items():
        k = Kuyu(P4, th)
        e_pw = k.pw(10)
        print(ad, "P4[meV]=", [round(p*1e3, 2) for p in P4], "L=%.1f A" % k.L)
        print("  PW E-E0 [meV]:", np.round((e_pw-e_pw[0])*1e3, 2))
        for h in (0.8, 0.6):
            t = time.time(); k.kur(h=h); w = k.coz(12)
            print("  FD h=%.2f n=%d E-E0:" % (k.h, k.n), np.round((w-w[0])*1e3, 2),
                  "Vmin=%.1f  %.0fs" % (k.Vg.min()*1e3, time.time()-t))

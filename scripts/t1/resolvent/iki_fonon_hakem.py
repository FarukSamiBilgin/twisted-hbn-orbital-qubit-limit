# -*- coding: utf-8 -*-
"""
HAKEM HATTI 'iki_fonon': hBN AA-moire kuyusunda E -> A1 gecisinin iki-fonon
kendiliginden emisyonu (T=0).

KANALLAR
 (1) LA+LA, LA+TA, TA+TA : ikinci mertebe, ara durum toplami KAPALI FORMDA
     (cozucu/resolvent R(z)=(z-H+i eta)^-1, Dirichlet kutusunda TUM ara durumlar).
     Kuplajlar:  deformasyon  D eps_kk                     (yalniz LA)
                 psodo-ayar   (hbar^2/m*) A_T . p,  A = (beta/2a)(eps_xx-eps_yy, -2 eps_xy)
                 (A_T: k'ya dik kisim; boyuna kisim saf ayar -> Coulomb ayarinda atilir)
                 -> TA'yi kuplayan tek mekanizma bu (TA'da eps_kk = 0).
 (2) ZA+ZA : egilme modlarinin KUADRATIK kuplaji, BIRINCI mertebe:
     u_kk -> (1/2)(grad h)^2 ; duzlem-ici esnek perdeleme (dinamik) dahil;
     perdelemesiz ve statik-perdelemeli sinirlar ayrica.
 (3) Ayni makineyle tek-fonon LA/TA (kiyas).

Hiz:  Gamma = (2pi/hbar) S (2pi)^-4 Int de rho_a(e) rho_b(E21-e) Int dphi_a dphi_b |T|^2
      S = 1/2 ozdes dal, 1 farkli dal;  rho(e) = k/(hbar v_g).
Birimler: eV, A; u_k = sqrt(hbar/(2 rho w)) [A^2] (alan 1/sqrt(A) disari alinmis).
Cikti: data/iki_fonon_hakem.json
"""
import json, math, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kuyu import Kuyu, noktalar, KIN

os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
HBAR_eVs = 6.582119569e-16
HBAR_Js = 1.054571817e-34
EC = 1.602176634e-19

# ---------------- literatur parametreleri ---------------------------------
RHO = 7.59e-7          # kg/m^2   Bruzzone&Fiori arXiv:1111.1953 Tablo I
D_BAND = (3.66, 4.5)   # eV  3.66: Bruzzone&Fiori (Gamma-LUMO!), 4.5: Kol E degeri (kaynaksiz)
VLA_BAND = (18.86e3, 19.94e3, 21.0e3)  # Bosak PRB 73,041402 (bulk IXS) / orta / Sanchez-Portal cond-mat/0211173
VTA_BAND = (10.98e3, 11.9e3, 13.0e3)   # arXiv:2502.10649 / Bosak (11.86) / cond-mat/0211173
A_LAT = 2.504          # A
BETA = 1.7             # Luo, Lafuente-Bartolome, Giustino arXiv:2601.17235 (A_K = beta/(2a)(...), a=2.504)
GA_MERKEZ = BETA / (2 * A_LAT)          # 1/A  = 0.339
GA_UST = 2.0 / (2 * 1.446)              # grafen-tipi konvansiyon (beta~2, a_cc) ust sinir 0.692
ZA_M = 39.4e-3         # eV  ZA(M) = 78.8/2 meV, Serrano+ PRL 98, 095503 (2007) Tablo II (IXS)
Q_M = 2 * math.pi / (math.sqrt(3) * A_LAT)   # 1/A Gamma-M
KAPPA_BAND = (0.86, 1.29)  # eV egilme sertligi (1.29: Singh+ PRB 87,184106 atfi; 0.86: Kudin+ PRB 64,235406, dogrulanmadi)
ETA = 0.3e-3           # eV ara-durum genislemesi (Kol E ile ayni); duyarlilik ayrica


def u_amp(w_eV):
    """sqrt(hbar/(2 rho w)) [A^2]; w_eV = hbar*omega [eV]."""
    om = w_eV / HBAR_eVs
    return math.sqrt(HBAR_Js / (2 * RHO * om)) * 1e20


def ga_vec(kv, ev, off):
    """In-plane mod (k, polarizasyon e) icin A_T/(-i u) [birim g_A basina, 1/A].
    S_ij = (k_i e_j + k_j e_i)/2 ; A = (S_xx - S_yy, -2 S_xy) orgu cercevesinde
    (orgu cercevesi moire cercevesine gore 'off' kadar donuk)."""
    c, s = math.cos(off), math.sin(off)
    R = np.array([[c, s], [-s, c]])       # moire -> orgu
    kl, el = R @ kv, R @ ev
    S = 0.5 * (np.outer(kl, el) + np.outer(el, kl))
    A = np.array([S[0, 0] - S[1, 1], -2 * S[0, 1]])
    A = R.T @ A                            # geri moire cercevesine
    kh = kv / np.linalg.norm(kv)
    return A - kh * (kh @ A)               # enine kisim


class Hesap:
    def __init__(self, P4, th, h=0.75, nphi=24):
        self.k = Kuyu(P4, th).kur(h=h)
        self.k.coz(nst=12)
        g = self.k.gruplar()
        self.ig, self.ie = g[0][0], g[1]
        assert len(self.ie) == 2, g[:3]
        w = self.k.w
        self.E21 = float(np.mean(w[self.ie]) - w[self.ig])
        self.nphi = nphi
        self.phis = 2 * np.pi * (np.arange(nphi) + 0.25) / nphi
        X, Y = self.k.X, self.k.Y
        self.X, self.Y = X, Y
        self.dA = self.k.dA
        self.H = self.k.H
        self.Npt = len(X)
        self.Dx, self.Dy = self.k.Dx, self.k.Dy
        self.DxT, self.DyT = self.Dx.T.tocsr(), self.Dy.T.tocsr()
        self.psi = self.k.psi

    # --- mod vektorleri -----------------------------------------------
    def mod(self, lam, eps, v, phi):
        kk = eps / (HBAR_eVs * v * 1e10)            # 1/A (dogrusal dispersiyon)
        kh = np.array([math.cos(phi), math.sin(phi)])
        ev = kh if lam == "LA" else np.array([-kh[1], kh[0]])
        return kk, kk * kh, ev

    def kaynaklar(self, lam, eps, v, psi_in):
        """W|psi_in> parcalari: {'d': def (D=1), 'g0','g30': ayar (g_A=1)}; [nphi, Npt]."""
        u = u_amp(eps)
        out = {p: np.zeros((self.nphi, self.Npt), complex) for p in ("d", "g0", "g30")}
        gx = self.Dx @ psi_in; gy = self.Dy @ psi_in
        for ip, phi in enumerate(self.phis):
            kk, kv, ev = self.mod(lam, eps, v, phi)
            ph = np.exp(-1j * (kv[0] * self.X + kv[1] * self.Y))
            if lam == "LA":
                out["d"][ip] = (-1j * u) * kk * ph * psi_in
            for p, off in (("g0", 0.0), ("g30", math.pi / 6)):
                a = ga_vec(kv, ev, off)
                out[p][ip] = (-1j * u) * (-2j * KIN) * ph * (a[0] * gx + a[1] * gy)
        return out

    def bralar(self, lam, eps, v, psi_g):
        """<g|W  dogrusal fonksiyonelleri (satir), dA dahil; ayni parca anahtarlari."""
        u = u_amp(eps)
        out = {p: np.zeros((self.nphi, self.Npt), complex) for p in ("d", "g0", "g30")}
        for ip, phi in enumerate(self.phis):
            kk, kv, ev = self.mod(lam, eps, v, phi)
            r = np.conj(psi_g) * np.exp(-1j * (kv[0] * self.X + kv[1] * self.Y))
            if lam == "LA":
                out["d"][ip] = self.dA * (-1j * u) * kk * r
            rx = self.DxT @ r; ry = self.DyT @ r
            for p, off in (("g0", 0.0), ("g30", math.pi / 6)):
                a = ga_vec(kv, ev, off)
                out[p][ip] = self.dA * (-1j * u) * (-2j * KIN) * (a[0] * rx + a[1] * ry)
        return out

    # --- tek fonon ------------------------------------------------------
    def tek_fonon(self, v_by):
        res = {}
        g = self.psi[self.ig]
        for lam in ("LA", "TA"):
            v = v_by[lam]
            amps = {p: np.zeros((2, self.nphi), complex) for p in ("d", "g0", "g30")}
            for j, ie in enumerate(self.ie):
                br = self.bralar(lam, self.E21, v, g)
                for p in amps:
                    amps[p][j] = br[p] @ self.psi[ie]
            kk = self.E21 / (HBAR_eVs * v * 1e10)
            rho_e = kk / (HBAR_eVs * v * 1e10)
            res[lam] = {"amps": amps, "pref": (2 * math.pi / HBAR_eVs) * (2 * math.pi) ** -2
                        * rho_e * (2 * math.pi / self.nphi), "k": kk}
        return res

    # --- iki fonon (in-plane) --------------------------------------------
    def iki_fonon(self, v_by, Ne=24, eta=ETA, log=True, lams=("LA", "TA")):
        """Bloklar: B[tip][i] = {(pa,pb): T matrisi [phi_b, phi_a]} ; e-ortaklarina gore liste."""
        E21 = self.E21
        eps = E21 * (np.arange(Ne) + 0.5) / Ne
        de = E21 / Ne
        g = self.psi[self.ig]
        I = sp.identity(self.Npt, format="csc")
        kayit = []
        for ie in self.ie:
            Ee = self.k.w[ie]
            psi_e = self.psi[ie]
            for i in range(Ne // 2):
                i2 = Ne - 1 - i
                chi, bra = {}, {}
                for j in (i, i2):
                    z = Ee - eps[j] + 1j * eta
                    lu = splu((z * I - self.H).tocsc())
                    for lam in lams:
                        src = self.kaynaklar(lam, eps[j], v_by[lam], psi_e)
                        bra[(lam, j)] = self.bralar(lam, eps[j], v_by[lam], g)
                        for p, M in src.items():
                            if lam == "TA" and p == "d":
                                continue
                            chi[(lam, j, p)] = lu.solve(M.T.copy()).T   # [nphi, Npt]
                # T bloklari
                def T(la, ja, lb, jb):
                    """a=(la, eps[ja]), b=(lb, eps[jb]); T[phi_b, phi_a] her (pa,pb)."""
                    out = {}
                    pa_l = ("d", "g0", "g30") if la == "LA" else ("g0", "g30")
                    pb_l = ("d", "g0", "g30") if lb == "LA" else ("g0", "g30")
                    for pa in pa_l:
                        for pb in pb_l:
                            t1 = bra[(lb, jb)][pb] @ chi[(la, ja, pa)].T          # [phib, phia]
                            t2 = (bra[(la, ja)][pa] @ chi[(lb, jb, pb)].T).T      # [phib, phia]
                            out[(pa, pb)] = t1 + t2
                    return out
                rec = {"i": i, "eps": (eps[i], eps[i2]), "LL": T("LA", i, "LA", i2)}
                if "TA" in lams:
                    rec.update({"TT": T("TA", i, "TA", i2),
                                "LT_a": T("LA", i, "TA", i2), "LT_b": T("LA", i2, "TA", i)})
                kayit.append((ie, rec))
            if log:
                print("    e-ortagi %d bitti" % ie, flush=True)
        return {"kayit": kayit, "eps": eps, "de": de, "Ne": Ne}


def birlestir(blk, v_by, D, gA, off, nphi, kanal):
    """Bloklardan hiz [1/s]; kanal in LL, TT, LT. e-ortaklari ortalamasi."""
    gp = "g0" if off == 0 else "g30"
    coef = {"d": D, gp: gA}
    eps, de = blk["eps"], blk["de"]
    dphi2 = (2 * math.pi / nphi) ** 2
    def rho(e, lam):
        v = v_by[lam]; kk = e / (HBAR_eVs * v * 1e10)
        return kk / (HBAR_eVs * v * 1e10)
    tot = {}
    for ie, rec in blk["kayit"]:
        e1, e2 = rec["eps"]
        if kanal == "LL":
            parts = [(rec["LL"], rho(e1, "LA") * rho(e2, "LA"))]
            S = 1.0      # (1/2) * 2 (i<->i2 simetrisi)
        elif kanal == "TT":
            parts = [(rec["TT"], rho(e1, "TA") * rho(e2, "TA"))]
            S = 1.0
        else:
            parts = [(rec["LT_a"], rho(e1, "LA") * rho(e2, "TA")),
                     (rec["LT_b"], rho(e2, "LA") * rho(e1, "TA"))]
            S = 1.0
        for Tb, rr in parts:
            Tm = 0
            for (pa, pb), M in Tb.items():
                ca = coef.get(pa, 0.0); cb = coef.get(pb, 0.0)
                if ca == 0 or cb == 0:
                    continue
                Tm = Tm + ca * cb * M
            if isinstance(Tm, int):
                continue
            tot[ie] = tot.get(ie, 0.0) + S * de * rr * dphi2 * float(np.sum(np.abs(Tm) ** 2))
    pref = (2 * math.pi / HBAR_eVs) * (2 * math.pi) ** -4
    vals = [pref * t for t in tot.values()]
    return float(np.mean(vals)) if vals else 0.0


def tek_birlestir(tf, lam, D, gA, off):
    gp = "g0" if off == 0 else "g30"
    a = tf[lam]["amps"]
    T = (D * a["d"] if lam == "LA" else 0) + gA * a[gp]
    return float(tf[lam]["pref"] * np.mean(np.sum(np.abs(T) ** 2, axis=1)))


# ======================= ZA + ZA ============================================
def za_disp(model, kappa):
    """(w(q) [eV], v_g(q) [eV A]) ; q [1/A]."""
    if model == "sin2":
        def w(q): return ZA_M * np.sin(np.pi * q / (2 * Q_M)) ** 2
        def vg(q): return ZA_M * np.pi / (2 * Q_M) * np.sin(np.pi * q / Q_M)
    else:
        c = HBAR_eVs * math.sqrt(kappa * EC / RHO) * 1e20     # eV A^2
        qs = Q_M / math.sqrt((c * Q_M ** 2 / ZA_M) ** 2 - 1)
        def w(q): return c * q ** 2 / np.sqrt(1 + (q / qs) ** 2)
        def vg(q):
            f = 1 + (q / qs) ** 2
            return c * (2 * q * f - q ** 3 / qs ** 2) / f ** 1.5
    return w, vg


def q_of(w, e, qmax=Q_M):
    lo, hi = 1e-6, qmax
    for _ in range(80):
        m = 0.5 * (lo + hi)
        if w(m) < e: lo = m
        else: hi = m
    return 0.5 * (lo + hi)


def za_za(H, model, kappa, D, vL, vT, ekran="dinamik", gamLA=0.5e-3, gA=0.0, off=0.0,
          Nk=40, nphi=36):
    """Gamma_ZZ [1/s], e-ortaklari ortalamasi. Ara degisken: yuksek-enerjili fononun q'su."""
    w, vg = za_disp(model, kappa)
    E21 = H.E21
    wmax = float(w(Q_M))
    if E21 / 2 > wmax:
        return 0.0, {}
    # yuksek-enerjili fonon: e_h in [E21/2, min(wmax, E21)]
    eh_lo, eh_hi = E21 / 2, min(wmax, E21)
    qh_lo, qh_hi = q_of(w, eh_lo), q_of(w, eh_hi)
    xs = (np.arange(Nk) + 0.5) / Nk
    qhs = qh_lo + (qh_hi - qh_lo) * xs; dq = (qh_hi - qh_lo) / Nk
    phis = 2 * np.pi * (np.arange(nphi) + 0.25) / nphi
    g = H.psi[H.ig]
    Om = E21 / HBAR_eVs
    # esnek sabitler (2B): mu = rho vT^2, lam+2mu = rho vL^2
    rvL2, rmu = vL ** 2, vT ** 2           # rho'ya bolunmus
    rlam = vL ** 2 - 2 * vT ** 2
    tot = {ie: 0.0 for ie in H.ie}
    pref = (2 * math.pi / HBAR_eVs) * 0.5 * (2 * math.pi) ** -4
    for qh in qhs:
        eh = float(w(qh)); el = E21 - eh
        if el <= 0:
            continue
        ql = q_of(w, el)
        uh, ul = u_amp(eh), u_amp(el)
        # yogunluk: dq_h * q_h * [q_l/(v_g(q_l))] ; x2 (yuksek/dusuk degisimi) ; S=1/2 pref'te
        dens = 2 * qh * dq * ql / float(vg(ql))
        # tum (phi_h, phi_l) ciftleri
        K = []
        k1s = np.stack([qh * np.cos(phis), qh * np.sin(phis)], 1)
        k2s = np.stack([ql * np.cos(phis), ql * np.sin(phis)], 1)
        k1 = np.repeat(k1s, nphi, 0); k2 = np.tile(k2s, (nphi, 1))
        Kv = k1 + k2
        Kn = np.linalg.norm(Kv, axis=1)
        Kh = Kv / np.maximum(Kn, 1e-12)[:, None]
        fkk = -np.sum(k1 * k2, 1)                                  # f_kk/(u1u2)
        fLL = -(np.sum(Kh * k1, 1) * np.sum(Kh * k2, 1))
        if ekran == "yok":
            ukk = fkk
        elif ekran == "statik":
            ukk = fkk - (rlam * fkk + 2 * rmu * fLL) / rvL2
        else:
            # dinamik: f_kk - K^2 (lam f_kk + 2 mu f_LL)/(rho(vL^2 K^2 - Om^2 - i Om gam))
            K2 = (Kn * 1e10) ** 2
            den = rvL2 * K2 - Om ** 2 - 1j * Om * (gamLA / HBAR_eVs)
            ukk = fkk - K2 * (rlam * fkk + 2 * rmu * fLL) / den
        # psodo-ayar (perdelemesiz, isteğe bagli): A = gA (f_xx - f_yy, -2 f_xy), K'ya dik
        # faz matrisi [npair, Npt] -> F(K) = <g| e^{-iK r} |e>
        for ie in H.ie:
            e = H.psi[ie]
            amp = np.zeros(len(Kv), complex)
            # bloklar halinde
            for s0 in range(0, len(Kv), 256):
                sl = slice(s0, s0 + 256)
                ph = np.exp(-1j * (np.outer(Kv[sl, 0], H.X) + np.outer(Kv[sl, 1], H.Y)))
                F = (ph * (np.conj(g) * e)[None, :]).sum(1) * H.dA
                a = D * ukk[sl] * F
                if gA != 0.0:
                    gx = H.Dx @ e; gy = H.Dy @ e
                    c, s = math.cos(off), math.sin(off)
                    R = np.array([[c, s], [-s, c]])
                    for jj, idx in enumerate(range(s0, min(s0 + 256, len(Kv)))):
                        k1l, k2l = R @ k1[idx], R @ k2[idx]
                        Sf = -0.5 * (np.outer(k1l, k2l) + np.outer(k2l, k1l))
                        A = R.T @ np.array([Sf[0, 0] - Sf[1, 1], -2 * Sf[0, 1]])
                        A = A - Kh[idx] * (Kh[idx] @ A)
                        Fg = np.sum(np.conj(g) * ph[jj] * (A[0] * gx + A[1] * gy)) * H.dA
                        a[jj] += gA * (-2j * KIN) * Fg
                amp[sl] = a * uh * ul
            tot[ie] += dens * (2 * math.pi / nphi) ** 2 * float(np.sum(np.abs(amp) ** 2))
    vals = [pref * t for t in tot.values()]
    # boyut: amp [eV A^2], dens [A^-2 / (eV A)] * A^-2 ... -> pref(1/(eV s)) * eV^2 A^4 * A^-4 eV^-1 = 1/s
    # dens'de v_g [eV A] = hbar v_g -> rho = q/(hbar v_g) dogru.
    return float(np.mean(vals)), {"qh_range": (qh_lo, qh_hi), "wmax_meV": wmax * 1e3}


if __name__ == "__main__":
    pass

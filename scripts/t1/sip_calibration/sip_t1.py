# -*- coding: utf-8 -*-
"""
KALIBRASYON: Si:P donoru 2p0 -> 1s ve 2p+- -> {1s, 2p0} tek-akustik-fonon emisyonu
(intravadi LA/TA, Herring-Vogt deformasyon potansiyeli, anizotrop Christoffel fononlari,
Kohn-Luttinger anizotrop hidrojenik zarflar -- ustel kuyruk, GAUSS DEGIL).

Parametre kaynaklari (rapora bkz.):
  ml, mt        Hensel, Hasegawa, Nakayama, Phys. Rev. 138, A225 (1965) doi:10.1103/PhysRev.138.A225
  eps           11.4 (Faulkner, Phys. Rev. 184, 713 (1969) doi:10.1103/PhysRev.184.713); 11.7 bant ucu
  seviyeler     Vinh+ PNAS 105, 10649 (2008) doi:10.1073/pnas.0802721105 (arXiv:0812.0148) Fig.1
  c11,c12,c44   165.7/63.9/79.6 GPa (oda T; Ioffe/Schaffler 2001 tablosu; ~1% T-bagimliligi)
  rho           2329 kg/m^3
  Xi_u          8.77 eV (arXiv:1408.1666 vb.), bant 8.0-10.5 (arXiv:2104.08998 tablosu)
  Xi_d          5.0 eV (arXiv:1408.1666) / 1.01 eV (arXiv:2104.08998 ab initio) -> bant 1-5 eV
"""
import json, math, os, sys, time
import numpy as np
from scipy.optimize import minimize
from scipy.special import j0, j1, roots_legendre, roots_genlaguerre

HERE = os.path.dirname(os.path.abspath(__file__))
hbar = 1.054571817e-34; e = 1.602176634e-19; m0 = 9.1093837e-31; eps0 = 8.8541878128e-12
aB = 0.529177210903e-10; Ry = 13.605693  # eV

ML, MT = 0.9163, 0.1905
GAM = MT / ML

def units(eps):
    astar = eps * aB / MT                    # m
    Rstar = Ry * MT / eps ** 2 * 1e3         # meV
    return astar, Rstar

# ---------------- variasyonel EMA (Rydberg* birimleri, uzunluk a*) ----------------
# H = -(d_rho^2 + 1/rho d_rho + m^2/rho^2) - GAM d_z^2 - 2/r
NS, NT = 80, 64
xs, ws = roots_genlaguerre(NS, 0.0)      # int_0^inf e^{-x} f(x) dx, x = 2s
ts, wt = roots_legendre(NT)

def energy(a, b, kind):
    s = xs[:, None] / 2.0; ws_ = ws[:, None] / 2.0 * np.exp(0)   # e^{-2s} absorbed: weight e^{-x}
    t = ts[None, :]
    rho = a * s * np.sqrt(1 - t ** 2); z = b * s * t
    r = s * np.sqrt(a ** 2 * (1 - t ** 2) + b ** 2 * t ** 2)
    dsr = rho / (a ** 2 * s); dsz = z / (b ** 2 * s)
    if kind == "1s":
        P, Pr, Pz, m = 1.0, 0.0, 0.0, 0
    elif kind == "2p0":
        P, Pr, Pz, m = z, 0.0, 1.0, 0
    elif kind == "2p1":
        P, Pr, Pz, m = rho, 1.0, 0.0, 1
    # F = P e^{-s};  |F|^2 = P^2 e^{-2s}
    Fr = Pr - P * dsr; Fz = Pz - P * dsz
    kin = Fr ** 2 + GAM * Fz ** 2 + (m ** 2) * P ** 2 / rho ** 2
    pot = -2.0 * P ** 2 / r
    jac = a ** 2 * b * s ** 2 * 2 * np.pi
    W = ws_ * wt[None, :] * jac
    N = np.sum(W * P ** 2 * np.ones_like(t)); K = np.sum(W * kin); V = np.sum(W * pot)
    return (K + V) / N

def solve(kind, x0):
    f = lambda p: energy(abs(p[0]), abs(p[1]), kind)
    r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-10, maxiter=4000))
    return abs(r.x[0]), abs(r.x[1]), r.fun

# ---------------- zarf fonksiyonlari ve form faktorleri ----------------
# grid: rho in [0,R], z in [0,Z] (z-parite ile)
def make_grid(L, N):
    # kok etrafinda sik: x = L*u^2 eslemesi (u uniform) -> kusp iyi cozulur
    u = np.linspace(0, 1, N)
    x = L * u ** 2; w = np.gradient(x)
    w[0] *= 0.5; w[-1] *= 0.5
    w = np.zeros(N); w[1:-1] = (x[2:] - x[:-2]) / 2; w[0] = (x[1] - x[0]) / 2; w[-1] = (x[-1] - x[-2]) / 2
    return x, w

def envelope(kind, a, b, rho, z, gauss=None):
    """gauss=None: KL ustel; gauss=(sr,sz): ayni 2. momentli anizotrop Gauss"""
    R, Z = np.meshgrid(rho, z, indexing="ij")
    if gauss is None:
        s = np.sqrt(R ** 2 / a ** 2 + Z ** 2 / b ** 2); env = np.exp(-s)
    else:
        sr, sz = gauss; env = np.exp(-R ** 2 / (4 * sr ** 2) - Z ** 2 / (4 * sz ** 2))
    if kind == "1s": P = 1.0
    elif kind == "2p0": P = Z
    else: P = R
    return P * env

def normalize(F, rho, wr, wz):
    W = 2 * np.pi * rho[:, None] * wr[:, None] * wz[None, :] * 2.0  # z>=0 ve z<0
    n = np.sum(W * F ** 2)
    return F / math.sqrt(n), W

def moments(F, rho, z, W):
    R, Z = np.meshgrid(rho, z, indexing="ij")
    return math.sqrt(np.sum(W * F ** 2 * R ** 2) / 2), math.sqrt(np.sum(W * F ** 2 * Z ** 2))
    # <x^2> = <rho^2>/2 (her eksen icin sigma_x); <z^2>

def formfactor_table(Fi, ki, Ff, kf, rho, wr, z, wz, qr, qz):
    """|<f|e^{iqr}|i>| (qr,qz) tablosu. m farki ve z-paritesi ile Bessel/trig secimi."""
    mi = 1 if ki == "2p1" else 0; mf = 1 if kf == "2p1" else 0
    dm = abs(mi - mf)
    par_i = -1 if ki == "2p0" else 1; par_f = -1 if kf == "2p0" else 1
    G = Fi * Ff * (2 * np.pi * rho * wr)[:, None] * wz[None, :] * 2.0
    Jm = j0 if dm == 0 else j1
    Bq = Jm(np.outer(qr, rho))                     # (Nqr, Nrho)
    if par_i * par_f > 0: Tz = np.cos(np.outer(z, qz))
    else: Tz = np.sin(np.outer(z, qz))
    return np.abs(Bq @ G @ Tz)                    # (Nqr, Nqz)

# ---------------- fononlar ----------------
def christoffel(c11, c12, c44, rho_m, n):
    n1, n2, n3 = n
    G = np.array([[c11 * n1 ** 2 + c44 * (n2 ** 2 + n3 ** 2), (c12 + c44) * n1 * n2, (c12 + c44) * n1 * n3],
                  [(c12 + c44) * n1 * n2, c11 * n2 ** 2 + c44 * (n1 ** 2 + n3 ** 2), (c12 + c44) * n2 * n3],
                  [(c12 + c44) * n1 * n3, (c12 + c44) * n2 * n3, c11 * n3 ** 2 + c44 * (n1 ** 2 + n2 ** 2)]]) / rho_m
    w, v = np.linalg.eigh(G)
    return np.sqrt(w), v      # v[:,k] polarizasyon

def rate(dE_meV, FFinterp, P):
    """T=0 tek-fonon emisyon hizi (1/s); valley ekseni z; FFinterp(qr,qz)->|F|"""
    Xu, Xd = P["Xu"] * e, P["Xd"] * e
    nth, nph = 48, 48
    ct, wct = roots_legendre(nth)
    ph = np.linspace(0, 2 * np.pi, nph, endpoint=False); wph = 2 * np.pi / nph
    tot = 0.0; parts = np.zeros(3)
    for c, wc in zip(ct, wct):
        st = math.sqrt(1 - c * c)
        for p in ph:
            n = np.array([st * math.cos(p), st * math.sin(p), c])
            v, pol = christoffel(P["c11"], P["c12"], P["c44"], P["rho"], n)
            for k in range(3):
                ek = pol[:, k]
                D = Xd * np.dot(ek, n) + Xu * ek[2] * n[2]
                q = dE_meV * 1e-3 * e / (hbar * v[k])
                F = FFinterp(q * st, q * abs(c))
                g = q ** 3 * D ** 2 * F ** 2 / (8 * np.pi ** 2 * hbar * P["rho"] * v[k] ** 2)
                tot += g * wc * wph; parts[k] += g * wc * wph
    return tot, parts      # parts: k=0 en yavas (TA1), 1 (TA2), 2 (LA)

def main(eps=11.4, Xu=8.77, Xd=5.0, a1_contract=True, N=2400, verbose=True, gauss_mode=False, et_contract=False, cvals=(165.7e9,63.9e9,79.6e9)):
    astar, Rstar = units(eps)
    res = {}
    a1, b1, E1 = solve("1s", [2.4 / 3.17, 1.4 / 3.17])
    a2, b2, E2 = solve("2p0", [3.0, 1.8])
    a3, b3, E3 = solve("2p1", [1.5, 1.0])
    st = {"1s": (a1 * astar, b1 * astar, -E1 * Rstar), "2p0": (a2 * astar, b2 * astar, -E2 * Rstar),
          "2p1": (a3 * astar, b3 * astar, -E3 * Rstar)}
    if verbose:
        print("a*=%.3f nm R*=%.2f meV" % (astar * 1e9, Rstar))
        for k, (a, b, E) in st.items():
            print("  %-4s a=%.3f nm b=%.3f nm  E_bind(var)=%.2f meV" % (k, a * 1e9, b * 1e9, E))
    Lr = 14 * max(st["2p0"][0], st["2p1"][0]); Lz = 14 * max(st["2p0"][1], st["2p1"][1])
    rho, wr = make_grid(Lr, N); z, wz = make_grid(Lz, N)
    E_exp = {"A1": 45.59, "T2": 33.89, "E": 32.58, "2p0": 11.48, "2p1": 6.40}
    env = {}
    for k in ("1s", "2p0", "2p1"):
        a, b, _ = st[k]
        F, W = normalize(envelope(k, a, b, rho, z), rho, wr, wz)
        env[k] = F
    # A1 zarfi: merkez-hucre buzulmesi (kuantum-kusur): kappa ~ sqrt(E) -> a,b * sqrt(E_EMA/E_A1)
    sc = math.sqrt(st["1s"][2] / E_exp["A1"]) if a1_contract else 1.0
    FA1, W = normalize(envelope("1s", st["1s"][0] * sc, st["1s"][1] * sc, rho, z), rho, wr, wz)
    env["A1"] = FA1
    if et_contract:  # 1s(E),1s(T2) icin de kuantum-kusur buzulmesi (ortalama 33.2 meV)
        sc2 = math.sqrt(st["1s"][2] / 33.24)
        env["1s"], _ = normalize(envelope("1s", st["1s"][0] * sc2, st["1s"][1] * sc2, rho, z), rho, wr, wz)
    if gauss_mode:  # her zarfi ayni <x^2>,<z^2> Gauss ile degistir
        for k in list(env.keys()):
            sx, sz = moments(env[k], rho, z, W)
            kind = "1s" if k in ("1s", "A1") else k
            # Gauss envelope*P icin 2. moment eslesmesi: P=1: sigma=sx; P=z: <z^2>=3 sz^2 ; P=rho: <x^2>=2 sr^2
            if kind == "1s": g = (sx, sz)
            elif kind == "2p0": g = (sx, sz / math.sqrt(3))
            else: g = (sx / math.sqrt(2), sz)
            Fg, _ = normalize(envelope(kind, 0, 0, rho, z, gauss=g), rho, wr, wz)
            env[k] = Fg
    P = dict(c11=cvals[0], c12=cvals[1], c44=cvals[2], rho=2329.0, Xu=Xu, Xd=Xd)
    qmax = 1.4e10
    qr = np.linspace(0, qmax, 281); qz = np.linspace(0, qmax, 281)
    from scipy.interpolate import RegularGridInterpolator
    def FF(ki, kf):
        T = formfactor_table(env[ki], ki, env[kf], kf if kf != "A1" else "1s", rho, wr, z, wz, qr, qz)
        itp = RegularGridInterpolator((qr, qz), T, method="cubic")
        return (lambda a_, b_: float(itp([[a_, b_]])[0])), T
    # kanallar: (baslangic, son, dE, vadi agirligi)
    ch = [("2p0", "A1", E_exp["A1"] - E_exp["2p0"], 1 / 6), ("2p0", "1s", E_exp["T2"] - E_exp["2p0"], 3 / 6),
          ("2p0", "1s", E_exp["E"] - E_exp["2p0"], 2 / 6),
          ("2p1", "A1", E_exp["A1"] - E_exp["2p1"], 1 / 6), ("2p1", "1s", E_exp["T2"] - E_exp["2p1"], 3 / 6),
          ("2p1", "1s", E_exp["E"] - E_exp["2p1"], 2 / 6), ("2p1", "2p0", E_exp["2p0"] - E_exp["2p1"], 1.0)]
    names = ["2p0->1s(A1)", "2p0->1s(T2)", "2p0->1s(E)", "2p+-->1s(A1)", "2p+-->1s(T2)", "2p+-->1s(E)", "2p+-->2p0"]
    cache = {}
    out = []
    for (ki, kf, dE, w), nm in zip(ch, names):
        key = (ki, kf)
        if key not in cache: cache[key] = FF(ki, kf)
        g, parts = rate(dE, cache[key][0], P)
        out.append(dict(kanal=nm, dE_meV=dE, agirlik=w, Gamma=g * w, TA=(parts[0] + parts[1]) * w, LA=parts[2] * w))
    G2p0 = sum(o["Gamma"] for o in out[:3]); G2p1 = sum(o["Gamma"] for o in out[3:])
    if verbose:
        for o in out:
            print("  %-14s dE=%5.2f meV  Gamma=%.3e 1/s (LA %.0f%%)  tau_kismi=%.3g ps" % (
                o["kanal"], o["dE_meV"], o["Gamma"], 100 * o["LA"] / o["Gamma"], 1e12 / o["Gamma"]))
        print("  T1(2p0) = %.1f ps   T1(2p+-) = %.1f ps" % (1e12 / G2p0, 1e12 / G2p1))
    # form faktoru ornegi: 2p0->1s q boyunca z ve rho
    res = dict(et_contract=et_contract, eps=eps, Xu=Xu, Xd=Xd, a1_contract=a1_contract, gauss=gauss_mode, N=N,
               states={k: [v[0] * 1e9, v[1] * 1e9, v[2]] for k, v in st.items()},
               kanallar=out, T1_2p0_ps=1e12 / G2p0, T1_2p1_ps=1e12 / G2p1)
    return res, cache, qr, qz

if __name__ == "__main__":
    t0 = time.time()
    r, cache, qr, qz = main()
    print("t=%.0fs" % (time.time() - t0))

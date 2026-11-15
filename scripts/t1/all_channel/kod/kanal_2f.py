# -*- coding: utf-8 -*-
"""
IKI-FONON emisyon hizlari, E -> A1 + 2 fonon, T=0.

(A) ARDISIK (2. mertebe dogrusal tepe): dallar {Ls, La, Ts, Ta}; tepeler DP+PZ+registry (kuplaj.Tepe).
    Gamma = (2pi/hbar)(1/2)(2pi)^-4 sum_{l1,l2} int_0^E21 dE1 q1 q2/(hbar^2 v1 v2) int dphi1 dphi2 |T|^2
    T = a1 a2 sum_m [ V2_0m V1_mi/(E_i-E_m-E1+i eta) + V1_0m V2_mi/(E_i-E_m-E2+i eta) ],  a^2 = hbar/(4 rho1 w)
(B) TEMAS (1. mertebe ikinci-derece tepe), Q = q1+q2 kucuk -> F(Q) bastirilmaz:
    B1 egilme: (D/2) w_l (grad h_l)^2  [geometrik dogrusal-olmayan gerinim]  + (E''/2) dz^2 (nefes)
        dallar ZA(s), ZO'(a: sqrt(ZA^2+wB^2));  C_ss=-D q1.q2, C_aa=-D q1.q2 + 4E'', C_sa=-dw D q1.q2
    B2 duzlem-ici: Phi = -D (q1.q2)(e1.e2) - lam_e (q1.e1)(q2.e2) - mu_e[(q1.q2)(e1.e2)+(q1.e2)(q2.e1)]
        (lam_e, mu_e: DFT ikinci-turev), + registry egriligi (yalniz aa):
        -2 sum_G P_G (g.e1)(g.e2)[F(Q+G)+F(Q-G)]   (d = 2a)
    Ozdes dal ciftleri 1/2 simetri carpani; farkli dallar (s,a) carpansiz (sirali toplam 1/2 ile ayni).
Cikti: data/kanal_2f.json
"""
import json, math, os, sys, time, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from ortak import *
from kuplaj import Tepe
import fonon as fn

ETA = 0.3e-3 * eC
JM = eC * 1e10           # eV/A -> J/m


def gl(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


def dallar_ip(vL, vT, dn):
    mk = {"lin": fn.dogrusal, "sin": fn.sinus}[dn]
    LA, TA = mk("LA", vL), mk("TA", vT)
    return {"Ls": ("L", "s", LA), "La": ("L", "a", gapli(LA, fn.W_SH)),
            "Ts": ("T", "s", TA), "Ta": ("T", "a", gapli(TA, fn.W_SH))}


# ---------------------------------------------------------------- (A) ardisik
def ardisik(k, FT, par, E21, dal, nE=40, nphi=24, regsign=1.0, sadece=None, nst=None):
    nst = nst or len(k.E)
    tp = Tepe(k, FT, par, None)
    Ek = k.E[:nst] * eC
    En, wn = gl(nE, 0.0, E21)
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    tab = {}
    for ad, (pol, br, d) in dal.items():
        if sadece and ad not in sadece:
            continue
        q = d.q_of_w(En / hbar)                       # [nE]
        ok = np.isfinite(q)
        qA = np.where(ok, q, 1e8)[:, None] * 1e-10
        PH = np.broadcast_to(phi, (nE, nphi))
        V = np.zeros((nst, nst, nE, nphi), complex)
        for a in range(nst):
            for b in range(nst):
                V[a, b] = tp.V(a, b, qA, PH, pol, br, ("dp", "pz", "reg"), regsign) * JM
        V[:, :, ~ok, :] = 0.0
        w = np.where(ok, d.w(np.where(ok, q, 1e8)), 1.0)
        amp = np.sqrt(hbar / (4 * RHO1 * w))
        vg = np.where(ok, d.vg(np.where(ok, q, 1e8)), 1.0)
        tab[ad] = (V * amp[None, None, :, None], np.where(ok, q, 0.0), vg)
    G = {}
    for i in (1, 2):
        for l1, l2 in itertools.product(tab.keys(), tab.keys()):
            V1, q1, v1 = tab[l1]; V2, q2, v2 = tab[l2]
            tot = 0.0
            for n in range(nE):
                m2 = nE - 1 - n                       # E2 = E21 - E1 (simetrik GL)
                if q1[n] == 0 or q2[m2] == 0:
                    continue
                d1 = Ek[i] - Ek - En[n] + 1j * ETA     # [m]
                d2 = Ek[i] - Ek - En[m2] + 1j * ETA
                # T[phi1,phi2]
                T = np.einsum("mq,mp->pq", V2[0, :, m2, :] / d1[:, None], V1[:, i, n, :]) \
                    + np.einsum("mp,mq->pq", V1[0, :, n, :] / d2[:, None], V2[:, i, m2, :])
                tot += wn[n] * q1[n] * q2[m2] / (hbar ** 2 * v1[n] * v2[m2]) * np.mean(np.abs(T) ** 2) * (2 * np.pi) ** 2
            G[(i, l1, l2)] = (2 * np.pi / hbar) * 0.5 * tot / (2 * np.pi) ** 4
    # E ortalamasi, dal-cifti tablosu
    out = {}
    for l1, l2 in itertools.product(tab.keys(), tab.keys()):
        out["%s+%s" % (l1, l2)] = 0.5 * (G[(1, l1, l2)] + G[(2, l1, l2)])
    out["TOPLAM"] = sum(out.values())
    return out


# ---------------------------------------------------------------- (B) temas
def temas(k, FT, E21, d1, d2, katsayi, ozdes, nE=80, nphi1=24, ndel=256, regF=None):
    """katsayi(q1v, q2v, e1v, e2v) -> eV/A^2 (skaler alan), regF: opsiyonel ek genlik fonksiyonu
    regF(i, Qx, Qy, e1, e2) [eV/A^2 * F]. Donus: E-ortalamali Gamma [1/s]."""
    En, wn = gl(nE, 0.0, E21)
    q1 = d1.q_of_w(En / hbar); q2 = d2.q_of_w((E21 - En) / hbar)
    ok = np.isfinite(q1) & np.isfinite(q2)
    phi1 = np.linspace(0, 2 * np.pi, nphi1, endpoint=False)
    dl = np.linspace(-np.pi, np.pi, ndel, endpoint=False)
    G = 0.0
    for i in (1, 2):
        tot = 0.0
        for n in np.where(ok)[0]:
            a, b = q1[n], q2[n]
            w1, w2 = d1.w(a), d2.w(b); v1, v2 = d1.vg(a), d2.vg(b)
            amp2 = (hbar / (4 * RHO1 * w1)) * (hbar / (4 * RHO1 * w2))
            P1, DL = np.meshgrid(phi1, dl, indexing="ij")
            P2 = P1 + np.pi + DL
            n1 = np.stack([np.cos(P1), np.sin(P1)], -1); n2 = np.stack([np.cos(P2), np.sin(P2)], -1)
            q1v = a * 1e-10 * n1; q2v = b * 1e-10 * n2
            Q = q1v + q2v
            F = FT(0, i, Q[..., 0], Q[..., 1])
            C = katsayi(q1v, q2v, n1, n2)              # eV/A^2
            M = C * F
            if regF is not None:
                M = M + regF(i, Q[..., 0], Q[..., 1], n1, n2)
            M2 = np.abs(M) ** 2 * (eC * 1e20) ** 2 * amp2
            tot += wn[n] * a * b / (hbar ** 2 * v1 * v2) * np.mean(M2) * (2 * np.pi) ** 2
        G += 0.5 * (2 * np.pi / hbar) * (0.5 if ozdes else 1.0) * tot / (2 * np.pi) ** 4
    return G


def main():
    dpp, _ = dp_parametreleri()
    D = dpp["D_eV"]                          # isaretli (PBE)
    lam_mu = None
    bj = os.path.join(DATA, "beta.json")
    if os.path.exists(bj):
        b = json.load(open(bj)); lam_mu = (b["lam_e"], b["mu_e"])
    elif "d2E_ux" in dpp:
        # biaksiyel: d2E = 2(D + 2lam + 2mu) ; uniaksiyel: d2E = D + lam + 2mu
        A = np.array([[4.0, 4.0], [1.0, 2.0]])
        bb = np.array([dpp["d2E_biax"] - 2 * D, dpp["d2E_ux"] - D])
        lam_e, mu_e = np.linalg.solve(A, bb)
        lam_mu = (float(lam_e), float(mu_e))
    if lam_mu is None:
        lam_mu = (0.0, 0.0)
    print("D=%.3f eV  lam_e=%.2f mu_e=%.2f eV" % (D, lam_mu[0], lam_mu[1]))
    out = {"D_eV": D, "lam_e_eV": lam_mu[0], "mu_e_eV": lam_mu[1], "noktalar": {}}
    secim = sys.argv[1:] or ["ortam_1.09", "3GPa_1.09", "5GPa_1.20"]
    for nokta in secim:
        t0 = time.time()
        k, FT, _ = hazirla(nokta, gauge=False)
        E21 = (k.E[1] - k.E[0]) * eC
        Ez = Ez_turev(nokta)
        rec = {"E21_meV": E21 / eC * 1e3, "E2_AA": Ez["AA"]["E2"]}
        # ---------- (A) dogrulayici tekrari: yalniz DP-LA, tek dal rho1 esdegeri (Ls+La dw=1, ayni dispersiyon)
        for vL, Dv in ((21.0e3, 4.5), (18.86e3, 4.5)):
            LA = fn.dogrusal("LA", vL)
            dal = {"Ls": ("L", "s", LA), "La": ("L", "a", LA)}
            par = {"D": Dv, "e11": 0.0, "eps": 1.0, "dw": 1.0, "d_int": ZAA[nokta], "mstar": 0.77, "beta": 0.0}
            k_reg = k.Plist.copy(); k.Plist = 0 * k.Plist     # registry kapali
            r = ardisik(k, FT, par, E21, dal)
            k.Plist = k_reg
            rec["dogrulayici_tekrar_D%.1f_v%.2f_T1ps" % (Dv, vL / 1e3)] = 1e12 / r["TOPLAM"]
            print("  [%s] dogrulayici-tekrar 2LA D=%.1f v=%.2f: T1=%.1f ps" % (nokta, Dv, vL / 1e3, 1e12 / r["TOPLAM"]), flush=True)
        # ---------- (A) tam ardisik
        ard = {}
        for (vL, vT), dn, e11, eps, dw, rs in [((21.0e3, 13.0e3), "lin", 1.38e-10, 5.10, 1.0, 1.0),
                                              ((21.0e3, 13.0e3), "lin", 1.38e-10, 5.10, 1.0, -1.0),
                                              ((18.86e3, 11.87e3), "lin", 2.91e-10, 2.45, 1.0, 1.0),
                                              ((21.0e3, 13.0e3), "sin", 1.38e-10, 5.10, 1.0, 1.0),
                                              ((21.0e3, 13.0e3), "lin", 1.38e-10, 5.10, 0.0, 1.0)]:
            par = {"D": abs(D), "e11": e11, "eps": eps, "dw": dw, "d_int": ZAA[nokta], "mstar": 0.77, "beta": 0.0}
            r = ardisik(k, FT, par, E21, dallar_ip(vL, vT, dn), regsign=rs)
            key = "v%.2f_%s_e%.2f_eps%.2f_dw%.0f_rs%+.0f" % (vL / 1e3, dn, e11 * 1e10, eps, dw, rs)
            ard[key] = r
            top = sorted(r.items(), key=lambda kv: -kv[1])[:5]
            print("  [%s] ardisik %s: T1=%.1f ps  baskin: %s" % (nokta, key, 1e12 / r["TOPLAM"],
                  ", ".join("%s %.0f%%" % (a, 100 * b / r["TOPLAM"]) for a, b in top if a != "TOPLAM")), flush=True)
        rec["ardisik"] = ard
        # ---------- (B1) egilme temas
        za = fn.za_dal(); zop = fn.za_dal(fn.W_B)
        E2 = Ez["AA"]["E2"]
        tem = {}
        for dw in (1.0, 0.0):
            for Dv in (abs(D), 4.5):
                Ds = -Dv     # isaret: DFT (gerilmede E_c duser)
                c_ss = lambda a, b, n1, n2: -Ds * np.sum(a * b, -1)
                c_aa_g = lambda a, b, n1, n2: -Ds * np.sum(a * b, -1)
                c_aa = lambda a, b, n1, n2: -Ds * np.sum(a * b, -1) + 4 * E2
                c_ee = lambda a, b, n1, n2: 4 * E2 + 0 * a[..., 0]
                c_sa = lambda a, b, n1, n2: -dw * Ds * np.sum(a * b, -1)
                g_ss = temas(k, FT, E21, za, za, c_ss, True)
                g_aa = temas(k, FT, E21, zop, zop, c_aa, True)
                g_aa_g = temas(k, FT, E21, zop, zop, c_aa_g, True)
                g_aa_e = temas(k, FT, E21, zop, zop, c_ee, True)
                g_sa = temas(k, FT, E21, za, zop, c_sa, False) if dw != 0 else 0.0
                key = "D%.2f_dw%.0f" % (Dv, dw)
                tem[key] = {"ZAs+ZAs": g_ss, "ZOp+ZOp(geo+nefes)": g_aa, "ZOp+ZOp_yalniz_geo": g_aa_g,
                            "ZOp+ZOp_yalniz_nefes": g_aa_e, "ZAs+ZOp": g_sa, "TOPLAM": g_ss + g_aa + g_sa}
                print("  [%s] egilme-temas %s: ss %.3g aa %.3g (geo %.3g nefes %.3g) sa %.3g -> T1=%.1f ps" % (
                    nokta, key, 1e12 / max(g_ss, 1e-30), 1e12 / max(g_aa, 1e-30), 1e12 / max(g_aa_g, 1e-30),
                    1e12 / max(g_aa_e, 1e-30), 1e12 / max(g_sa, 1e-30), 1e12 / (g_ss + g_aa + g_sa)), flush=True)
        rec["egilme_temas"] = tem
        # ZA dispersiyon duyarliligi: sin^2 modeli
        zs2 = fn.Dal("ZAsin2", lambda q: fn.ZA_M * np.sin(np.pi * q / (2 * fn.Q_M)) ** 2,
                     lambda q: fn.ZA_M * np.pi / fn.Q_M * np.sin(np.pi * q / (2 * fn.Q_M)) * np.cos(np.pi * q / (2 * fn.Q_M)))
        zs2a = gapli(zs2, fn.W_B)
        Ds = -abs(D)
        g1 = temas(k, FT, E21, zs2, zs2, lambda a, b, n1, n2: -Ds * np.sum(a * b, -1), True)
        g2 = temas(k, FT, E21, zs2a, zs2a, lambda a, b, n1, n2: -Ds * np.sum(a * b, -1) + 4 * E2, True)
        g3 = temas(k, FT, E21, zs2, zs2a, lambda a, b, n1, n2: -Ds * np.sum(a * b, -1), False)
        rec["egilme_temas_sin2"] = {"TOPLAM": g1 + g2 + g3, "ss": g1, "aa": g2, "sa": g3}
        print("  [%s] egilme-temas sin^2-ZA: T1=%.1f ps" % (nokta, 1e12 / (g1 + g2 + g3)), flush=True)
        # ---------- (B2) duzlem-ici temas
        lam_e, mu_e = lam_mu
        th = math.radians(k.theta)
        gv = np.stack([-k.Glist[:, 1], k.Glist[:, 0]], -1) / (2 * math.sin(th / 2))

        def phi_ip(Ds_):
            def f(a, b, e1, e2):
                return (-Ds_ * np.sum(a * b, -1) * np.sum(e1 * e2, -1)
                        - lam_e * np.sum(a * e1, -1) * np.sum(b * e2, -1)
                        - mu_e * (np.sum(a * b, -1) * np.sum(e1 * e2, -1) + np.sum(a * e2, -1) * np.sum(b * e1, -1)))
            return f
        ipt = {}
        for (vL, vT) in ((21.0e3, 13.0e3), (18.86e3, 11.87e3)):
            dal = dallar_ip(vL, vT, "lin")
            Ds = -abs(D)
            res = {}
            for l1, l2 in (("Ls", "Ls"), ("La", "La"), ("Ls", "La"), ("Ts", "Ts"), ("Ta", "Ta"), ("Ts", "Ta"),
                           ("Ls", "Ts"), ("La", "Ta"), ("Ls", "Ta"), ("La", "Ts")):
                p1, b1, d1 = dal[l1]; p2, b2, d2 = dal[l2]
                pol_e = lambda nv, p: nv if p == "L" else np.stack([-nv[..., 1], nv[..., 0]], -1)
                base = phi_ip(Ds)
                fac = 1.0
                if b1 != b2:
                    fac = 1.0  # dw=1
                kat = (lambda a, b, n1, n2, p1=p1, p2=p2: base(a, b, pol_e(n1, p1), pol_e(n2, p2)))
                regF = None
                if b1 == "a" and b2 == "a":
                    def regF(i, Qx, Qy, n1, n2, p1=p1, p2=p2):
                        e1 = pol_e(n1, p1); e2 = pol_e(n2, p2)
                        acc = 0.0
                        for Gv, g, Pv in zip(k.Glist, gv, k.Plist):
                            acc = acc - 2 * Pv * (e1 @ g) * (e2 @ g) * (FT(0, i, Qx + Gv[0], Qy + Gv[1]) + FT(0, i, Qx - Gv[0], Qy - Gv[1]))
                        return acc
                ozd = (l1 == l2)
                g = temas(k, FT, E21, d1, d2, kat, ozd, nE=48, ndel=128, regF=regF)
                res["%s+%s" % (l1, l2)] = g
                if regF is not None:
                    res["%s+%s_yalniz_reg" % (l1, l2)] = temas(k, FT, E21, d1, d2, lambda a, b, n1, n2: 0 * a[..., 0], ozd, nE=48, ndel=128, regF=regF)
            res["TOPLAM"] = sum(v for kk, v in res.items() if "yalniz" not in kk)
            ipt["v%.2f" % (vL / 1e3)] = res
            print("  [%s] duzlem-ici-temas v%.2f: T1=%.1f ps ; reg(La+La)=%.3g ps" % (
                nokta, vL / 1e3, 1e12 / res["TOPLAM"], 1e12 / max(res["La+La_yalniz_reg"], 1e-30)), flush=True)
        rec["duzlem_ici_temas"] = ipt
        out["noktalar"][nokta] = rec
        print("== %s bitti [%.0fs]" % (nokta, time.time() - t0), flush=True)
        json.dump(out, open(os.path.join(DATA, "kanal_2f_%s.json" % "_".join(secim)), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

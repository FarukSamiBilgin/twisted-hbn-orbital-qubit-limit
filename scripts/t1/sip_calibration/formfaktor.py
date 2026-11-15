# Si:P 2p0->1s form faktoru: KL-ustel vs ayni-2.-moment Gauss (analitik) ; log-egim ; hBN dogrulayici F2 log-egimi
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sip_t1 as S
eps = 11.4
astar, Rstar = S.units(eps)
a1, b1, _ = S.solve("1s", [0.75, 0.43]); a2, b2, _ = S.solve("2p0", [1.1, 0.7])
a1 *= astar; b1 *= astar; a2 *= astar; b2 *= astar
rho, wr = S.make_grid(14 * a2, 2400); z, wz = S.make_grid(14 * b2, 2400)
F1, W = S.normalize(S.envelope("1s", a1, b1, rho, z), rho, wr, wz)
F2, _ = S.normalize(S.envelope("2p0", a2, b2, rho, z), rho, wr, wz)
q = np.linspace(1e7, 1.2e10, 600)
Tz = S.formfactor_table(F2, "2p0", F1, "1s", rho, wr, z, wz, np.array([0.0]), q)[0]     # q || z (valley ekseni)
Tr = S.formfactor_table(F2, "2p0", F1, "1s", rho, wr, z, wz, q, np.array([0.0]))[:, 0]  # q _|_ z : 0 (parite) 
# 45 derece
qq = q / math.sqrt(2)
T45 = np.array([S.formfactor_table(F2, "2p0", F1, "1s", rho, wr, z, wz, np.array([x]), np.array([x]))[0, 0] for x in qq[::20]])
# Gauss (ayni <x^2>,<z^2>) analitik
sx1, sz1 = S.moments(F1, rho, z, W); sx2, sz2 = S.moments(F2, rho, z, W)
sz2g = sz2 / math.sqrt(3)
def gpair(s1, s2, k):  # int g1 g2 e^{ikx}, g=(2 pi s^2)^-1/4 exp(-x^2/4s^2)
    S2 = 2 / (1 / s1 ** 2 + 1 / s2 ** 2)          # 1/(2 Sig^2) = 1/4s1^2+1/4s2^2 -> Sig^2 = S2/... 
    Sig2 = 1 / (1 / (2 * s1 ** 2) + 1 / (2 * s2 ** 2))
    pref = (2 * math.pi * s1 ** 2) ** -0.25 * (2 * math.pi * s2 ** 2) ** -0.25 * math.sqrt(2 * math.pi * Sig2)
    return pref * np.exp(-k ** 2 * Sig2 / 2), Sig2
def gz_1s_2p0(k):
    # int g(z;s1) (z/s2) g(z;s2) e^{ikz} dz = (1/s2) * (-i d/dk) [pref exp(-k^2 Sig2/2)] -> |.| = k Sig2/s2 * pref exp(..)
    val, Sig2 = gpair(sz1, sz2g, k)
    return np.abs(k * Sig2 / sz2g * val)
Gz = gpair(sx1, sx2, 0 * q)[0] ** 2 * gz_1s_2p0(q)
# KL analitik yaklasik guc yasasi kontrol: log-egim
def slope(qa, Fa):
    return -np.gradient(np.log(Fa ** 2), np.log(qa))
# fonon q* (LA ~ 8.9 km/s [001], TA 5.84 km/s)
out = {}
for nm, dE in (("1s(E) 21.10", 21.10), ("1s(T2) 22.41", 22.41), ("1s(A1) 34.11", 34.11)):
    for vn, v in (("LA[001]", math.sqrt(165.7e9 / 2329)), ("TA[001]", math.sqrt(79.6e9 / 2329))):
        qs = dE * 1e-3 * S.e / (S.hbar * v)
        i = np.argmin(abs(q - qs))
        out[nm + " " + vn] = dict(q_nm=qs * 1e-9, q_a2=qs * a2, q_b2=qs * b2, F2_KL=float(Tz[i] ** 2),
                                  F2_Gauss=float(Gz[i] ** 2), oran=float(Tz[i] ** 2 / Gz[i] ** 2),
                                  logegim_KL=float(slope(q, Tz)[i]), logegim_G=float(slope(q, Gz)[i]))
        print(nm, vn, {k: ("%.3g" % v) for k, v in out[nm + " " + vn].items()})
print("Gauss sigma: 1s (%.2f,%.2f) nm, 2p0 (%.2f,%.2f) nm" % (sx1 * 1e9, sz1 * 1e9, sx2 * 1e9, sz2g * 1e9))
# Gauss sayisal tablonun analitikle uyumu (dusuk q)
Fg1, _ = S.normalize(S.envelope("1s", 0, 0, rho, z, gauss=(sx1, sz1)), rho, wr, wz)
Fg2, _ = S.normalize(S.envelope("2p0", 0, 0, rho, z, gauss=(sx2, sz2g)), rho, wr, wz)
Tg = S.formfactor_table(Fg2, "2p0", Fg1, "1s", rho, wr, z, wz, np.array([0.0]), q)[0]
for qq_ in (1e9, 2e9, 3e9, 3.6e9):
    i = np.argmin(abs(q - qq_)); print("  q=%.1e  Gauss sayisal %.3e  analitik %.3e" % (q[i], Tg[i], Gz[i]))
json.dump(dict(q=q.tolist(), F_KL_z=Tz.tolist(), F_G_z=Gz.tolist(), qstar=out), open("formfaktor.json", "w"))
# hBN dogrulayicinin F2 tablosu log-egim
H = json.load(open(r"C:\Users\PROFDR~1.SAM\AppData\Local\Temp\claude\C--Users-prof-dr-f-sami-Desktop-BABAM--claude-worktrees-tbg-fotonik-bariyeri-chatimi-35a068\9b6a327a-26b2-469d-978b-55fdc4c53201\scratchpad\pareto_dogrulama\t1_yeniden\data\t1_yeniden.json"))
for x in H:
    qa = np.array(x["F2_q"]["q_invA"]); f2 = np.array(x["F2_q"]["F2_N220"])
    m = f2 > 0
    sl = -np.gradient(np.log(f2[m]), np.log(qa[m]))
    i = np.argmin(abs(qa[m] - x["q_rep_invA"]))
    print("hBN %s xi=%.2f E21=%.1f q_rep=%.3f 1/A  F2=%.3e  logegim n_eff=%.1f" % (x["capa"], x["xi"], x["E21_meV"], x["q_rep_invA"], f2[m][i], sl[i]))

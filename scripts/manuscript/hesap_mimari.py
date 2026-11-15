# -*- coding: utf-8 -*-
"""
Kol E MIMARI makalesi — acik hesaplar (brief H1 + yarisma haritasi + H7 + AB butcesi).

(1) H1 @1.09: ilk 6-8 seviye, tam dipol matrisi, anharmoniklik
    alpha = E32 - E21 (iki f-adayi: 2. E dubleti ve ust A1), mu23 elemanlari,
    p+- yapisi (eleman-basi vs temsil-toplami -> J'de 2x bandi), E1 - V_min,
    seviye bagliligi (W3..W6).
(2) J(theta) vs W1(theta) yarisi: theta in {0.4..0.6} U [1.0..1.4];
    her theta'da cozulmus E21, mu (eleman/toplam), J_XY = mu^2/(4 pi eps0 eps aM^3),
    alpha(theta); W1 ek_tarama2'den (0.6-4.0) + burada 0.4/0.5 eklenir.
(3) H7 derinlestirme: V -> s*V (s=1..8) @1.09: E21(s), uzam, mu(s), lambda(s),
    Gamma_1ph(s) (DP tek-fonon, T=0) ve Gamma_rad(s); hBN'de hedefler
    hbar*Gamma = 1 ueV ve 0.041 ueV (10 MHz) icin gereken s*.
(4) AB/M rezervuar butcesi: registry verisinden E_M profili (global CBM - K),
    Delta_n = E_M - E_n; FE (+-51.3 meV) / Stark karsilastirmasi; t_AM ust siniri
    (W1 < 0.01 meV kistasi) ve t_eff = t_AM^2/Delta.

Cikti: data/mimari_hesap.json
"""
import csv, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KOLE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(KOLE, "preprint"))
from pw_cozucu import Solver, STARS, moire_L, KIN, MSTAR, HBAR2_2ME, A0

DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

fit = json.load(open(os.path.join(KOLE, "results", "data", "v0_fit_2d_K.json")))["cbK_abs"]
P = {1: fit["yildiz1_P_meV"] * 1e-3,
     2: fit["yildiz2_P_meV"] * 1e-3,
     3: fit["yildiz3_P_meV"] * 1e-3}
c0_eV = fit["c0_eV"]

EPS_R = 5.0          # ortam dielektrigi (brief)
E2NM = 1.44          # e^2/(4 pi eps0) [eV nm]
A0_NM = 0.0529177
HBAR_meVps = 0.658211951


def gf_adaptif(theta, pp_eV=0.2407, taban=10.0):
    b = 4 * math.pi / (math.sqrt(3.0) * moire_L(theta))
    return max(taban, 2.6 * math.sqrt(pp_eV / KIN) / b)


def formfaktor_sayisal(s, q0_inv_A, n=201, nang=24):
    """<1|e^{iq r}|2> aci-ortalamali |F|^2, dogrudan dalga fonksiyonlarindan.
    (HO yaklasikligi yerine; sombrero/duz-dip kuyularda fark onemli.)"""
    w, labels, X, Y, psi, dA, c0 = s.real_space_states(nst=2, n=n)
    p0 = psi[..., 0]; p1 = psi[..., 1]
    tot = 0.0
    for a in range(nang):
        phi = math.pi * a / nang
        q = q0_inv_A * np.array([math.cos(phi), math.sin(phi)])
        ph = np.exp(1j * (q[0] * X + q[1] * Y))
        Fq = np.sum(p0.conj() * ph * p1) * dA
        tot += abs(Fq) ** 2
    return tot / nang


def coz(theta, scale=1.0, nst=6, n_grid=201):
    """Olculen potansiyel (x scale) icin seviye + dipol + uzam paketi.
    mu 'eleman' = mu_toplam/sqrt(2): E-dubletinde tekil eleman taban-secimine
    bagli (dejenere donme serbestligi); yalniz temsil-toplami invaryant."""
    Ps = {k: v * scale for k, v in P.items()}
    s = Solver(float(theta), ("measured", Ps),
               gcut_factor=gf_adaptif(theta, 0.2407 * scale))
    w, labels, D, extent = s.dipoles_extent(nst=nst, n=n_grid)
    lv = (np.array(w[:nst]) * 1e3).tolist()
    E21 = lv[1] - lv[0]
    # anharmoniklik adaylari (seviye sirasi: A1, E, E, E2, E2, A1b beklenir)
    alpha_E = (lv[3] - lv[1]) - E21
    alpha_A = (lv[5] - lv[1]) - E21 if nst >= 6 else None
    d01, d02 = float(D[0, 1]), float(D[0, 2])
    mu_tot = math.hypot(d01, d02)
    mu_el = mu_tot / math.sqrt(2.0)
    # mu23: ilk E dubletinden (1,2) ikinci E dubletine (3,4) ve A1b'ye (5)
    mu23_E = math.sqrt(sum(float(D[i, j]) ** 2 for i in (1, 2) for j in (3, 4)) / 2.0)
    mu23_A = math.sqrt((float(D[1, 5]) ** 2 + float(D[2, 5]) ** 2)) if nst >= 6 else None
    _, vmin, vmax = s.well_center()
    return {"theta": float(theta), "scale": scale, "N": s.N,
            "levels_meV": lv, "labels": labels,
            "E21_meV": E21, "alpha_E_meV": alpha_E, "alpha_A1_meV": alpha_A,
            "mu_eleman_a0e": mu_el, "mu_toplam_a0e": mu_tot,
            "mu23_E_a0e": mu23_E, "mu23_A1_a0e": mu23_A,
            "extent_a0": extent, "Vmin_meV": vmin * 1e3, "Vmax_meV": vmax * 1e3,
            "E1_ustu_Vmin_meV": lv[0] - vmin * 1e3,
            "L_nm": s.L / 10.0, "_solver": s}


def J_XY(mu_a0e, aM_nm, eps=EPS_R):
    """Dipolar flip-flop olcegi J = mu^2/(4 pi eps0 eps aM^3) [meV]."""
    mu_enm = mu_a0e * A0_NM
    return E2NM * 1e3 * mu_enm ** 2 / (eps * aM_nm ** 3)


out = {}

# ------------------------------------------------------------------ (1) H1
print("=== H1: 1.09 derece, olculen potansiyel ===")
h1 = coz(1.09, nst=8, n_grid=201)
h1.pop("_solver")
s109 = Solver(1.09, ("measured", P), gcut_factor=12.0)
Wb, _ = s109.bandwidths(nb=8, nseg=10, mesh_n=7)
h1["bandgenislikleri_meV"] = (Wb * 1e3).tolist()
out["H1"] = h1
print(" seviyeler:", [round(x, 2) for x in h1["levels_meV"]])
print(" etiketler:", [l["label"] for l in h1["labels"]])
print(" E21=%.2f  alpha_E=%.2f  alpha_A1=%.2f meV"
      % (h1["E21_meV"], h1["alpha_E_meV"], h1["alpha_A1_meV"]))
print(" mu12 eleman=%.1f toplam=%.1f ; mu23(E2)=%.1f mu23(A1)=%.1f a0e"
      % (h1["mu_eleman_a0e"], h1["mu_toplam_a0e"], h1["mu23_E_a0e"], h1["mu23_A1_a0e"]))
print(" W(1..8) meV:", ["%.2g" % x for x in h1["bandgenislikleri_meV"]])
print(" E1 - Vmin = %.1f meV" % h1["E1_ustu_Vmin_meV"])

# kacak ve blokaj sayilari (Nokta A)
aM_A = h1["L_nm"]
J_el = J_XY(h1["mu_eleman_a0e"], aM_A)
J_tot = J_XY(h1["mu_toplam_a0e"], aM_A)
out["H1"]["J_NoktaA_meV"] = {"eleman": J_el, "toplam": J_tot}
for nm, Jv in (("eleman", J_el), ("toplam", J_tot)):
    lE = (Jv / h1["alpha_E_meV"]) ** 2
    print(" J(%s)=%.3f meV ; kacak (J/alpha_E)^2=%.2e ; t_iSWAP=%.2f ps"
          % (nm, Jv, lE, math.pi * HBAR_meVps / (2 * Jv)))
out["H1"]["kacak_J_alphaE_kare"] = {
    "eleman": (J_el / h1["alpha_E_meV"]) ** 2,
    "toplam": (J_tot / h1["alpha_E_meV"]) ** 2}
# blokaj-CZ: V_B ~ 4J^2/|delta|, delta = alpha
for nm, Jv in (("eleman", J_el), ("toplam", J_tot)):
    VB = 4 * Jv ** 2 / abs(h1["alpha_E_meV"])
    out["H1"].setdefault("blokajCZ", {})[nm] = {
        "V_B_meV": VB, "t_CZ_ps": math.pi * HBAR_meVps / VB}
    print(" blokaj-CZ (%s): V_B=%.3f meV, t_CZ=%.1f ps" % (nm, VB,
          math.pi * HBAR_meVps / VB))

# ------------------------------------------- (2) J(theta) vs W1(theta) yarisi
print("\n=== Yarisma haritasi ===")
ek2 = json.load(open(os.path.join(KOLE, "preprint", "data", "ek_tarama2.json"),
                     encoding="utf-8"))
W1_tab = {round(r["theta"], 2): r["W1_meV"] for r in ek2["measured"]}
race = []
for th in (0.4, 0.5, 0.6, 1.0, 1.09, 1.2, 1.33, 1.4):
    r = coz(th, nst=6, n_grid=181)
    r.pop("_solver")
    if round(th, 2) in W1_tab:
        W1 = W1_tab[round(th, 2)]
    else:
        ss = Solver(th, ("measured", P), gcut_factor=gf_adaptif(th))
        Wb2, _ = ss.bandwidths(nb=2, nseg=8, mesh_n=5)
        W1 = float(Wb2[0] * 1e3)
    Jel = J_XY(r["mu_eleman_a0e"], r["L_nm"])
    Jto = J_XY(r["mu_toplam_a0e"], r["L_nm"])
    row = {"theta": th, "aM_nm": r["L_nm"], "E21_meV": r["E21_meV"],
           "alpha_E_meV": r["alpha_E_meV"],
           "mu_eleman_a0e": r["mu_eleman_a0e"], "mu_toplam_a0e": r["mu_toplam_a0e"],
           "J_eleman_meV": Jel, "J_toplam_meV": Jto, "W1_meV": W1,
           "J_el_over_W1": Jel / max(W1, 1e-300),
           "t_iSWAP_el_ps": math.pi * HBAR_meVps / (2 * Jel),
           "kacak_el": (Jel / r["alpha_E_meV"]) ** 2,
           "J_over_E21": Jto / r["E21_meV"]}
    race.append(row)
    print(" th=%.2f aM=%.1fnm E21=%.1f alpha=%.1f mu=%.0f/%.0f J=%.3g/%.3g meV "
          "W1=%.3g J/W1=%.1e" % (th, r["L_nm"], r["E21_meV"], r["alpha_E_meV"],
          r["mu_eleman_a0e"], r["mu_toplam_a0e"], Jel, Jto, W1, row["J_el_over_W1"]))
out["yarisma"] = race

# ------------------------------------------------------ (3) H7 derinlestirme
print("\n=== H7: Gamma(V0) derinlestirme zinciri (1.09 derece) ===")
hbar = 1.054571817e-34; eC = 1.602176634e-19; eps0 = 8.8541878128e-12
c_l = 2.99792458e8
rho = 7.6e-7; vLA = 19.9e3; D_eV = 4.5


def q0_of(hw_meV):
    return hw_meV * 1e-3 * eC / hbar / vLA   # m^-1


def gamma_tek_fonon_F2(hw_meV, F2):
    q0 = q0_of(hw_meV)
    return (D_eV * eC) ** 2 * q0 ** 2 / (2 * rho * hbar * vLA ** 2) * F2


def F2_HO(hw_meV, l_A):
    x = q0_of(hw_meV) * l_A * 1e-10
    return (x ** 2 / 4) * math.exp(-x ** 2 / 2), x


def gamma_rad(hw_meV, d_a0e):
    dE_J = hw_meV * 1e-3 * eC
    om = dE_J / hbar
    d_Cm = d_a0e * A0 * 1e-10 * eC
    return d_Cm ** 2 * om ** 3 / (3 * math.pi * eps0 * hbar * c_l ** 3)


deep = []
for sc in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
    r = coz(1.09, scale=sc, nst=6, n_grid=201)
    sol = r.pop("_solver")
    q0_A = q0_of(r["E21_meV"]) * 1e-10            # 1/Angstrom
    F2n = formfaktor_sayisal(sol, q0_A, n=201, nang=24)
    F2h, q0l = F2_HO(r["E21_meV"], r["extent_a0"] * A0)
    G1 = gamma_tek_fonon_F2(r["E21_meV"], F2n)
    Gr = gamma_rad(r["E21_meV"], r["mu_toplam_a0e"])
    Jto = J_XY(r["mu_toplam_a0e"], r["L_nm"])
    row = {"scale": sc, "V0_esdeger_meV": 40.04 * sc,
           "E21_meV": r["E21_meV"], "alpha_E_meV": r["alpha_E_meV"],
           "lambda_um": 1239.842 / r["E21_meV"],
           "extent_a0": r["extent_a0"],
           "mu_toplam_a0e": r["mu_toplam_a0e"],
           "q0l_HO": q0l, "F2_sayisal": F2n, "F2_HO": F2h,
           "G1ph_s1": G1, "hbarG1ph_ueV": G1 * hbar / eC * 1e6,
           "tau1ph_ns": 1e9 / G1,
           "Grad_s1": Gr, "taurad_us": 1e6 / Gr,
           "J_toplam_meV": Jto, "J_eleman_meV": Jto / math.sqrt(2),
           "J_over_hbarG": Jto * 1e-3 / (G1 * hbar / eC)}
    deep.append(row)
    print(" s=%.1f V0eq=%.0f E21=%.1f lam=%.1fum l=%.1fa0 mu=%.0f "
          "F2num=%.3g (HO %.3g) hbarG=%.3g ueV tau=%.3g ns J=%.2f J/hG=%.1e"
          % (sc, row["V0_esdeger_meV"], row["E21_meV"], row["lambda_um"],
             row["extent_a0"], row["mu_toplam_a0e"], F2n, F2h,
             row["hbarG1ph_ueV"], row["tau1ph_ns"], Jto, row["J_over_hbarG"]))
out["derinlestirme"] = deep

# hedef kesisimler (log-interpolasyon)
xs = [r["scale"] for r in deep]
ys = [math.log(r["hbarG1ph_ueV"]) for r in deep]


def s_star(hedef_ueV):
    t = math.log(hedef_ueV)
    for i in range(len(xs) - 1):
        if (ys[i] - t) * (ys[i + 1] - t) <= 0:
            f = (t - ys[i]) / (ys[i + 1] - ys[i])
            return xs[i] + f * (xs[i + 1] - xs[i])
    return None


out["hedefler"] = {"s_1ueV": s_star(1.0), "s_10MHz_0.041ueV": s_star(0.0411)}
print(" hedef: hbarG=1 ueV -> s=%.2f ; 10 MHz (0.041 ueV) -> s=%s"
      % (out["hedefler"]["s_1ueV"] or -1,
         "%.2f" % out["hedefler"]["s_10MHz_0.041ueV"]
         if out["hedefler"]["s_10MHz_0.041ueV"] else ">8"))

# ------------------------------------------------- (4) AB/M rezervuar butcesi
print("\n=== AB/M rezervuar butcesi ===")
rows = [r for r in csv.DictReader(open(os.path.join(
    KOLE, "results", "dft_raw", "registry_scan.csv"))) if r["status"] == "ok"]
K = json.load(open(os.path.join(KOLE, "results", "dft_raw", "K_edges.json")))
prof = []
for r in rows:
    s = float(r["s"]); tag = "s%03d" % round(s * 100)
    kabs = K[tag]["cbK"] - K[tag]["Vvac_mean"]
    gabs = float(r["CBM_abs_eV"])
    prof.append({"s": s, "K_abs_eV": kabs, "global_abs_eV": gabs,
                 "M_altinda_meV": (gabs - kabs) * 1e3})
E1_abs = c0_eV + h1["levels_meV"][0] * 1e-3
E2_abs = c0_eV + h1["levels_meV"][1] * 1e-3
E3_abs = c0_eV + h1["levels_meV"][3] * 1e-3
EM_AB = next(p["global_abs_eV"] for p in prof if abs(p["s"] - 1.0) < 1e-6)
EM_min = min(p["global_abs_eV"] for p in prof if p["M_altinda_meV"] < -1)
butce = {
    "E_M_AB_abs_eV": EM_AB, "E_M_halka_min_abs_eV": EM_min,
    "M_K_farki_AB_meV": (EM_AB - (K["s100"]["cbK"] - K["s100"]["Vvac_mean"])) * 1e3,
    "Delta1_meV": (EM_AB - E1_abs) * 1e3,
    "Delta2_meV": (EM_AB - E2_abs) * 1e3,
    "Delta3_meV": (EM_AB - E3_abs) * 1e3,
    "Delta1_halka_meV": (EM_min - E1_abs) * 1e3,
    "FE_adim_meV": 51.3,
    "profil": prof,
}
# t_AM ust siniri ve koprunun dispersif olcegi
t_AM_max = 0.11  # meV (W1<0.01 meV kistasindan, brief)
butce["t_AM_max_meV"] = t_AM_max
butce["t_eff_max_ueV"] = t_AM_max ** 2 / (butce["Delta2_meV"] * 1e-3) * 1e3 * 1e-3 * 1e6
# t_eff = t^2/Delta: (0.11 meV)^2 / 172 meV = 7.0e-5 meV = 0.07 ueV
butce["t_eff_max_ueV"] = (t_AM_max ** 2 / butce["Delta2_meV"]) * 1e3  # ueV
# V_zz dispersif kanal (aM/sqrt3)
butce["Vzz_aM_sqrt3_meV"] = {
    "eleman": J_el * 3 ** 1.5, "toplam": J_tot * 3 ** 1.5}
out["AB_butce"] = butce
print(" E_M(AB)=%.4f eV (K'nin %.0f meV alti); halka min %.4f eV"
      % (EM_AB, butce["M_K_farki_AB_meV"], EM_min))
print(" Delta1=%.0f Delta2=%.0f Delta3=%.0f meV (halka Delta1=%.0f)"
      % (butce["Delta1_meV"], butce["Delta2_meV"], butce["Delta3_meV"],
         butce["Delta1_halka_meV"]))
print(" FE adimi 51.3 meV -> rezonant-shuttle TEK BASINA YETMEZ")
print(" t_AM<=%.2f meV -> t_eff<=%.3f ueV (Delta2) ; V_zz(aM/sqrt3)=%.2f-%.2f meV"
      % (t_AM_max, butce["t_eff_max_ueV"],
         butce["Vzz_aM_sqrt3_meV"]["eleman"], butce["Vzz_aM_sqrt3_meV"]["toplam"]))

with open(os.path.join(DATA, "mimari_hesap.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\n->", os.path.join(DATA, "mimari_hesap.json"))

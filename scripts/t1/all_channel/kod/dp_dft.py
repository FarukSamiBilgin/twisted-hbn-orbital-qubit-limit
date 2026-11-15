#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hBN tek-katman iletim-bandi (K) DEFORMASYON POTANSIYELI + yalanci-ayar (gauge) kaymasi.
PBE, PAW psl 1.0.0 (Kol E ile ayni pseudolar, sadece okunur), vakum-hizali.

Gerinimler (u_ij): 0; biaksiyel +-0.5%, +-1%; uniaksiyel u_xx=+-1%; saf kayma u_xy=1%.
Her biri: scf (12x12x1) -> pp.x V(z) (vakum) -> nscf K etrafinda 5x5 k-agi
-> paraboloid fit: CB minimum enerjisi E_c - V_vac ve minimum konumu (gauge kaymasi).
D = dE_c/d(u_xx+u_yy); kayma bileseni ve beta = gauge buyuklugu.
WSL m2env, ~/miniforge3/bin (QE 7.5) ile calisir.
"""
import os, re, subprocess, json, shutil, sys
import numpy as np

BOHR = 0.529177210903; RY = 13.605693122994
A_LAT = 2.51; C_VAC = 20.0
PSEUDO = os.path.expanduser("~/kolE_dft/pseudo")
QE = os.path.expanduser("~/miniforge3/bin")
BASE = os.path.expanduser("~/t1hakem_dp")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "dp_sonuc.json")
NP = 8
os.makedirs(BASE, exist_ok=True)

STRAINS = {
    "e0": (0.0, 0.0, 0.0),
    "bi+0.5": (0.005, 0.005, 0.0), "bi-0.5": (-0.005, -0.005, 0.0),
    "bi+1": (0.01, 0.01, 0.0), "bi-1": (-0.01, -0.01, 0.0),
    "ux+1": (0.01, 0.0, 0.0), "ux-1": (-0.01, 0.0, 0.0),
    "sxy+1": (0.0, 0.0, 0.01),
}


def cell(uxx, uyy, uxy):
    a1 = np.array([1.0, 0.0]) * A_LAT
    a2 = np.array([-0.5, np.sqrt(3) / 2]) * A_LAT
    M = np.array([[1 + uxx, uxy], [uxy, 1 + uyy]])
    return M @ a1, M @ a2


def inp(calc, pref, uxx, uyy, uxy, kblock, nbnd=8):
    a1, a2 = cell(uxx, uyy, uxy)
    s = ["&CONTROL", "  calculation='%s', prefix='%s', outdir='./tmp'," % (calc, pref),
         "  pseudo_dir='%s', verbosity='high'," % PSEUDO, "/",
         "&SYSTEM", "  ibrav=0, nat=2, ntyp=2, ecutwfc=60.0, ecutrho=480.0,",
         "  nbnd=%d, occupations='fixed'," % nbnd, "/",
         "&ELECTRONS", "  conv_thr=1e-10, mixing_beta=0.4,", "/",
         "ATOMIC_SPECIES", " B 10.811 B.pbe-n-kjpaw_psl.1.0.0.UPF",
         " N 14.007 N.pbe-n-kjpaw_psl.1.0.0.UPF",
         "CELL_PARAMETERS angstrom",
         " %.10f %.10f 0.0" % tuple(a1), " %.10f %.10f 0.0" % tuple(a2),
         " 0.0 0.0 %.6f" % C_VAC,
         "ATOMIC_POSITIONS crystal",
         " B 0.0 0.0 0.5", " N 0.3333333333 0.6666666667 0.5"]
    return "\n".join(s) + "\n" + kblock


def run(exe, cwd, infile, outfile):
    # MPI baslatma asilmasi goruldu (ux+1) -> MPI'siz, OpenMP 6 is parcacigi
    env = dict(os.environ, OMP_NUM_THREADS="6")
    cmd = [os.path.join(QE, exe)]
    with open(os.path.join(cwd, infile)) as fi, open(os.path.join(cwd, outfile), "w") as fo:
        subprocess.run(cmd, stdin=fi, stdout=fo, stderr=subprocess.STDOUT, cwd=cwd,
                       env=env, timeout=1500)
    return open(os.path.join(cwd, outfile), errors="ignore").read()


BANDS_RE = re.compile(r"k =\s*([-\d.]+)\s*([-\d.]+)\s*([-\d.]+).*?bands \(ev\):\s*\n\n((?:\s+[-\d.\s]+\n)+)", re.S)


def vac_level(cwd, pref):
    with open(os.path.join(cwd, "pp.in"), "w") as f:
        f.write("&INPUTPP\n prefix='%s', outdir='./tmp', filplot='p.pot', plot_num=11,\n/\n"
                "&PLOT\n iflag=3, output_format=6, fileout='p.cube',\n/\n" % pref)
    run("pp.x", cwd, "pp.in", "pp.out")
    with open(os.path.join(cwd, "p.cube")) as f:
        f.readline(); f.readline()
        nat = int(f.readline().split()[0])
        n = [int(f.readline().split()[0]) for _ in range(3)]
        for _ in range(abs(nat)):
            f.readline()
        d = np.array(f.read().split(), float)[:n[0] * n[1] * n[2]].reshape(n)
    vz = d.mean(axis=(0, 1)) * RY
    z = np.arange(n[2]) / n[2]
    os.remove(os.path.join(cwd, "p.cube"))
    return float(vz[(z < 0.05) | (z > 0.95)].mean()), float(vz[(z < 0.05) | (z > 0.95)].std())


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
DK = 0.012  # kristal birimi
grid = [(i, j) for i in (-2, -1, 0, 1, 2) for j in (-2, -1, 0, 1, 2)]
for tag, (uxx, uyy, uxy) in STRAINS.items():
    if tag in res:
        continue
    cwd = os.path.join(BASE, tag); os.makedirs(cwd, exist_ok=True)
    kscf = "K_POINTS automatic\n 12 12 1 0 0 0\n"
    open(os.path.join(cwd, "scf.in"), "w").write(inp("scf", "h", uxx, uyy, uxy, kscf))
    t = run("pw.x", cwd, "scf.in", "scf.out")
    assert "convergence has been achieved" in t, tag
    vv, vstd = vac_level(cwd, "h")
    kl = ["K_POINTS crystal", str(len(grid))]
    for (i, j) in grid:
        kl.append(" %.8f %.8f 0.0 1" % (1 / 3 + i * DK, 1 / 3 + j * DK))
    open(os.path.join(cwd, "nscf.in"), "w").write(
        inp("bands", "h", uxx, uyy, uxy, "\n".join(kl) + "\n", nbnd=8))
    t = run("pw.x", cwd, "nscf.in", "nscf.out")
    bk = [((float(m.group(1)), float(m.group(2))), [float(x) for x in m.group(4).split()])
          for m in BANDS_RE.finditer(t)]
    # QE k: kartezyen 2pi/alat (alat = |a1| ilk satir) -> 1/A
    alat = float(re.search(r"lattice parameter \(alat\)\s*=\s*([\d.]+)", t).group(1)) * BOHR
    K = np.array([b[0] for b in bk]) * 2 * np.pi / alat
    Ec = np.array([b[1][4] for b in bk]) - vv
    Ev = np.array([b[1][3] for b in bk]) - vv
    # paraboloid fit E = c0 + c1 x + c2 y + c3 x^2 + c4 y^2 + c5 xy
    X, Y = K[:, 0], K[:, 1]
    A = np.column_stack([np.ones_like(X), X, Y, X ** 2, Y ** 2, X * Y])
    out = {}
    for nm, E in (("c", Ec), ("v", Ev)):
        c, *_ = np.linalg.lstsq(A, E, rcond=None)
        H = np.array([[2 * c[3], c[5]], [c[5], 2 * c[4]]])
        k0 = -np.linalg.solve(H, c[1:3])
        Emin = c[0] + 0.5 * c[1:3] @ k0
        out[nm] = {"E_ext_eV": float(Emin), "k_ext_invA": k0.tolist(),
                   "hess_eVA2": H.tolist(), "rms_meV": float(np.std(A @ c - E) * 1e3)}
    i0 = grid.index((0, 0))
    res[tag] = {"u": [uxx, uyy, uxy], "Vvac_eV": vv, "Vvac_std": vstd,
                "Ec_corner_eV": float(Ec[i0]), "Ev_corner_eV": float(Ev[i0]),
                "K_corner_invA": K[i0].tolist(), "fit": out}
    print(tag, "Ec(K)-Vvac=%.5f  Ec_min=%.5f  gap=%.4f  vac_std=%.4f" %
          (Ec[i0], out["c"]["E_ext_eV"], Ec[i0] - Ev[i0], vstd), flush=True)
    shutil.rmtree(os.path.join(cwd, "tmp"), ignore_errors=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("TAMAM")

# -*- coding: utf-8 -*-
"""nscf K-cevresi 5x5 agdan CB minimumu (enerji, konum): gauge beta ve 2. mertebe DP.
k noktalari GIRDI kristal koordinatlarindan (gerilmis ters orgu) hesaplanir."""
import re, json, os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
A_LAT = 2.51; DK = 0.012; A_CC = A_LAT/np.sqrt(3)
grid = [(i, j) for i in (-2,-1,0,1,2) for j in (-2,-1,0,1,2)]
dp = json.load(open(os.path.join(D, "dp_sonuc.json")))
R = re.compile(r"bands \(ev\):\s*\n\n((?:\s+[-\d.\s]+\n)+)")
def cell(u):
    uxx,uyy,uxy=u; M=np.array([[1+uxx,uxy],[uxy,1+uyy]])
    a1=M@np.array([1,0.])*A_LAT; a2=M@np.array([-.5,np.sqrt(3)/2])*A_LAT
    A=np.array([a1,a2]); B=2*np.pi*np.linalg.inv(A).T
    return B
res={}
for t in ("e0","ux+1","ux-1","sxy+1"):
    txt=open(os.path.join(D,t+"_nscf.out")).read()
    E=np.array([[float(x) for x in m.group(1).split()][4] for m in R.finditer(txt)])-dp[t]["Vvac_eV"]
    B=cell(dp[t]["u"])
    K=np.array([(1/3+i*DK)*B[0]+(1/3+j*DK)*B[1] for i,j in grid]); Kc=(B[0]+B[1])/3
    X,Y=(K-Kc).T
    # izotrop paraboloid + dogrusal: E = c0 + c1 x + c2 y + c3 (x^2+y^2)  (merkeze yakin 13 nokta)
    sel=np.array([abs(i)+abs(j)<=2 for i,j in grid])
    A=np.column_stack([np.ones_like(X),X,Y,X**2+Y**2])[sel]
    c,*_=np.linalg.lstsq(A,E[sel],rcond=None)
    k0=-c[1:3]/(2*c[3]); Emin=c[0]-(c[1]**2+c[2]**2)/(4*c[3])
    res[t]={"Emin":Emin,"k0":k0.tolist(),"mstar":3.80998/c[3],"rms_meV":float(np.std(A@c-E[sel])*1e3),"Ecorner":float(E[12])}
    print(t,"Emin=%.5f Ecorner=%.5f k0=(%.4f,%.4f)/A m*=%.3f rms=%.2f meV"%(Emin,E[12],k0[0],k0[1],3.80998/c[3],res[t]["rms_meV"]))
# beta: |dk| = beta/(2 a_cc) * |(u_xx-u_yy, -2u_xy)|
dk_s=np.linalg.norm(np.array(res["sxy+1"]["k0"])-np.array(res["e0"]["k0"]))
dk_u=0.5*np.linalg.norm(np.array(res["ux+1"]["k0"])-np.array(res["ux-1"]["k0"]))
beta_s=dk_s*2*A_CC/0.02; beta_u=dk_u*2*A_CC/0.01
dbi={k:v["Ec_corner_eV"] for k,v in dp.items()}
Dv=(dbi["bi+1"]-dbi["bi-1"])/0.04
c2bi=(dbi["bi+1"]+dbi["bi-1"]-2*dbi["e0"])/1e-4
c2ux=(res["ux+1"]["Emin"]+res["ux-1"]["Emin"]-2*res["e0"]["Emin"])/1e-4
c2ux_c=(res["ux+1"]["Ecorner"]+res["ux-1"]["Ecorner"]-2*res["e0"]["Ecorner"])/1e-4
lam=(c2bi/2-Dv)-(c2ux-Dv); mu=(c2ux-Dv-lam)/2
out={"beta_kayma":beta_s,"beta_uniaks":beta_u,"beta":0.5*(beta_s+beta_u),"D_eV":Dv,"dE_ux_min":(res["ux+1"]["Emin"]-res["ux-1"]["Emin"])/0.02,
     "d2E_biax":c2bi,"d2E_ux_min":c2ux,"d2E_ux_kose":c2ux_c,"lam_e":lam,"mu_e":mu,"fit":res}
print(json.dumps({k:v for k,v in out.items() if k!="fit"},indent=1))
json.dump(out,open(os.path.join(D,"beta.json"),"w"),indent=1)

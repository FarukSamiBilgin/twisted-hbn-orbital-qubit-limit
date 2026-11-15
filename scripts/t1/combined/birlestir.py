import json, numpy as np
hbar=6.582119569e-16 # eV s
# Iki hattin kanallari (ps); ortusen kanallar (1f, LA+LA/ardisik, ZA cifti) cift sayilmadan birlestirildi.
# opt = en uzun T1 (iyimser), pes = en kisa, mer = merkez
P={ # E21: kanal: (pes, mer, opt)  [ps]
 32:{'1f':(7,12,117),'LALA':(1.9,15,134),'egilme':(37,53,272),'temas_ip':(6500,6500,6500)},
 48:{'1f':(53,85,1150),'LALA':(4.4,28,121),'egilme':(23,25,134),'temas_ip':(900,1200,1200)},
 64:{'1f':(550,870,21000),'LALA':(13,25.5,162),'egilme':(11.8,11.8,47),'temas_ip':(340,490,490)},
}
out={}
for E,ch in P.items():
  r={k:1/np.array(v,float) for k,v in ch.items()}
  tot=sum(r.values()); T=1/tot
  dom={lab:max(r,key=lambda k:r[k][i]) for i,lab in enumerate(['pes','mer','opt'])}
  out[E]=dict(T1_ps=dict(pes=round(T[0],1),mer=round(T[1],1),opt=round(T[2],1)),
              hGamma_ueV=dict(mer=round(hbar/(T[1]*1e-12)*1e6,1)),
              eksik_kat_30ns=dict(mer=round(30000/T[1]),opt=round(30000/T[2])),baskin=dom)
# Kaldirac: 88.4 meV'de LA+LA 0.098-0.231 ns (D 4.5 / 3.66). D^4 olcegi.
lev={}
for T1,D in [(0.098,4.5),(0.231,3.66)]:
  f=30/T1; lev[f"{T1}ns@D={D}"]=dict(kat=round(f),D_gerekli_eV=round(D/f**0.25,2))
# LA+LA fonon dalga boylari
for E in (32.4,48.2,64.4):
  q=E*1e-3/(hbar*2.1e4); lev[f"lambda_LA+LA_{E}meV_nm"]=round(2*np.pi/(q/2)*1e9,2)
out['kaldirac']=lev
print(json.dumps(out,indent=1,ensure_ascii=False))
json.dump(out,open(__file__.replace('birlestir.py','hukum.json'),'w'),indent=1,ensure_ascii=False)

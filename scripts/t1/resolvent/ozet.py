import json, glob, os, numpy as np
ns=lambda G: 1e9/G if G>0 else float('inf')
runs={}
for f in glob.glob("data/run_*_Ne24_nphi24_h0.75_eta0.30_LT.json"):
    d=json.load(open(f)); runs.setdefault(d["nokta"],{})[round(d["vLA"])]=d
za=json.load(open("data/za_za.json"))
OUT={}
for ad in ("ortam_0GPa_1.09","calisma_3GPa_1.09","5GPa_1.20_uzatma"):
    R=runs[ad]; E=list(R.values())[0]["E21_meV"]
    print("\n=== %s E21=%.2f ===" % (ad,E))
    res={}
    def band(fn):
        vals=[fn(d,D) for d in R.values() for D in ("3.66","4.50")]
        return min(vals),max(vals)
    res["1f_LA_def"]=band(lambda d,D: ns(d["tek"]["D%s_gA0.000_off0"%D]["G_LA"]))
    res["1f_LA_def+gauge"]=band(lambda d,D: min(ns(d["tek"]["D%s_gA%s_off%d"%(D,g,o)]["G_LA"]) for g in ("0.339","0.692") for o in (0,1)))
    res["1f_TA_gauge"]=band(lambda d,D: min(ns(d["tek"]["D%s_gA%s_off%d"%(D,g,o)]["G_TA"]) for g in ("0.339","0.692") for o in (0,1)))
    res["LA+LA_def"]=band(lambda d,D: ns(d["iki"]["D%s_gA0.000_off0"%D]["G_LL"]))
    res["LA+LA_def+gauge(min)"]=band(lambda d,D: min(ns(d["iki"]["D%s_gA%s_off%d"%(D,g,o)]["G_LL"]) for g in ("0.339","0.692") for o in (0,1)))
    res["LA+TA_gauge_b1.7"]=band(lambda d,D: min(ns(d["iki"]["D%s_gA0.339_off%d"%(D,o)]["G_LT"]) for o in (0,1)))
    res["LA+TA_gauge_ust"]=band(lambda d,D: min(ns(d["iki"]["D%s_gA0.692_off%d"%(D,o)]["G_LT"]) for o in (0,1)))
    res["TA+TA_gauge_b1.7"]=band(lambda d,D: min(ns(d["iki"]["D%s_gA0.339_off%d"%(D,o)]["G_TT"]) for o in (0,1)))
    res["TA+TA_gauge_ust"]=band(lambda d,D: min(ns(d["iki"]["D%s_gA0.692_off%d"%(D,o)]["G_TT"]) for o in (0,1)))
    z=za[ad]["sonuc"]
    def zb(ekran, models=("sin2_k1.00","kq2_k1.29","kq2_k0.86")):
        v=[z[k]["T1_ns"] for k in z if k.split("|")[2]==ekran and k.split("|")[0] in models and k.endswith("gA0.000") and z[k]["T1_ns"]]
        return (min(v),max(v)) if v else (None,None)
    res["ZA+ZA_dinamik(kutup haric)"]=zb("dinamik"); res["ZA+ZA_dinamik_kutuplu"]=zb("dinamik_kutuplu")
    res["ZA+ZA_perdelemesiz"]=zb("yok"); res["ZA+ZA_statik"]=zb("statik")
    res["ZA+ZA_dinamik_sin2"]=zb("dinamik",("sin2_k1.00",)); res["ZA+ZA_dinamik_kq2"]=zb("dinamik",("kq2_k1.29","kq2_k0.86"))
    gz=[k for k in z if k.endswith("gA0.339")]
    res["ZA+ZA_din_ayar_etkisi"]=[(k, z[k]["T1_ns"], z[k.replace("gA0.339","gA0.000")]["T1_ns"]) for k in gz]
    # merkez (D=4.5, v orta)
    d=R[19940]
    cen={"1f":ns(d["tek"]["D4.50_gA0.000_off0"]["G_LA"]),"LL":ns(d["iki"]["D4.50_gA0.000_off0"]["G_LL"]),
         "LT":ns(d["iki"]["D4.50_gA0.339_off0"]["G_LT"]),"TT":ns(d["iki"]["D4.50_gA0.339_off0"]["G_TT"]),
         "ZZ":z["sin2_k1.00|v1|dinamik|D4.50|gA0.000"]["T1_ns"],"ZZ_kq2":z["kq2_k1.29|v1|dinamik|D4.50|gA0.000"]["T1_ns"]}
    G=sum(1/cen[k] for k in ("1f","LL","LT","TT","ZZ") if cen[k])
    cen["toplam"]=1/G
    res["merkez_D4.5_vorta"]=cen
    # toplam bant: min/max kombinasyonlari kaba: her kanalin bandini topla
    lo=1/sum(1/res[k][0] for k in ("1f_LA_def","LA+LA_def","ZA+ZA_dinamik(kutup haric)"))
    hi=1/sum(1/res[k][1] for k in ("1f_LA_def","LA+LA_def","ZA+ZA_dinamik(kutup haric)"))
    res["toplam_bant_ns"]=(lo,hi)
    for k,v in res.items():
        if k=="ZA+ZA_din_ayar_etkisi": continue
        print(" %-30s %s"%(k, v if isinstance(v,dict) else "%.4g - %.4g ns"%v if v[0] else v))
    # LL enerji dagilimi
    dd=np.array([(x[1],x[2]) for x in d["LL_dagilim"]]); 
    OUT[ad]={"E21_meV":E,**{k:v for k,v in res.items()}}
    print(" ZA bilgi:", {m:(round(i["qh_range"][0],3),round(i["qh_range"][1],3), round(i["ql_range"][0],3),round(i["ql_range"][1],3)) for m,i in za[ad].get("bilgi",{}).items()})
    b=za[ad]["bilgi"]["sin2_k1.00"]; h=np.array(b["K_dagilimi_pay"]); kc=np.array(b["K_kenar"])
    print(" ZA K dagilimi: K<0.1: %.2f, 0.1-0.2: %.2f, 0.2-0.4: %.2f, >0.4: %.2f; q0_LA=%.3f"%(h[kc[:-1]<0.1].sum(),h[(kc[:-1]>=0.1)&(kc[:-1]<0.2)].sum(),h[(kc[:-1]>=0.2)&(kc[:-1]<0.4)].sum(),h[kc[:-1]>=0.4].sum(),b["q0_LA"]))
for ad in ("dz04_xi0.75_1.09","dz04_xi1.00_1.20"):
    fs=glob.glob("data/run_%s_v*_Ne24_nphi24_h0.75_eta0.30_L.json"%ad)
    vals=[]
    for f in fs:
        d=json.load(open(f))
        for D in ("3.66","4.50"):
            vals.append((round(d["vLA"]),D,ns(d["iki"]["D%s_gA0.000_off0"%D]["G_LL"]),ns(d["tek"]["D%s_gA0.000_off0"%D]["G_LA"])))
    z=za[ad]["sonuc"]; zv=[z[k]["T1_ns"] for k in z if "|dinamik|" in k and k.endswith("gA0.000") and z[k]["T1_ns"]]
    print("\n%s E21=%.2f LL/1f:"%(ad,d["E21_meV"]), vals, " ZZ din:", (min(zv),max(zv)) if zv else None)
    OUT[ad]={"E21_meV":d["E21_meV"],"LL_1f":vals,"ZZ_din":(min(zv),max(zv)) if zv else None}
json.dump(OUT,open("data/ozet.json","w"),indent=1,default=str)

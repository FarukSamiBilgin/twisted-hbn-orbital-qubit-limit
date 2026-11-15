# -*- coding: utf-8 -*-
import json, os, numpy as np
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
k1 = json.load(open(os.path.join(D, "kanal_1f.json")))
oz = {}
for nk in ("ortam_1.09", "3GPa_1.09", "5GPa_1.20"):
    r1 = k1["noktalar"][nk]; t = r1["tablo"]
    tops = [v["TOPLAM_max"] for v in t.values()]
    c = t["vL21.00_lin_D5.94_e1.38_eps5.10_dw1"]
    kan = {}
    for name, keys in (("LA-DP", ["Ls_dp", "La_dp"]), ("LA-PZ", ["Ls_pz", "La_pz"]), ("TA-PZ", ["Ts_pz", "Ta_pz"]),
                       ("LA-gauge", ["Ls_g", "La_g"]), ("TA-gauge", ["Ts_g", "Ta_g"]), ("LA-registry", ["La_reg"]), ("TA-registry", ["Ta_reg"])):
        G = sum(c[k] for k in keys); kan[name] = G
        # bant
        Gs = [sum(v[k] for k in keys) for v in t.values()]
        kan[name + "_bant"] = [min(Gs), max(Gs)]
    kan["1f_TOPLAM"] = c["TOPLAM_max"]; kan["1f_TOPLAM_bant"] = [min(tops), max(tops)]
    kan["ZOp_1f"] = r1["ZOp_nefes_1f"]["Gamma"]
    r2 = json.load(open(os.path.join(D, "kanal_2f_%s.json" % nk)))["noktalar"][nk]
    a = r2["ardisik"]
    kan["2f_ardisik"] = a["v21.00_lin_e1.38_eps5.10_dw1_rs+1"]["TOPLAM"]
    kan["2f_ardisik_bant"] = [min(v["TOPLAM"] for v in a.values()), max(v["TOPLAM"] for v in a.values())]
    e = r2["egilme_temas"]; kan["2f_egilme"] = e["D5.94_dw1"]["TOPLAM"]
    kan["2f_egilme_ZAs"] = e["D5.94_dw1"]["ZAs+ZAs"]; kan["2f_egilme_ZOp_nefes"] = e["D5.94_dw1"]["ZOp+ZOp_yalniz_nefes"]
    kan["2f_egilme_ZOp_geo"] = e["D5.94_dw1"]["ZOp+ZOp_yalniz_geo"]; kan["2f_egilme_ZAs+ZOp"] = e["D5.94_dw1"]["ZAs+ZOp"]
    eb = [v["TOPLAM"] for v in e.values()] + [r2["egilme_temas_sin2"]["TOPLAM"]]
    kan["2f_egilme_bant"] = [min(eb), max(eb)]
    ip = r2["duzlem_ici_temas"]; kan["2f_duzlem_ici_temas"] = ip["v21.00"]["TOPLAM"]
    kan["2f_duzlem_ici_bant"] = [min(v["TOPLAM"] for v in ip.values()), max(v["TOPLAM"] for v in ip.values())]
    kan["2f_registry_egrilik"] = ip["v21.00"]["La+La_yalniz_reg"]
    tot = kan["1f_TOPLAM"] + kan["2f_ardisik"] + kan["2f_egilme"] + kan["2f_duzlem_ici_temas"] + kan["ZOp_1f"]
    lo = kan["1f_TOPLAM_bant"][0] + kan["2f_ardisik_bant"][0] + kan["2f_egilme_bant"][0] + kan["2f_duzlem_ici_bant"][0]
    hi = kan["1f_TOPLAM_bant"][1] + kan["2f_ardisik_bant"][1] + kan["2f_egilme_bant"][1] + kan["2f_duzlem_ici_bant"][1]
    kan["GENEL_TOPLAM"] = tot; kan["GENEL_bant"] = [lo, hi]
    kan["KolE_tekrar_T1_ps"] = r1["KolE_tekrar_T1_ps"]
    kan["dogrulayici_2LA_tekrar_T1ps"] = {k: v for k, v in r2.items() if k.startswith("dogrulayici")}
    kan["E21_meV"] = r1["E21_meV"]
    i = np.argmin(abs(np.array(r1["F2_tablo"]["q_invA"]) - (c["q_Ls"] or 0)))
    kan["q_LA_invA"] = c["q_Ls"]; kan["q_TA_invA"] = c["q_Ts"]; kan["n_eff_qLA"] = r1["F2_tablo"]["n_eff"][i]; kan["F2_qLA"] = r1["F2_tablo"]["F2"][i]
    oz[nk] = kan
    print("\n=== %s  E21=%.2f meV  q_LA=%.3f q_TA=%.3f /A  n_eff(q_LA)=%.1f" % (nk, kan["E21_meV"], c["q_Ls"], c["q_Ts"], kan["n_eff_qLA"]))
    for kk, v in kan.items():
        if isinstance(v, float) and not kk.startswith(("q_", "n_eff", "F2", "E21", "KolE")):
            b = kan.get(kk + "_bant")
            print("  %-24s G=%.3e  T1=%9.3g ps %s" % (kk, v, 1e12 / v if v > 0 else float('inf'),
                  ("bant %.3g-%.3g ps" % (1e12 / b[1], 1e12 / b[0])) if b else ""))
json.dump(oz, open(os.path.join(D, "ozet.json"), "w"), indent=1)

import json, itertools, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sip_t1 import main
out = []
def run(**kw):
    r, *_ = main(verbose=False, **kw)
    print(kw, "T1(2p0)=%.0f ps  T1(2p+-)=%.0f ps" % (r["T1_2p0_ps"], r["T1_2p1_ps"]), flush=True)
    out.append(r); return r
print("== yakinsama"); run(N=1600); run(N=2400); run(N=3600)
print("== Gauss (ayni 2. moment)"); run(gauss_mode=True)
print("== A1 buzulme yok"); run(a1_contract=False)
print("== E/T2 buzulme"); run(et_contract=True)
print("== tarama")
for eps, Xu, Xd in itertools.product([11.4, 11.7], [8.0, 8.77, 10.5], [1.01, 5.0]):
    run(eps=eps, Xu=Xu, Xd=Xd)
for eps, Xu, Xd in [(11.4,10.5,5.0),(11.4,8.0,1.01)]:
    run(eps=eps, Xu=Xu, Xd=Xd, et_contract=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tarama.json"), "w"), indent=1)

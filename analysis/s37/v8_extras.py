"""V8 extras (2 Nov 2026 ~20:15 UTC): top-group contrast (P5) and our own-run forecasts (P6). Run from analysis/s37."""
# --- part 1: top group (public >= 40), flat skill prior, f = 1 ---
import sys; sys.argv=['x']
import numpy as np, importlib.util
spec=importlib.util.spec_from_file_location('v8','v8_sealed_prediction.py'); v8=importlib.util.module_from_spec(spec); spec.loader.exec_module(v8)
S_all,N_all=v8.S_all,v8.N_all.astype(float)
top=S_all>=40; St,Nt=S_all[top],N_all[top]
print('top group',top.sum(),'teams; n median',np.median(Nt))
rng=np.random.default_rng(1)
T=np.arange(15,90,0.1)
for sigma,f in ((2.87,1.0),(2.3,1.0),(3.3,1.0),(1.0,1.0)):
    D=v8.max_pdf((St[:,None]-T[None,:])/sigma,Nt[:,None],f); post=D/D.sum(1,keepdims=True); cdf=np.cumsum(post,1)
    sp=[]; drop=[]
    for _ in range(2000):
        j=np.minimum((cdf<rng.random(len(St))[:,None]).sum(1),len(T)-1); th=T[j]+rng.uniform(-0.05,0.05,len(St))
        P=th+sigma*rng.standard_normal((2,len(St))).max(0)
        sp.append(v8.spearman(St,P)); drop.append(np.mean(St-P))
    sp=np.array(sp); print(f'sigma {sigma}: top-group spearman {sp.mean():.3f} [{np.quantile(sp,0.05):.3f}, {np.quantile(sp,0.95):.3f}], mean drop {np.mean(drop):.2f}')

# --- part 2: board-wide shift estimate D-hat, 56832595 forecast, 8-run SD range ---
import numpy as np, importlib.util, math
spec=importlib.util.spec_from_file_location('v8','v8_sealed_prediction.py'); v8=importlib.util.module_from_spec(spec); spec.loader.exec_module(v8)
S,N=v8.S,v8.N
print('our public percentile in C', round(float((S<S[v8.iours]).mean()),3), 'C top-decile size', int(v8.TOP.sum()))
rng=np.random.default_rng(5)
for sigma,(mu,tau,f) in ((2.87,(24.97,2.25,1.0)),(2.3,(25.88,2.25,1.0)),(3.3,(24.26,2.25,1.0))):
    D,_=v8.mats(sigma,f); post=v8.prior(mu,tau,0.0)*D; post/=post.sum(1,keepdims=True); cdf=np.cumsum(post,1)
    d=[]; b=[]
    fam=np.array(v8.FAMILY); fm,fs=fam.mean(),fam.std(ddof=1)
    for _ in range(1500):
        j=np.minimum((cdf<rng.random(len(S))[:,None]).sum(1),len(v8.T)-1); th=v8.T[j]+rng.uniform(-0.1,0.1,len(S))
        w=rng.standard_normal((2,len(S))); P=th+sigma*np.where(N>1,w.max(0),w[0])
        Dh=float((P-S)[v8.G1].mean()); d.append(Dh)
        mo=fm+rng.standard_normal()*fs/math.sqrt(8); b.append(mo+fs*rng.standard_normal()-Dh)
    d=np.array(d); b=np.array(b)
    print(f'sigma {sigma}: D-hat mean {d.mean():.2f} sd {d.std():.2f}; 56832595 private - D-hat: {b.mean():.2f} [{np.quantile(b,0.05):.2f}, {np.quantile(b,0.95):.2f}], P(>=33.5)={np.mean(b>=33.5):.3f}')
fam=np.array(v8.FAMILY); s=fam.std(ddof=1)
sd=[np.std(fam.mean()+s*rng.standard_normal(8),ddof=1) for _ in range(20000)]
print('8-run private SD 90%:', np.quantile(sd,[0.05,0.5,0.95]).round(2))

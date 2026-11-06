"""Fig 3: level-attempt survival in generated-token time, four FB25 control runs (480 attempts), Kaplan-Meier with a
90% Greenwood band, and the fitted cure-lognormal (wall share q) and cure-log-logistic curves."""
import math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v5_budget_prediction import FB, load, attempts, fit, surv
runs = {r: load(r) for r in FB}
A = [a for r in FB for g in runs[r] for a in attempts(g)[0]]
t = np.array([max(a[1], 200) for a in A], float); d = np.array([a[2] for a in A], bool)
order = np.argsort(t); t, d = t[order], d[order]
S, var_acc, km_t, km_s, lo, hi = 1.0, 0.0, [0.0], [1.0], [1.0], [1.0]
n = len(t)
for i, (ti, di) in enumerate(zip(t, d)):
    at_risk = n - i
    if di:
        S *= 1 - 1 / at_risk; var_acc += 1 / (at_risk * (at_risk - 1)) if at_risk > 1 else 0
        se = S * math.sqrt(var_acc); km_t.append(ti); km_s.append(S); lo.append(max(0, S - 1.645 * se)); hi.append(min(1, S + 1.645 * se))
pl = fit(A, 'lognormal')[1]; pg = fit(A, 'loglogistic')[1]
x = np.logspace(math.log10(1e3), math.log10(4e5), 300)
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
ax.step(km_t, km_s, where='post', color='black', lw=1.4, label='Kaplan-Meier, 480 attempts (411 cleared, 69 open at 121 min)')
ax.fill_between(km_t, lo, hi, step='post', color='grey', alpha=0.25, label='90% band')
ax.plot(x, surv(x, pl, 'lognormal'), color='tab:blue', lw=1.8, label=f'cure-lognormal: walls q={pl[0]:.2f}, median {math.exp(pl[1])/1e3:.0f}k, s={pl[2]:.2f}')
ax.plot(x, surv(x, pg, 'loglogistic'), color='tab:orange', lw=1.2, ls='--', label=f'cure-log-logistic: q={pg[0]:.2f}, median {math.exp(pg[1])/1e3:.0f}k')
ax.axhline(pl[0], color='tab:blue', lw=0.8, ls=':')
ax.set_xscale('log'); ax.set_xlabel('generated tokens spent on the level'); ax.set_ylabel('share of attempts not yet cleared')
ax.set_title('A level is a wait for a crack: most clear within ~60k tokens, a few never do', fontsize=10)
ax.set_ylim(0, 1.02); ax.legend(fontsize=7.5, loc='upper right', frameon=False); ax.grid(alpha=0.25, which='both')
fig.tight_layout(); fig.savefig('/data/arc/paper/figs/s37_survival_fit.png'); print('saved; KM at 62k: %.3f, model %.3f' % (
    km_s[np.searchsorted(km_t, 62e3) - 1], float(surv(62e3, pl, 'lognormal'))))

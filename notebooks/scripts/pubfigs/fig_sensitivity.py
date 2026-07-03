#!/usr/bin/env python
"""Publication figure: one-at-a-time parameter sensitivity around the optimum.

Sigma-chi2 vs each free detector-model parameter (fixed detector-RNG seed),
from results/sensitivity_oat.tsv.

Run from notebooks/:  python scripts/pubfigs/fig_sensitivity.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import EXP, REF, panel_label, save

rows = []
for ln in open('results/sensitivity_oat.tsv'):
    p = ln.split()
    if len(p) >= 3 and p[0] not in ('param', '==='):
        try:
            rows.append((p[0], float(p[1]), float(p[2])))
        except ValueError:
            pass

PARAMS = [('blob', r'PSF width $\sigma_\mathrm{blob}$ (px)', 0.40),
          ('decay', 'phosphor decay time (ns)', 16.7),
          ('nsec', r'pixels per photon $n_\mathrm{sec}$', 10),
          ('keep', 'effective optical yield', 0.256),
          ('ap_prob', 'afterpulse probability', 0.012)]
ALIASES = {}
LETTERS = 'abcde'

fig, axs = plt.subplots(1, 5, figsize=(9.6, 2.2), sharey=True)
for i, (ax, (key, xlab, opt)) in enumerate(zip(axs, PARAMS)):
    names = ALIASES.get(key, (key,))
    xy = sorted((v, t) for n, v, t in rows if n in names)
    x = [v for v, _ in xy]
    y = [t for _, t in xy]
    ax.axvline(opt, color=REF, ls='--', lw=1.0)
    ax.plot(x, y, 'o-', color=EXP, ms=4)
    ax.set_xlabel(xlab, fontsize=8)
    panel_label(ax, f'({LETTERS[i]})', dx=0.06, dy=0.95)
    if i == 0:
        ax.set_ylabel(r'$\Sigma\chi^2$')
axs[0].set_ylim(0.60, 1.08)

fig.tight_layout(w_pad=0.8)
save(fig, 'sensitivity_oat')
print({n for n, _, _ in rows})

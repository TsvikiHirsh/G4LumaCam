#!/usr/bin/env python
"""Publication figure: satellite-aware event positioning.

(a) simulated out-of-focus event-position resolution (sigma vs optical truth)
    for the three photon2event position modes; the largest-cluster (parent)
    position removes the satellite centroid pull and approaches the in-focus
    single-photon limit.
(b) Sigma-chi2 agreement between simulation and the air45 data for the previous
    calibration, the re-optimized detector model (centroid positions), and the
    re-optimized model with largest-cluster positions (3-seed mean +/- std on
    the 1e6 archive; values from OPTIMIZATION_RESULTS.md).

Run from notebooks/:  python scripts/pubfigs/fig_position_mode.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
from pubstyle import EXP, SIM, REF, panel_label, save

# sim oof resolution sigma vs truth (px), default windows; POSITION_MODE_STUDY.md
POS = [('centroid', 1.76), ('first photon', 0.80), ('largest cluster', 0.66)]
INF = 0.60

# Sigma-chi2 progression, 1e6 archive replicates; OPTIMIZATION_RESULTS.md
PROG = [('previous\ncalibration', 1.044, 0.003),
        ('re-optimized,\ncentroid', 0.644, 0.010),
        ('re-optimized,\nlargest cluster', 0.518, 0.018)]

fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9))

x = range(len(POS))
bars = a.bar(x, [v for _, v in POS], width=0.62, color=EXP, alpha=0.85)
for i, (_, v) in enumerate(POS):
    a.text(i, v + 0.04, f'{v:.2f}', ha='center', fontsize=8.5)
a.axhline(INF, color=REF, ls='--', lw=1.2)
a.text(1.0, INF + 0.04, 'in-focus limit', fontsize=8, ha='center', color=REF)
a.set_xticks(list(x))
a.set_xticklabels([n for n, _ in POS], fontsize=8.5)
a.set_ylabel(r'out-of-focus resolution $\sigma$ (px)')
a.set_ylim(0, 2.0)
panel_label(a, '(a)')

x = range(len(PROG))
b.bar(x, [m for _, m, _ in PROG], yerr=[s for _, _, s in PROG], width=0.62,
      color=SIM, alpha=0.85, capsize=3, error_kw={'lw': 1})
for i, (_, m, s) in enumerate(PROG):
    b.text(i, m + s + 0.03, f'{m:.2f}', ha='center', fontsize=8.5)
b.set_xticks(list(x))
b.set_xticklabels([n for n, _, _ in PROG], fontsize=8.5)
b.set_ylabel(r'$\Sigma\chi^2$ (simulation vs experiment)')
b.set_ylim(0, 1.2)
panel_label(b, '(b)')

fig.tight_layout(w_pad=2.0)
save(fig, 'position_mode')

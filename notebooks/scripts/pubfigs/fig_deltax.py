#!/usr/bin/env python
"""Publication figure: cumulative per-event position error of the
out-of-focus reconstruction, referenced to the ray-traced optical truth.

Fraction of events with |dx| below a given value, for raw pixel hits (hitmap
equivalent) and for events positioned by the photon mean (centroid), the
earliest photon, and the largest cluster; the in-focus reconstruction of the
same simulation provides the reference curve. Monotone curves make the
estimators easy to compare: at any error budget the vertical reading gives
the fraction of usable events.

Run from notebooks/:  python scripts/pubfigs/fig_deltax.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pubstyle import EXP, SIM, OPT, REF, save

P2MM = 0.464

df = pd.read_csv('archive/openbeam_ptb_n1e5/oof_h3opt/AssociatedResults/sim_merged.csv',
                 usecols=['px/id', 'ph/id', 'ev/id', 'ph/x', 'ph/toa', 'px/x',
                          'sim/x_opt'])
npx = df.groupby('ph/id')['px/id'].nunique().rename('npx')
ph = (df.drop_duplicates(['ev/id', 'ph/id']).dropna(subset=['ph/x', 'ev/id'])
        .join(npx, on='ph/id'))
ph = ph.sort_values(['ev/id', 'npx'], ascending=[True, False])
g = ph.groupby('ev/id')
ev = pd.DataFrame({'cogx': g['ph/x'].mean(), 'n': g['ph/id'].nunique()})
ev['fx'] = ph.sort_values(['ev/id', 'ph/toa']).groupby('ev/id')['ph/x'].first()
L = g.first()
ev['lx'], ev['ox'] = L['ph/x'], L['sim/x_opt']
m = ev[ev.n >= 2].dropna(subset=['ox'])

# raw pixels of the same multi-photon events (hitmap equivalent)
pix = df.dropna(subset=['px/x', 'sim/x_opt', 'ev/id'])
pix = pix[pix['ev/id'].isin(m.index)]

# in-focus reconstruction of the same trace (single-photon events)
di = pd.read_csv('archive/openbeam_ptb_n1e5/inf_h3opt/AssociatedResults/sim_merged.csv',
                 usecols=['ev/id', 'ev/x', 'sim/x_opt'])
inf = di.dropna(subset=['ev/id', 'ev/x', 'sim/x_opt']).drop_duplicates('ev/id')

SERIES = [((pix['px/x'] - pix['sim/x_opt']) * P2MM, '0.45', '-',
           'raw pixels (hitmap)'),
          ((m.cogx - m.ox) * P2MM, SIM, '-', 'centroid'),
          ((m.fx - m.ox) * P2MM, OPT, '-', 'first photon'),
          ((m.lx - m.ox) * P2MM, EXP, '-', 'largest cluster'),
          ((inf['ev/x'] - inf['sim/x_opt']) * P2MM, 'k', '--',
           'in-focus reference')]

fig, ax = plt.subplots(figsize=(4.6, 3.4))
for d, colr, ls, lab in SERIES:
    a = np.sort(np.abs(np.asarray(d.dropna())))
    frac = np.arange(1, len(a) + 1) / len(a)
    ax.plot(a, frac, color=colr, ls=ls, lw=1.7, label=lab)
ax.axhline(0.68, color='0.75', lw=0.7, zorder=0)
ax.text(2.93, 0.685, '68%', fontsize=7, color='0.45', ha='right', va='bottom')
ax.set_xlim(0, 3)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r'position error $|\Delta x|$ (mm)')
ax.set_ylabel('fraction of events')
ax.legend(loc='lower right', fontsize=7.5)

fig.tight_layout()
save(fig, 'deltax_modes')
for d, _, _, lab in SERIES:
    a = np.abs(np.asarray(d.dropna()))
    print(f'{lab}: 68% within {np.percentile(a, 68):.2f} mm, '
          f'frac<0.5mm={np.mean(a < 0.5):.2f}')

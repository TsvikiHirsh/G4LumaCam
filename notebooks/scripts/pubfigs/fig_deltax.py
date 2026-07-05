#!/usr/bin/env python
"""Publication figure: per-event position error of the out-of-focus
reconstruction, referenced to the ray-traced optical truth.

Histograms of dx = x_reco - x_truth for multi-photon (out-of-focus) simulated
events, comparing raw pixel hits (hitmap equivalent) with the three
photon2event position estimators (centroid / first photon / largest cluster).
The afterpulse satellites produce the shoulder at |dx| ~ 3-7 px in the raw and
centroid distributions; anchoring the event to its largest cluster removes it.

Run from notebooks/:  python scripts/pubfigs/fig_deltax.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pubstyle import EXP, SIM, OPT, REF, panel_label, save

df = pd.read_csv('archive/openbeam_ptb_n1e5/oof_h3opt/AssociatedResults/sim_merged.csv',
                 usecols=['px/id', 'ph/id', 'ev/id', 'ph/x', 'ph/toa', 'px/x',
                          'sim/x_opt'])
npx = df.groupby('ph/id')['px/id'].nunique().rename('npx')
ph = (df.drop_duplicates(['ev/id', 'ph/id']).dropna(subset=['ph/x', 'ev/id'])
        .join(npx, on='ph/id'))
ph = ph.sort_values(['ev/id', 'npx'], ascending=[True, False])
g = ph.groupby('ev/id')
ev = pd.DataFrame({'cogx': g['ph/x'].mean(), 'n': g['ph/id'].nunique()})
first = ph.sort_values(['ev/id', 'ph/toa']).groupby('ev/id').first()
ev['fx'] = first['ph/x']
L = g.first()
ev['lx'], ev['ox'] = L['ph/x'], L['sim/x_opt']
m = ev[ev.n >= 2].dropna(subset=['ox'])

# raw pixels of the same multi-photon events (hitmap equivalent)
pix = df.dropna(subset=['px/x', 'sim/x_opt', 'ev/id'])
pix = pix[pix['ev/id'].isin(m.index)]
dx_pix = pix['px/x'] - pix['sim/x_opt']

P2MM = 0.464          # object-space mm per detector pixel
SERIES = [(dx_pix * P2MM, '0.45', 'raw pixels (hitmap)'),
          ((m.cogx - m.ox) * P2MM, SIM, 'event centroid'),
          ((m.fx - m.ox) * P2MM, OPT, 'first photon'),
          ((m.lx - m.ox) * P2MM, EXP, 'largest cluster')]

bins = np.linspace(-3.5, 3.5, 36)
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for d, colr, lab in SERIES:
    h, e = np.histogram(d, bins=bins, density=True)
    if lab.startswith('raw'):
        ax.stairs(h, e, color=colr, fill=True, alpha=0.35, label=lab)
    else:
        ax.stairs(h, e, color=colr, lw=1.6, label=lab)
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(bottom=0)
ax.set_xlabel(r'$\Delta x$ = reconstructed $-$ true position (mm)')
ax.set_ylabel('probability density (mm$^{-1}$)')
ax.legend(loc='upper left', fontsize=7.5)

fig.tight_layout()
save(fig, 'deltax_modes')
for d, _, lab in SERIES:
    d = np.asarray(pd.Series(d).dropna())
    s = (np.percentile(d, 84.1) - np.percentile(d, 15.9)) / 2
    print(f'{lab}: sigma={s:.2f} frac|dx|>3 = {np.mean(np.abs(d) > 3):.2f}')

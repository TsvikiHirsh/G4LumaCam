#!/usr/bin/env python
"""Publication figure: photon-multiplicity probability by parent particle.

P(photons per event | parent particle) shown as an annotated probability map,
from tagged simulation truth (all reconstructed events of the calibrated
out-of-focus simulation).

Run from notebooks/:  python scripts/pubfigs/fig_parent_multiplicity.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pubstyle import EXP, save

df = pd.read_csv('archive/openbeam_ptb_n1e5/oof_h3opt/AssociatedResults/sim_merged.csv',
                 usecols=['ph/id', 'ev/id', 'ph/x', 'sim/parentName'])
ph = df.drop_duplicates(['ev/id', 'ph/id']).dropna(subset=['ph/x', 'ev/id'])
g = ph.groupby('ev/id')
ev = pd.DataFrame({'n': g['ph/id'].nunique(), 'par': g['sim/parentName'].first()})

PARENTS = [('proton', 'recoil proton'),
           ('e-', r'electron ($\gamma$-induced)'),
           ('C12', 'carbon recoil')]
NMAX = 4
xs = np.arange(1, NMAX + 1)

P = np.zeros((len(PARENTS), NMAX))
N = []
for i, (par, _) in enumerate(PARENTS):
    d = ev[ev.par == par]['n'].clip(upper=NMAX)
    N.append(len(d))
    P[i] = [(d == k).mean() for k in xs]

fig, ax = plt.subplots(figsize=(4.4, 2.5))
cmap = LinearSegmentedColormap.from_list('b', ['white', EXP])
ax.imshow(P, cmap=cmap, vmin=0, vmax=1.0, aspect='auto')
for i in range(len(PARENTS)):
    for j in range(NMAX):
        ax.text(j, i, f'{P[i, j]:.2f}', ha='center', va='center', fontsize=8.5,
                color='white' if P[i, j] > 0.55 else '0.15')
ax.set_xticks(range(NMAX))
ax.set_xticklabels(['1', '2', '3', r'$\geq$4'])
ax.set_yticks(range(len(PARENTS)))
ax.set_yticklabels([f'{lab}\n(N={n:,})' for (_, lab), n in zip(PARENTS, N)],
                   fontsize=8)
ax.set_xlabel('photons per event')
ax.tick_params(top=False, right=False)
for s in ax.spines.values():
    s.set_visible(False)

fig.tight_layout()
save(fig, 'parent_multiplicity')
print(P.round(3))

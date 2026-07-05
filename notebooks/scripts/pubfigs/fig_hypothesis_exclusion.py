#!/usr/bin/env python
"""Publication figure: the satellite clusters cannot originate in the
scintillator or in the collection optics.

(a) NIST PSTAR recoil-proton range in plastic vs proton energy; the kinematic
    maximum (full neutron energy transfer, equal masses) at the 10 MeV beam
    endpoint gives a track of only ~1.2 mm, far short of the observed ~3.0 mm
    satellite displacement.
(b) In-event cluster separation vs interaction depth (simulated out-of-focus
    reconstruction): the pure-optical separation (ray-traced truth) is flat at
    ~0.05 mm at every depth, while the simulation including intensifier
    afterpulsing reproduces the experimental level.

All quantities in object-plane mm (0.464 mm per detector pixel).

Run from notebooks/:  python scripts/pubfigs/fig_hypothesis_exclusion.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pubstyle import EXP, SIM, OPT, REF, panel_label, save

P2MM = 0.464          # object-space mm per detector pixel
SAT_MM = 6.4 * P2MM   # observed satellite displacement, air45 oof (median)
EN_MAX = 10.0         # beam endpoint = kinematic-max recoil (equal masses)

# ---------------- (a) recoil-proton range (NIST PSTAR, water ~ plastic)
E = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 10])
R_mm = np.array([0.0080, 0.0237, 0.0723, 0.143, 0.233, 0.342, 0.461, 0.815, 1.23])

# ---------------- (b) depth scan from the simulated oof reconstruction
df = pd.read_csv('archive/openbeam_ptb_n1e5/oof_h3opt/AssociatedResults/sim_merged.csv',
                 usecols=['px/id', 'ph/id', 'ev/id', 'ph/x', 'ph/y',
                          'sim/x_opt', 'sim/y_opt', 'sim/pz'])
df['npx'] = df.groupby('ph/id')['px/id'].transform('nunique')
ph = df.drop_duplicates('ph/id').dropna(subset=['ph/x', 'sim/x_opt'])
ph = ph[ph.groupby('ev/id')['ph/id'].transform('nunique') >= 2].copy()
ph = ph.sort_values(['ev/id', 'npx'], ascending=[True, False])
par = ph.groupby('ev/id').first().rename(columns={
    'ph/x': 'px0', 'ph/y': 'py0', 'sim/x_opt': 'ox0', 'sim/y_opt': 'oy0',
    'sim/pz': 'pz0'})
sat = ph[ph.groupby('ev/id').cumcount() >= 1]
sat = sat.merge(par[['px0', 'py0', 'ox0', 'oy0', 'pz0']], on='ev/id')
sat['d_det'] = np.hypot(sat['ph/x'] - sat['px0'], sat['ph/y'] - sat['py0'])
sat['d_opt'] = np.hypot(sat['sim/x_opt'] - sat['ox0'], sat['sim/y_opt'] - sat['oy0'])
sat = sat[(sat['d_det'] > 1.5) & (sat['d_det'] < 40)]
bins = np.arange(0, 21, 2)
ctr = 0.5 * (bins[:-1] + bins[1:])
g = sat.groupby(pd.cut(sat['pz0'], bins), observed=True)
d_det = g['d_det'].median().values * P2MM
d_opt = g['d_opt'].median().values * P2MM

# ---------------- figure
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))

a.plot(E, R_mm, 'o-', color=EXP, ms=4, label='recoil-proton range (NIST PSTAR)')
a.axhline(SAT_MM, color='k', ls='--', lw=1.4)
a.text(5.0, SAT_MM - 0.09, 'observed satellite displacement', fontsize=8,
       ha='center', va='top')
a.axvspan(EN_MAX, 12, color='0.92')
a.text(11.0, 1.6, 'kinematically\nforbidden', color='0.45', ha='center',
       va='center', fontsize=8, rotation=90)
a.set_xlabel('recoil-proton energy (MeV)')
a.set_ylabel('recoil-proton track length (mm)')
a.set_xlim(0, 12)
a.set_ylim(0, 3.4)
a.legend(loc='center left', bbox_to_anchor=(0.02, 0.62), fontsize=7.5)
panel_label(a, '(a)')

b.plot(ctr, d_det, 's-', color=SIM, ms=4,
       label='simulation, optics + afterpulsing')
b.plot(ctr, d_opt, '^-', color=OPT, ms=4, label='simulation, optical truth')
b.axhline(SAT_MM, color='k', ls='--', lw=1.4)
b.text(10.0, SAT_MM - 0.09, 'experiment (open beam)', fontsize=8,
       ha='center', va='top')
b.set_xlabel('interaction depth in scintillator (mm)')
b.set_ylabel('in-event cluster separation (mm)')
b.set_xlim(0, 20)
b.set_xticks([0, 5, 10, 15, 20])
b.set_ylim(0, 3.4)
b.legend(loc='center left', bbox_to_anchor=(0.02, 0.42), fontsize=7.5)
panel_label(b, '(b)')

fig.tight_layout(w_pad=2.0)
save(fig, 'hypothesis_exclusion')
print(f'optical truth median {np.nanmin(d_opt):.3f}-{np.nanmax(d_opt):.3f} mm; '
      f'sim detector {np.nanmin(d_det):.2f}-{np.nanmax(d_det):.2f} mm; '
      f'satellite {SAT_MM:.2f} mm')

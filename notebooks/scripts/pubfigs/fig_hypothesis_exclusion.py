#!/usr/bin/env python
"""Publication figure: quantitative exclusion of H1 (recoil track) and H2 (optics).

(a) H1 — NIST PSTAR recoil-proton range imaged onto the detector vs proton
    energy; the kinematic maximum (full neutron energy transfer, equal masses)
    at the 10 MeV beam endpoint gives a track of only ~2.6 px, far short of the
    observed ~6.4 px satellite displacement.
(b) H2 — in-event cluster separation vs interaction depth (sim, out-of-focus
    reconstruction): the pure-optical separation (sim truth) is flat at ~0.1 px,
    the geometric defocus-disc diameter bounds any optical explanation below
    ~5.6 px, while the simulation including intensifier afterpulsing reproduces
    the experimental level at every depth.

Run from notebooks/:  python scripts/pubfigs/fig_hypothesis_exclusion.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pubstyle import EXP, SIM, OPT, REF, panel_label, save

P2MM = 0.464          # object-space mm per detector pixel
SAT_PX = 6.4          # observed satellite displacement, air45 oof (median, px)
EN_MAX = 10.0         # beam endpoint = kinematic-max recoil (equal masses)

# ---------------- (a) H1: recoil-proton range (NIST PSTAR, water ~ plastic)
E = np.array([0.5, 1, 2, 3, 4, 5, 6, 8, 10])
R_mm = np.array([0.0080, 0.0237, 0.0723, 0.143, 0.233, 0.342, 0.461, 0.815, 1.23])
R_px = R_mm / P2MM

# ---------------- (b) H2: depth scan from the simulated oof reconstruction
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
d_det = g['d_det'].median().values
d_opt = g['d_opt'].median().values

# geometric ceiling: defocus-disc diameter c = dz * D / s (aperture D, object s)
D_AP, S_OBJ = 64.95, 500.0
coc = ctr * D_AP / S_OBJ / P2MM

# ---------------- figure
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.1))

a.plot(E, R_px, 'o-', color=EXP, ms=4, label='recoil-proton range (NIST PSTAR)')
a.axhline(SAT_PX, color='k', ls='--', lw=1.4)
a.text(5.0, SAT_PX - 0.18, 'observed satellite displacement', fontsize=8,
       ha='center', va='top')
a.axvspan(EN_MAX, 12, color='0.92')
a.text(11.0, 3.5, 'kinematically\nforbidden', color='0.45', ha='center',
       va='center', fontsize=8, rotation=90)
a.set_xlabel('recoil-proton energy (MeV)')
a.set_ylabel('track length imaged on detector (px)')
a.set_xlim(0, 12)
a.set_ylim(0, 7.2)
a.legend(loc='center left', bbox_to_anchor=(0.02, 0.62))
panel_label(a, '(a)')

b.plot(ctr, coc, '-', color=REF, ls=(0, (4, 2)),
       label='defocus-disc diameter (optical bound)')
b.plot(ctr, d_det, 's-', color=SIM, ms=4,
       label='simulation, optics + afterpulsing')
b.plot(ctr, d_opt, '^-', color=OPT, ms=4,
       label='simulation, optical truth')
b.axhline(SAT_PX, color='k', ls='--', lw=1.4)
b.text(10.0, SAT_PX - 0.20, 'observed satellite displacement', fontsize=8,
       ha='center', va='top')
b.set_xlabel('interaction depth in scintillator (mm)')
b.set_ylabel('in-event cluster separation (px)')
b.set_xlim(0, 20)
b.set_xticks([0, 5, 10, 15, 20])
b.set_ylim(0, 9.0)
b.legend(loc='upper left', bbox_to_anchor=(0.02, 0.99), fontsize=7.2,
         handlelength=1.8, labelspacing=0.35)
panel_label(b, '(b)', dx=0.90)

fig.tight_layout(w_pad=2.0)
save(fig, 'hypothesis_exclusion')
print(f'optical truth median {np.nanmin(d_opt):.2f}-{np.nanmax(d_opt):.2f} px; '
      f'sim detector {np.nanmin(d_det):.1f}-{np.nanmax(d_det):.1f} px; '
      f'CoC max {coc.max():.1f} px')

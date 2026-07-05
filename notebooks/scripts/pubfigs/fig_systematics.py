#!/usr/bin/env python
"""Publication figure: simulation-based interrogation of systematic effects.

Using tagged simulation truth (interaction depth, neutron energy, parent
particle) the calibrated model separates the two contributions to the
out-of-focus position error:
(a) vs interaction depth: the centroid error is dominated by the afterpulse
    satellites and is depth-independent, while the first-photon and
    largest-cluster positions reveal the underlying depth-dependent optical
    blur;
(b) vs neutron energy: the satellite pull on the centroid grows at low energy
    (fewer photons per event), the corrected estimators do not;
(c) mean photon multiplicity by parent particle (recoil proton, gamma-induced
    electron, carbon recoil) - a parent-type signature relevant for
    pulse-shape discrimination.

Run from notebooks/:  python scripts/pubfigs/fig_systematics.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pubstyle import EXP, SIM, OPT, REF, panel_label, save

rng = np.random.default_rng(1)

df = pd.read_csv('archive/openbeam_ptb_n1e5/oof_h3opt/AssociatedResults/sim_merged.csv',
                 usecols=['px/id', 'ph/id', 'ev/id', 'ph/x', 'ph/toa',
                          'sim/x_opt', 'sim/pz', 'sim/neutronEnergy',
                          'sim/parentName'])
npx = df.groupby('ph/id')['px/id'].nunique().rename('npx')
ph = (df.drop_duplicates(['ev/id', 'ph/id']).dropna(subset=['ph/x', 'ev/id'])
        .join(npx, on='ph/id'))
ph = ph.sort_values(['ev/id', 'npx'], ascending=[True, False])
g = ph.groupby('ev/id')
ev = pd.DataFrame({'cogx': g['ph/x'].mean(), 'n': g['ph/id'].nunique()})
ev['fx'] = ph.sort_values(['ev/id', 'ph/toa']).groupby('ev/id')['ph/x'].first()
L = g.first()
ev['lx'], ev['ox'] = L['ph/x'], L['sim/x_opt']
ev['pz'], ev['En'], ev['par'] = L['sim/pz'], L['sim/neutronEnergy'], L['sim/parentName']
m = ev[ev.n >= 2].dropna(subset=['ox'])

P2MM = 0.464          # object-space mm per detector pixel
MODES = [('cogx', SIM, 'centroid', 'o'),
         ('fx', OPT, 'first photon', '^'),
         ('lx', EXP, 'largest cluster', 's')]


def rsig(d):
    """Robust per-axis sigma and its bootstrap error."""
    d = np.asarray(d.dropna())
    s = (np.percentile(d, 84.1) - np.percentile(d, 15.9)) / 2
    bs = [(np.percentile(b, 84.1) - np.percentile(b, 15.9)) / 2
          for b in (rng.choice(d, size=len(d)) for _ in range(200))]
    return s, np.std(bs)


def binned(x, d, edges):
    ctr, val, err = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        if sel.sum() < 40:
            continue
        s, e = rsig(d[sel])
        ctr.append(0.5 * (lo + hi)); val.append(s); err.append(e)
    return np.array(ctr), np.array(val), np.array(err)


FWHM = 2.3548 * P2MM   # sigma (px) -> FWHM (mm)

fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.8, 2.9))

# (a) resolution vs interaction depth
edges = np.linspace(0, 20, 8)
for col, colr, lab, mk in MODES:
    x, v, e = binned(m.pz, m[col] - m.ox, edges)
    a.errorbar(x, v * FWHM, yerr=e * FWHM, color=colr, marker=mk, ms=4,
               lw=1.4, capsize=2, label=lab)
a.set_xlabel('interaction depth in scintillator (mm)')
a.set_ylabel('position resolution FWHM (mm)')
a.set_ylim(0, 2.7)
a.set_xlim(0, 20)
a.legend(loc='center left', bbox_to_anchor=(0.02, 0.55), fontsize=7.5)
panel_label(a, '(a)')

# (b) resolution vs neutron energy
edges = np.array([1, 3, 5, 7, 9, 10])
for col, colr, lab, mk in MODES:
    x, v, e = binned(m.En, m[col] - m.ox, edges)
    b.errorbar(x, v * FWHM, yerr=e * FWHM, color=colr, marker=mk, ms=4,
               lw=1.4, capsize=2, label=lab)
b.set_xlabel('neutron energy (MeV)')
b.set_ylabel('position resolution FWHM (mm)')
b.set_ylim(0, 2.7)
b.set_xlim(0, 10.5)
b.legend(loc='lower center', bbox_to_anchor=(0.55, 0.02), fontsize=7.5)
panel_label(b, '(b)')

# (c) photon-multiplicity distribution by parent particle (grouped bars)
PARENTS = [('proton', EXP, 'recoil proton'),
           ('e-', SIM, r'electron ($\gamma$-induced)'),
           ('C12', OPT, 'carbon recoil')]
nmax = 4
xs = np.arange(1, nmax + 1)
width = 0.26
for j, (par, colr, lab) in enumerate(PARENTS):
    d = ev[ev.par == par]['n'].clip(upper=nmax)
    frac = np.array([(d == k).mean() for k in xs])
    c.bar(xs + (j - 1) * width, frac, width=width * 0.92, color=colr,
          alpha=0.88, label=lab)
c.set_xticks(xs)
c.set_xticklabels(['1', '2', '3', r'$\geq$4'])
c.set_xlabel('photons per event')
c.set_ylabel('fraction of events')
c.set_ylim(0, 0.95)
c.legend(loc='upper right', fontsize=7.5)
panel_label(c, '(c)')

fig.tight_layout(w_pad=1.8)
save(fig, 'systematics')
for par in ['proton', 'e-', 'C12']:
    d = ev[ev.par == par]
    print(par, 'N_ev=%d mean nphot=%.2f' % (len(d), d.n.mean()))

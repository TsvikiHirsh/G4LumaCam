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


fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.8, 2.9))

# (a) resolution vs interaction depth
edges = np.linspace(0, 20, 8)
for col, colr, lab, mk in MODES:
    x, v, e = binned(m.pz, m[col] - m.ox, edges)
    a.errorbar(x, v, yerr=e, color=colr, marker=mk, ms=4, lw=1.4, capsize=2,
               label=lab)
a.set_xlabel('interaction depth in scintillator (mm)')
a.set_ylabel(r'position error $\sigma$ (px)')
a.set_ylim(0, 2.4)
a.set_xlim(0, 20)
a.legend(loc='upper left', bbox_to_anchor=(0.03, 0.88), fontsize=7.5)
a.tick_params(right=False)
sa = a.secondary_yaxis('right', functions=(lambda p: p * P2MM,
                                           lambda mm: mm / P2MM))
sa.set_ylabel(r'$\sigma$ (mm)', fontsize=8)
panel_label(a, '(a)')

# (b) resolution vs neutron energy
edges = np.array([1, 3, 5, 7, 9, 10])
for col, colr, lab, mk in MODES:
    x, v, e = binned(m.En, m[col] - m.ox, edges)
    b.errorbar(x, v, yerr=e, color=colr, marker=mk, ms=4, lw=1.4, capsize=2,
               label=lab)
b.set_xlabel('neutron energy (MeV)')
b.set_ylabel(r'position error $\sigma$ (px)')
b.set_ylim(0, 2.4)
b.set_xlim(0, 10.5)
b.legend(loc='upper left', bbox_to_anchor=(0.03, 0.88), fontsize=7.5)
b.tick_params(right=False)
sb = b.secondary_yaxis('right', functions=(lambda p: p * P2MM,
                                           lambda mm: mm / P2MM))
sb.set_ylabel(r'$\sigma$ (mm)', fontsize=8)
panel_label(b, '(b)')

# (c) mean photon multiplicity by parent particle (all events)
PARENTS = [('proton', 'recoil\nproton'),
           ('e-', 'electron\n($\\gamma$-induced)'),
           ('C12', 'carbon\nrecoil')]
means, sems = [], []
for par, _ in PARENTS:
    d = ev[ev.par == par]['n']
    means.append(d.mean())
    sems.append(d.std() / np.sqrt(len(d)))
c.bar(range(3), means, yerr=sems, width=0.62, color=EXP, alpha=0.85,
      capsize=3, error_kw={'lw': 1})
for i, v in enumerate(means):
    c.text(i, v + sems[i] + 0.05, f'{v:.2f}', ha='center', fontsize=8.5)
c.set_xticks(range(3))
c.set_xticklabels([lab for _, lab in PARENTS], fontsize=8)
c.set_ylabel('mean photons per event')
c.set_ylim(0, 2.3)
panel_label(c, '(c)')

fig.tight_layout(w_pad=1.8)
save(fig, 'systematics')
for par in ['proton', 'e-', 'C12']:
    d = ev[ev.par == par]
    print(par, 'N_ev=%d mean nphot=%.2f' % (len(d), d.n.mean()))

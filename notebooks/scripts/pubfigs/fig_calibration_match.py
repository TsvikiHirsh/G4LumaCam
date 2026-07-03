#!/usr/bin/env python
"""Publication figure: calibrated G4LumaCam vs the 2-s air45 open-beam data.

Four per-event observables x two reconstruction modes at the converged optimum
(opt_largest48h eval_0659: blob 0.405 px, decay 16.5 ns, n_secondaries 9,
photon_keep_fraction 0.241, ap_prob 0, position_mode="largest").

Run from notebooks/:  python scripts/pubfigs/fig_calibration_match.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from pubstyle import EXP, SIM, panel_label, save

EVAL = 'opt_largest48h/eval_0659'

PHN = np.arange(21)
EVN = np.arange(11)
DEVX = (np.linspace(-10, 10, 41)[:-1] + np.linspace(-10, 10, 41)[1:]) / 2
DTOA = ((np.linspace(0, 5e-7, 41)[:-1] + np.linspace(0, 5e-7, 41)[1:]) / 2) * 1e9

METRICS = [('ph_n', PHN, 'pixels per photon cluster', (0, 20)),
           ('ev_n', EVN, 'photons per event', (0, 8)),
           ('ev_dx', DEVX, 'pixel$-$event centroid residual (px)', (-10, 10)),
           ('ev_dtoa', DTOA, 'photon time in event (ns)', (0, 200))]
ROWS = [('in_focus', 'In focus'), ('out_of_focus', 'Out of focus')]
LETTERS = 'abcdefgh'

fig, axs = plt.subplots(2, 4, figsize=(9.6, 4.6))
for r, (foc, rowlab) in enumerate(ROWS):
    d = json.load(open(f'{EVAL}/{foc}.json'))
    for c, (m, x, xl, xlim) in enumerate(METRICS):
        ax = axs[r, c]
        s = np.array(d['distributions_sim'][m])
        e = np.array(d['distributions_exp'][m])
        xx = x[:len(s)]
        ax.stairs(e, np.append(xx, xx[-1] + (xx[1] - xx[0])) - (xx[1] - xx[0]) / 2,
                  color=EXP, fill=True, alpha=0.25)
        ax.stairs(e, np.append(xx, xx[-1] + (xx[1] - xx[0])) - (xx[1] - xx[0]) / 2,
                  color=EXP, lw=1.3, label='experiment')
        ax.stairs(s, np.append(xx, xx[-1] + (xx[1] - xx[0])) - (xx[1] - xx[0]) / 2,
                  color=SIM, lw=1.3, label='simulation')
        ax.set_xlim(*xlim)
        ax.set_ylim(bottom=0)
        panel_label(ax, f'({LETTERS[r * 4 + c]})', dx=0.72, dy=0.95)
        if r == 1:
            ax.set_xlabel(xl, fontsize=8)
        if c == 0:
            ax.set_ylabel(f'{rowlab}\n\nfraction of entries')
        if r == 0 and c == 1:
            ax.legend(loc='center right', fontsize=7.5)

fig.tight_layout(w_pad=1.2, h_pad=1.2)
save(fig, 'calibration_match')
tin = json.load(open(f'{EVAL}/in_focus.json'))['scores']
tout = json.load(open(f'{EVAL}/out_of_focus.json'))['scores']
print('chi2 in:', {k: round(v, 3) for k, v in tin.items()})
print('chi2 out:', {k: round(v, 3) for k, v in tout.items()})

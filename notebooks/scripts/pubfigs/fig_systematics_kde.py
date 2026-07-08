#!/usr/bin/env python
"""Publication figure: conditional KDE (ridge-style) of the per-event
position error versus interaction depth, neutron energy, and parent particle.

Rows, top to bottom:
  1. in-focus reconstruction, event position (reference width)
  2. out-of-focus, centroid (photon-mean) position
  3. out-of-focus, first-photon position
  4. out-of-focus, largest-cluster position
Columns: interaction depth z (left), neutron energy (middle), parent particle
(right, violin-style KDE per category). Each depth/energy panel shows the
conditional density of dx = x_reco - x_truth, normalized to unit peak in
every x slice, so the band width reads directly as the resolution; the
parent-particle panel shows the same normalized density as a violin per
category. A colorbar at the end of each row gives the density scale (white to
the row's color = 0 to peak conditional density). Row colors: gray = in-focus
reference, orange = centroid, green = first photon, blue = largest cluster.

Run from notebooks/:  python scripts/pubfigs/fig_systematics_kde.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pubstyle import EXP, SIM, OPT, REF, save

P2MM = 0.464

# ---------------- load out-of-focus (multi-photon) events with estimators
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
oof = ev[ev.n >= 2].dropna(subset=['ox'])

# ---------------- load in-focus (single-photon) events, same trace
di = pd.read_csv('archive/openbeam_ptb_n1e5/inf_h3opt/AssociatedResults/sim_merged.csv',
                 usecols=['ev/id', 'ev/x', 'sim/x_opt', 'sim/pz',
                          'sim/neutronEnergy', 'sim/parentName'])
inf = di.dropna(subset=['ev/id', 'ev/x', 'sim/x_opt']).drop_duplicates('ev/id')

ROWS = [('in-focus', 'event position', inf['sim/pz'], inf['sim/neutronEnergy'],
         inf['sim/parentName'],
         (inf['ev/x'] - inf['sim/x_opt']) * P2MM, '0.35'),
        ('out-of-focus', 'centroid', oof.pz, oof.En, oof.par,
         (oof.cogx - oof.ox) * P2MM, SIM),
        ('out-of-focus', 'first photon', oof.pz, oof.En, oof.par,
         (oof.fx - oof.ox) * P2MM, OPT),
        ('out-of-focus', 'largest cluster', oof.pz, oof.En, oof.par,
         (oof.lx - oof.ox) * P2MM, EXP)]

PARENTS = [('proton', 'recoil\nproton'), ('e-', r'electron' '\n' r'($\gamma$-ind.)'),
           ('C12', 'carbon\nrecoil')]

YR = 2.2          # dx range (mm)
NY, NX = 121, 61
LEVELS = [0.10, 0.30, 0.50, 0.75]
ALPHAS = [0.18, 0.38, 0.60, 0.85]


def cond_kde(x, dx, xr, hx, hy=0.12):
    """Conditional density of dx vs continuous x, unit peak per x slice."""
    xg = np.linspace(*xr, NX)
    yg = np.linspace(-YR, YR, NY)
    x, dx = np.asarray(x, float), np.asarray(dx, float)
    dens = np.zeros((NY, NX))
    for j, xc in enumerate(xg):
        w = np.exp(-0.5 * ((x - xc) / hx) ** 2)
        if w.sum() < 3:
            continue
        k = np.exp(-0.5 * ((yg[:, None] - dx[None, :]) / hy) ** 2)
        col = (k * w[None, :]).sum(axis=1)
        if col.max() > 0:
            dens[:, j] = col / col.max()
    return xg, yg, dens


def violin_kde(ax, cats, par_arr, dx_arr, color, width=0.40):
    """Violin-style KDE per category, same graded-alpha look as cond_kde.
    Bandwidth follows Scott's rule per category so low-count categories
    (e.g. carbon recoils) are not over-resolved into noise."""
    yg = np.linspace(-YR, YR, NY)
    for i, (key, _) in enumerate(cats):
        d = np.asarray(dx_arr[np.asarray(par_arr) == key], float)
        if len(d) < 3:
            continue
        hy = max(1.06 * d.std() * len(d) ** (-1 / 5), 0.18)
        k = np.exp(-0.5 * ((yg[:, None] - d[None, :]) / hy) ** 2)
        dens = k.sum(axis=1)
        dens = dens / dens.max()
        dens[dens < 0.03] = 0.0
        for lv, al in zip(LEVELS, ALPHAS):
            band = np.where(dens >= lv, dens, 0.0)
            xw = width * band
            ax.fill_betweenx(yg, i - xw, i + xw, color=color, alpha=al,
                             linewidth=0)
        xw0 = width * dens
        ax.plot(np.where(dens > 0, i - xw0, np.nan), yg, color=color,
                lw=0.7, alpha=0.9)
        ax.plot(np.where(dens > 0, i + xw0, np.nan), yg, color=color,
                lw=0.7, alpha=0.9)
    ax.set_xlim(-0.62, len(cats) - 0.38)
    ax.set_xticks(range(len(cats)))


fig = plt.figure(figsize=(9.4, 6.9))
gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 0.82, 0.045],
                      wspace=0.14, hspace=0.10)

for r, (pop, est, pz, en, par, dx, color) in enumerate(ROWS):
    axd = fig.add_subplot(gs[r, 0])
    axe = fig.add_subplot(gs[r, 1], sharey=axd)
    axp = fig.add_subplot(gs[r, 2], sharey=axd)
    axc = fig.add_subplot(gs[r, 3])

    for ax, (xv, xr, hx, xlab, ticks) in zip(
            (axd, axe),
            [(pz, (0, 20), 1.4, 'interaction depth $z$ (mm)', [0, 5, 10, 15, 20]),
             (en, (1, 10), 0.7, 'neutron energy (MeV)', [2, 4, 6, 8, 10])]):
        xg, yg, dens = cond_kde(xv, dx, xr, hx)
        for lv, al in zip(LEVELS, ALPHAS):
            ax.contourf(xg, yg, dens, levels=[lv, 1.001], colors=[color],
                        alpha=al)
        ax.contour(xg, yg, dens, levels=[LEVELS[0]], colors=[color],
                   linewidths=0.6, alpha=0.9)
        ax.axhline(0, color='0.5', lw=0.5, alpha=0.5)
        ax.set_xlim(*xr)
        ax.set_xticks(ticks)
        ax.set_ylim(-YR, YR)
        ax.set_yticks([-2, -1, 0, 1, 2])
        if r == 3:
            ax.set_xlabel(xlab)
        else:
            ax.tick_params(labelbottom=False)

    violin_kde(axp, PARENTS, par, dx, color)
    axp.axhline(0, color='0.5', lw=0.5, alpha=0.5)
    axp.set_ylim(-YR, YR)
    if r == 3:
        axp.set_xticklabels([lab for _, lab in PARENTS], fontsize=7)
    else:
        axp.tick_params(labelbottom=False)

    axe.tick_params(labelleft=False)
    axp.tick_params(labelleft=False)
    axd.set_ylabel(r'$\Delta x$ (mm)')
    axd.text(0.03, 0.94, f'{pop},  {est}', transform=axd.transAxes,
             ha='left', va='top', fontsize=8, color=color, fontweight='bold')

    cmap = LinearSegmentedColormap.from_list('c', ['white', color])
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,
                                   norm=mpl.colors.Normalize(0, 1))
    cb.set_ticks([0, 0.5, 1])
    if r == 3:
        cb.set_label('relative\ndensity', fontsize=7)
        cb.ax.tick_params(labelsize=7)
    else:
        cb.ax.set_xticklabels([])
        cb.ax.tick_params(labelleft=False, labelright=False, length=2)

axd0 = fig.axes[0]
axd0.set_title('vs. interaction depth', fontsize=9)
fig.axes[1].set_title('vs. neutron energy', fontsize=9)
fig.axes[2].set_title('vs. parent particle', fontsize=9)

save(fig, 'systematics_kde')
print(f'inf N={len(inf)}, oof N={len(oof)}')

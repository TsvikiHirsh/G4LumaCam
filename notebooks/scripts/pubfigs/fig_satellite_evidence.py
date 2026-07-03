#!/usr/bin/env python
"""Publication figure: directional evidence identifying the satellites as
intensifier afterpulses (H3).

(a) satellite azimuth in the lab frame (isotropic -> no preferred track/beam
    axis, excludes H1);
(b) satellite angle to the field-radial direction (flat -> no radial alignment,
    excludes lens aberrations, H2);
(c) mean satellite displacement vs parent distance from the field centre
    (flat, not ~r^2 -> field-independent, excludes H2);
(d) simulation with the afterpulse component: parent->satellite displacement of
    the detected clusters vs the same pairs in the optical-truth frame - the
    satellites have no optical counterpart, i.e. they are injected by the
    detection chain.

Run from notebooks/:  python scripts/pubfigs/fig_satellite_evidence.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pubstyle import EXP, SIM, OPT, REF, panel_label, save

SIM_DIR = 'archive/openbeam_ptb_n1e5/oof_h3opt'
REAL_DIR = '/work/nuclear/PTB/data/air45/out_of_focus'
DET_CENTER = (128.0, 128.0)
R_MIN, R_MAX = 1.5, 40.0


def load_real(d):
    ph = pd.read_csv(f'{d}/ExportedPhotons/photons.csv')
    ph['npx'] = ph['px/id'].astype(str).str.count(r'\|') + 1
    a = pd.read_csv(f'{d}/AssociatedResults/associated_data.csv')
    m = ph.merge(a[['ph/id', 'ev/id']].drop_duplicates(), on='ph/id', how='left')
    ev = pd.read_csv(f'{d}/ExportedEvents/events.csv')[['ev/id', 'ev/x', 'ev/y']]
    m = m.merge(ev, on='ev/id', how='left')
    return m.dropna(subset=['ev/id'])[['ev/id', 'ph/x', 'ph/y', 'npx', 'ev/x', 'ev/y']]


def load_sim(d):
    df = pd.read_csv(f'{d}/AssociatedResults/sim_merged.csv',
                     usecols=['px/id', 'ph/id', 'ev/id', 'ph/x', 'ph/y',
                              'sim/x_opt', 'sim/y_opt'])
    npx = df.groupby('ph/id')['px/id'].nunique().rename('npx')
    ph = (df.drop_duplicates(['ev/id', 'ph/id']).dropna(subset=['ph/x'])
            .join(npx, on='ph/id'))
    return ph.dropna(subset=['ev/id'])


def satellites(m, opt=False):
    sz = m.groupby('ev/id').size()
    m = m[m['ev/id'].isin(sz[sz >= 2].index)].sort_values(
        ['ev/id', 'npx'], ascending=[True, False])
    ren = {'ph/x': 'px0', 'ph/y': 'py0', 'npx': 'npx0'}
    cols = ['px0', 'py0', 'npx0']
    if opt:
        ren['sim/x_opt'], ren['sim/y_opt'] = 'ox0', 'oy0'
        cols += ['ox0', 'oy0']
    parent = m.groupby('ev/id').first().rename(columns=ren)
    sat = m[m.groupby('ev/id').cumcount() >= 1]
    sat = sat.merge(parent[cols], on='ev/id')
    dx = (sat['ph/x'] - sat['px0']).to_numpy()
    dy = (sat['ph/y'] - sat['py0']).to_numpy()
    r = np.hypot(dx, dy)
    keep = (r > R_MIN) & (r < R_MAX) & (sat['npx'].to_numpy() < sat['npx0'].to_numpy())
    res = dict(dx=dx[keep], dy=dy[keep], r=r[keep],
               px0=sat['px0'].to_numpy()[keep], py0=sat['py0'].to_numpy()[keep])
    if opt:
        res['r_opt'] = np.hypot((sat['sim/x_opt'] - sat['ox0']).to_numpy(),
                                (sat['sim/y_opt'] - sat['oy0']).to_numpy())[keep]
    return res


def radial_angle(s):
    rx, ry = s['px0'] - DET_CENTER[0], s['py0'] - DET_CENTER[1]
    rn = np.hypot(rx, ry)
    ok = rn > 5
    cos_r = np.clip(np.abs((s['dx'][ok] * rx[ok] + s['dy'][ok] * ry[ok])
                           / (s['r'][ok] * rn[ok])), 0, 1)
    return np.degrees(np.arccos(cos_r)), rn[ok], s['r'][ok]


rs = satellites(load_real(REAL_DIR))
ss = satellites(load_sim(SIM_DIR), opt=True)
real_az = np.degrees(np.arctan2(rs['dy'], rs['dx']))
sim_az = np.degrees(np.arctan2(ss['dy'], ss['dx']))
R_real = np.hypot(np.cos(np.radians(real_az)).mean(),
                  np.sin(np.radians(real_az)).mean())
real_ar, real_rn, real_dr = radial_angle(rs)
sim_ar, sim_rn, sim_dr = radial_angle(ss)
no_opt = np.mean(ss['r_opt'] < 2) * 100

fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.0))


def dist_panel(a, exp, sim, bins, iso, xlab):
    h, e = np.histogram(exp, bins=bins, density=True)
    a.stairs(h, e, color=EXP, fill=True, alpha=0.25)
    a.stairs(h, e, color=EXP, lw=1.4, label='experiment')
    h, e = np.histogram(sim, bins=bins, density=True)
    a.stairs(h, e, color=SIM, lw=1.4, label='simulation')
    a.axhline(iso, color=REF, ls='--', lw=1.1, label='isotropic')
    a.set_xlabel(xlab)
    a.set_ylabel('probability density')
    a.set_ylim(0, 2.1 * iso)


dist_panel(ax[0, 0], real_az, sim_az, np.linspace(-180, 180, 19), 1 / 360,
           'satellite azimuth, lab frame (deg)')
ax[0, 0].legend(loc='lower center', ncol=3, fontsize=7.5)
ax[0, 0].set_xticks([-180, -90, 0, 90, 180])
panel_label(ax[0, 0], '(a)')

dist_panel(ax[0, 1], real_ar, sim_ar, np.linspace(0, 90, 13), 1 / 90,
           'angle to field-radial direction (deg)')
ax[0, 1].legend(loc='lower center', ncol=3, fontsize=7.5)
panel_label(ax[0, 1], '(b)')

# (c) displacement vs field radius
c = ax[1, 0]
edges = np.linspace(5, 180, 8)
ctr = 0.5 * (edges[:-1] + edges[1:])
for rn, dr, col, lab, mk in [(real_rn, real_dr, EXP, 'experiment', 'o'),
                             (sim_rn, sim_dr, SIM, 'simulation', 's')]:
    idx = np.digitize(rn, edges) - 1
    me = np.array([dr[idx == i].mean() if (idx == i).sum() > 20 else np.nan
                   for i in range(len(ctr))])
    se = np.array([dr[idx == i].std() / np.sqrt(max((idx == i).sum(), 1))
                   if (idx == i).sum() > 20 else np.nan for i in range(len(ctr))])
    c.errorbar(ctr, me, yerr=se, color=col, marker=mk, ms=4, lw=1.3,
               capsize=2, label=lab)
c.plot(ctr, np.nanmean(real_dr) * (ctr / 90) ** 2, ':', color=REF, lw=1.4,
       label=r'lens coma $\propto r^2$')
c.set_ylim(0, 16)
c.set_xlabel('parent distance from field centre (px)')
c.set_ylabel('mean satellite displacement (px)')
c.legend(loc='upper left', bbox_to_anchor=(0.02, 0.92), fontsize=7.5)
panel_label(c, '(c)')

# (d) sim: detected vs optical-truth displacement
d = ax[1, 1]
b = np.linspace(0, 30, 31)
h, e = np.histogram(ss['r_opt'], bins=b, density=True)
d.stairs(h, e, color=OPT, fill=True, alpha=0.25)
d.stairs(h, e, color=OPT, lw=1.5, label='optical truth')
h, e = np.histogram(sim_dr, bins=b, density=True)
d.stairs(h, e, color=SIM, lw=1.5, label='detected clusters')
d.set_xlabel(r'parent $\rightarrow$ satellite displacement (px)')
d.set_ylabel('probability density')
d.set_ylim(0, 0.95)
d.text(0.96, 0.52, f'{no_opt:.0f}% of satellites have\nno optical counterpart',
       transform=d.transAxes, ha='right', va='top', fontsize=8, color='0.2')
d.legend(loc='upper right')
panel_label(d, '(d)')

fig.tight_layout(w_pad=1.6, h_pad=1.6)
save(fig, 'satellite_evidence')
print(f'R={R_real:.3f}  <|cos|>={np.cos(np.radians(real_ar)).mean():.2f}  '
      f'no-opt {no_opt:.0f}%  N exp={len(rs["r"])} sim={len(ss["r"])}')

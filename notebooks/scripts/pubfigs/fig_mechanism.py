#!/usr/bin/env python
"""Publication figure: the intensifier afterpulse mechanism and its imaging
signature.

(a) schematic cross-section of the image-intensifier front end: a scintillation
    photon releases a photoelectron at the photocathode, which crosses the
    proximity gap d and starts the main avalanche in a microchannel-plate (MCP)
    pore (parent cluster on the phosphor). With small probability the electron
    backscatters off the MCP input face, is re-accelerated across the gap, and
    re-enters the MCP displaced by up to ~2d, producing a second, smaller
    avalanche: the satellite cluster.
(b) the resulting camera image: the satellite pulls the charge-weighted event
    centroid away from the true position, while the position of the largest
    (parent) cluster is unaffected.

Run from anywhere:  python fig_mechanism.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from pubstyle import EXP, SIM, OPT, REF, panel_label, save

rng = np.random.default_rng(7)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.3),
                           gridspec_kw={'width_ratios': [1.25, 1]})

# ---------------------------------------------------------------- (a) schematic
a.set_xlim(0, 10)
a.set_ylim(0.2, 10.4)
a.set_aspect('equal')
a.axis('off')

Y0, Y1 = 1.6, 9.4          # component vertical extent
YP, YS = 4.6, 7.0          # parent / satellite beam heights
XPC0, XPC1 = 1.7, 1.95     # photocathode
XM0, XM1 = 3.35, 4.55      # MCP
XPH0, XPH1 = 6.1, 6.35     # phosphor

# components
a.add_patch(Rectangle((XPC0, Y0), XPC1 - XPC0, Y1 - Y0, fc='#C9D8E8', ec='k', lw=0.7))
a.add_patch(Rectangle((XM0, Y0), XM1 - XM0, Y1 - Y0, fc='0.93', ec='k', lw=0.7,
                      hatch='/////'))
a.add_patch(Rectangle((XPH0, Y0), XPH1 - XPH0, Y1 - Y0, fc='#F2E3C0', ec='k', lw=0.7))
a.text((XPC0 + XPC1) / 2, Y0 - 0.35, 'photo-\ncathode', ha='center', va='top', fontsize=7.5)
a.text((XM0 + XM1) / 2, Y0 - 0.35, 'MCP', ha='center', va='top', fontsize=7.5)
a.text((XPH0 + XPH1) / 2, Y0 - 0.35, 'P47\nphosphor', ha='center', va='top', fontsize=7.5)

# incoming photon
a.add_patch(FancyArrowPatch((0.2, YP), (XPC0, YP), arrowstyle='-|>',
                            mutation_scale=9, color=OPT, lw=1.5))
a.text(0.15, YP + 0.35, 'scintillation photon', fontsize=7.5, color=OPT)

# photoelectron across the gap
a.add_patch(FancyArrowPatch((XPC1, YP), (XM0, YP), arrowstyle='-|>',
                            mutation_scale=9, color=EXP, lw=1.5))
a.text((XPC1 + XM0) / 2, YP - 0.75, 'e$^-$', fontsize=8, color=EXP, ha='center')

# main avalanche in the MCP + electron cloud to phosphor
a.fill([XM0, XM1, XM1, XM0], [YP - 0.06, YP - 0.42, YP + 0.42, YP + 0.06],
       color=EXP, alpha=0.55, lw=0)
a.fill([XM1, XPH0, XPH0, XM1], [YP - 0.42, YP - 0.62, YP + 0.62, YP + 0.42],
       color=EXP, alpha=0.28, lw=0)
a.add_patch(Circle((XPH0 + 0.13, YP), 0.42, fc=EXP, ec='none', alpha=0.9))
a.text(XPH1 + 0.25, YP, 'parent cluster', fontsize=7.5, va='center', color=EXP)

# backscattered electron: arc back into the gap, re-entering displaced
a.add_patch(FancyArrowPatch((XM0, YP + 0.15), (XM0, YS), arrowstyle='-|>',
                            mutation_scale=9, color=SIM, lw=1.5, ls=(0, (4, 2)),
                            connectionstyle='arc3,rad=0.65'))
a.text(2.18, (YP + YS) / 2 + 0.9, 'backscattered\ne$^-$', fontsize=7.5,
       color=SIM, ha='center')

# satellite avalanche
a.fill([XM0, XM1, XM1, XM0], [YS - 0.05, YS - 0.30, YS + 0.30, YS + 0.05],
       color=SIM, alpha=0.55, lw=0)
a.fill([XM1, XPH0, XPH0, XM1], [YS - 0.30, YS - 0.45, YS + 0.45, YS + 0.30],
       color=SIM, alpha=0.28, lw=0)
a.add_patch(Circle((XPH0 + 0.13, YS), 0.30, fc=SIM, ec='none', alpha=0.9))
a.text(XPH1 + 0.25, YS, 'satellite', fontsize=7.5, va='center', color=SIM)

# gap and displacement dimensions
a.annotate('', xy=(XPC1, 2.15), xytext=(XM0, 2.15),
           arrowprops=dict(arrowstyle='<->', color=REF, lw=0.9))
a.text((XPC1 + XM0) / 2, 2.35, 'gap $d$', fontsize=7.5, color=REF, ha='center')
a.annotate('', xy=(7.9, YP), xytext=(7.9, YS),
           arrowprops=dict(arrowstyle='<->', color=REF, lw=0.9))
a.text(8.15, (YP + YS) / 2, r'$\lesssim 2d$', fontsize=8, color=REF, va='center')

panel_label(a, '(a)', dx=0.0, dy=1.0)

# ------------------------------------------------------- (b) camera-image view
b.set_aspect('equal')

# synthetic pixelated event: parent blob + satellite mini-blob
ext = 8
grid = np.zeros((2 * ext, 2 * ext))
d_sat = np.array([4.6, 2.6])                      # satellite displacement ~5.3 px
for n, sig, ctr in [(240, 1.05, np.array([0.0, 0.0])),
                    (70, 0.75, d_sat)]:
    pts = rng.normal(ctr, sig, size=(n, 2))
    ix = np.floor(pts[:, 0] + ext).astype(int)
    iy = np.floor(pts[:, 1] + ext).astype(int)
    ok = (ix >= 0) & (ix < 2 * ext) & (iy >= 0) & (iy < 2 * ext)
    np.add.at(grid, (iy[ok], ix[ok]), 1)
grid[grid < 3] = 0                                 # threshold: discrete pixels

b.imshow(grid, origin='lower', extent=[-ext, ext, -ext, ext],
         cmap='Greys', vmax=grid.max() * 1.15, interpolation='nearest')

# positions: truth (= parent centre), pulled centroid, largest cluster
yy, xx = np.mgrid[-ext:ext, -ext:ext]
xx, yy = xx + 0.5, yy + 0.5
w = grid.sum()
cog = (float((grid * xx).sum() / w), float((grid * yy).sum() / w))
b.plot(0, 0, 'o', mfc='none', mec=OPT, mew=1.8, ms=11, label='true position')
b.plot(*cog, 'X', color=SIM, ms=9, label='event centroid')
b.plot(0, 0, '+', color='k', ms=12, mew=1.8, label='largest cluster')
b.add_patch(FancyArrowPatch((0.4, 0.25), (cog[0] - 0.25, cog[1] - 0.15),
                            arrowstyle='-|>', mutation_scale=10, color=SIM,
                            lw=1.2))
b.text(d_sat[0] + 0.3, d_sat[1] + 1.6, 'satellite', fontsize=7.5, ha='center',
       color=SIM)
b.text(-0.2, -2.6, 'parent', fontsize=7.5, ha='center')
b.set_xlabel('x (px)')
b.set_ylabel('y (px)')
b.set_xticks([-8, -4, 0, 4, 8])
b.set_yticks([-8, -4, 0, 4, 8])
b.legend(loc='upper left', fontsize=7, handletextpad=0.4, borderaxespad=0.2)
panel_label(b, '(b)', dx=0.03, dy=0.14)

fig.tight_layout(w_pad=1.6)
save(fig, 'afterpulse_mechanism')
print('centroid pull:', np.hypot(*cog).round(2), 'px')

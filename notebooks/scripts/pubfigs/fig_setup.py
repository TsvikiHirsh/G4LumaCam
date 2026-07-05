#!/usr/bin/env python
"""Publication figure: schematic of the fast-neutron resonance imaging setup.

The full detection chain at the PTB fast-neutron beamline. The LumaCam is
drawn with its distinctive enclosure: the scintillator sits at the entrance
face, a 45-degree mirror folds the light path out of the beam, and the
lens / MCP-intensifier / Timepix3 stack protrudes from the top of the box.

Run from anywhere:  python fig_setup.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, Circle, FancyArrowPatch,
                                FancyBboxPatch, Polygon, Ellipse)
from pubstyle import EXP, SIM, OPT, REF, save

fig, ax = plt.subplots(figsize=(9.6, 4.0))
ax.set_xlim(0, 30)
ax.set_ylim(-2.4, 13.2)
ax.set_aspect('equal')
ax.axis('off')

YB = 4.2                       # beam axis height
SHADOW = dict(fc='0.82', ec='none')


def slab(x, y, w, h, fc, ec='0.25', lw=0.8, dz=0.10, **kw):
    """Component slab with a soft drop shadow."""
    ax.add_patch(Rectangle((x + dz, y - dz), w, h, **SHADOW))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, **kw))


def clabel(x, name, spec, y=-0.35):
    ax.text(x, y, name, ha='center', va='top', fontsize=8, fontweight='bold',
            color='0.15')
    ax.text(x, y - 0.62, spec, ha='center', va='top', fontsize=7, color='0.40')


# ---- pulsed neutron source -------------------------------------------------
ax.add_patch(Circle((1.7, YB), 0.40, fc='#7A2E1D', ec='none'))
ax.add_patch(Circle((1.7, YB), 0.28, fc=SIM, ec='none'))
for aa in np.linspace(0, 2 * np.pi, 12, endpoint=False):
    r0, r1 = 0.55, 0.55 + (0.42 if aa % (np.pi / 3) < 0.1 else 0.22)
    ax.plot([1.7 + r0 * np.cos(aa), 1.7 + r1 * np.cos(aa)],
            [YB + r0 * np.sin(aa), YB + r1 * np.sin(aa)], color=SIM, lw=1.0,
            solid_capstyle='round')
clabel(1.9, 'pulsed source', '$^{11}$B(d,n)\n1--20 MeV')

# ---- flight path (beam band with break marks) -------------------------------
ax.add_patch(Polygon([(3.0, YB - 0.30), (9.0, YB - 0.42), (9.0, YB + 0.42),
                      (3.0, YB + 0.30)], closed=True, fc='0.85', ec='none'))
ax.add_patch(FancyArrowPatch((8.35, YB), (9.05, YB), arrowstyle='-|>',
                             mutation_scale=13, color='0.45', lw=1.6))
for xb in (5.5, 5.95):
    ax.plot([xb - 0.16, xb + 0.16], [YB - 0.55, YB + 0.55], color='white',
            lw=4.0, zorder=3)
    ax.plot([xb - 0.16, xb + 0.16], [YB - 0.55, YB + 0.55], color='0.45',
            lw=1.0, zorder=4)
ax.text(5.75, 5.35, 'time of flight over  $L = 10.85$ m', ha='center',
        fontsize=7.5, color='0.30')
ax.text(5.75, 3.15, r'$E_n \leftarrow$ TOF', ha='center', fontsize=7.5,
        color='0.30', style='italic')

# beam continues from the graphite into the detector
ax.add_patch(FancyArrowPatch((10.95, YB), (12.05, YB), arrowstyle='-|>',
                             mutation_scale=10, color='0.55', lw=1.2))

# ---- graphite knife edge (upper half of the beam) ---------------------------
slab(9.4, YB - 0.05, 1.4, 3.1, fc='0.60')
ax.text(10.1, 7.85, 'graphite', ha='center', va='bottom', fontsize=8,
        fontweight='bold', color='0.15')
ax.text(10.1, 7.35, 'knife edge', ha='center', va='bottom', fontsize=7,
        color='0.40')

# ================= LumaCam enclosure (distinctive folded-optics box) =========
BX0, BX1 = 12.2, 20.6          # box footprint
BY0, BY1 = 0.9, 7.6
ax.add_patch(FancyBboxPatch((BX0 + 0.12, BY0 - 0.12), BX1 - BX0, BY1 - BY0,
                            boxstyle='round,pad=0.10', fc='0.82', ec='none'))
ax.add_patch(FancyBboxPatch((BX0, BY0), BX1 - BX0, BY1 - BY0,
                            boxstyle='round,pad=0.10', fc='#3A3F45',
                            ec='0.15', lw=1.0))
ax.text(BX1 - 0.35, BY0 + 0.42, 'LumaCam', ha='right', fontsize=8.5,
        color='white', fontweight='bold', style='italic')

# scintillator at the entrance face
ax.add_patch(Rectangle((BX0 + 0.45, 1.5), 1.05, 5.4, fc='#BBD3E8', ec='k',
                       lw=0.7))
ax.add_patch(Rectangle((BX0 + 0.45, 1.5), 0.24, 5.4, fc='#9FC0DC', ec='none'))
# neutron interaction: glow + star
ix, iy = BX0 + 0.97, YB - 0.4
for r, al in [(0.50, 0.22), (0.34, 0.38), (0.20, 0.60)]:
    ax.add_patch(Circle((ix, iy), r, fc=SIM, ec='none', alpha=al))
ax.plot(ix, iy, marker='*', ms=8, color='white', zorder=6)

# 45-degree mirror folding the light path upward
mcx, mcy = 17.6, iy            # mirror centre on the optical axis
ml = 1.55                       # mirror half-length
ax.add_patch(Polygon([(mcx - ml * 0.707 + 0.28, mcy - ml * 0.707 - 0.10),
                      (mcx + ml * 0.707 + 0.28, mcy + ml * 0.707 - 0.10),
                      (mcx + ml * 0.707, mcy + ml * 0.707),
                      (mcx - ml * 0.707, mcy - ml * 0.707)],
                     closed=True, fc='#C7D6E2', ec='0.3', lw=0.8))
ax.plot([mcx - ml * 0.707, mcx + ml * 0.707],
        [mcy - ml * 0.707, mcy + ml * 0.707], color='white', lw=1.6)
ax.text(mcx + 1.15, mcy - 1.15, "mirror", fontsize=7, color='white',
        ha='center')

# horizontal light cone: interaction -> mirror
cone = Polygon([(ix + 0.25, iy), (mcx - 0.25, iy + 1.05),
                (mcx - 0.25, iy - 1.05)], closed=True, fc=OPT, ec='none',
               alpha=0.20)
ax.add_patch(cone)
for s in (+1, -1):
    ax.plot([ix + 0.25, mcx - 0.25], [iy, iy + 1.05 * s], color=OPT, lw=0.9,
            alpha=0.85)

# vertical light cone: mirror -> lens (folded 90 degrees, out of the beam)
LY = 6.15                       # lens height
cone2 = Polygon([(mcx - 1.05, iy + 0.25), (mcx + 1.05, iy + 0.25),
                 (mcx + 0.55, LY), (mcx - 0.55, LY)], closed=True,
                fc=OPT, ec='none', alpha=0.20)
ax.add_patch(cone2)
for s in (+1, -1):
    ax.plot([mcx + 1.05 * s, mcx + 0.55 * s], [iy + 0.25, LY], color=OPT,
            lw=0.9, alpha=0.85)

# lens (horizontal, light travels upward)
ax.add_patch(Ellipse((mcx, LY + 0.15), 2.4, 0.6, fc='#D9EAD9', ec=OPT, lw=1.2))

# MCP intensifier stack (photocathode | MCP | phosphor), horizontal layers
for y0, h, fc, ht in [(6.75, 0.16, '#9FC0DC', None),
                      (6.98, 0.42, '0.94', '/////'),
                      (7.47, 0.16, '#F2DFA9', None)]:
    ax.add_patch(Rectangle((mcx - 1.15, y0), 2.3, h, fc=fc, ec='0.4', lw=0.5,
                           hatch=ht))

# TPX3 camera protruding from the enclosure top
slab(mcx - 1.35, 7.95, 2.7, 2.3, fc='0.28')
ax.add_patch(Rectangle((mcx - 0.95, 8.25), 1.9, 1.7, fc='#3F5E77', ec='none'))
ax.text(mcx, 9.1, 'TPX3', ha='center', va='center', fontsize=8.5,
        color='white', fontweight='bold')

# component callouts on the right of the folded stack
for ytxt, txt in [(6.15, 'lens 50 mm $f/0.95$'),
                  (7.2, 'MCP intensifier + P47'),
                  (9.1, 'event camera')]:
    ax.annotate(txt, xy=(mcx + (1.45 if ytxt < 8 else 1.55), ytxt),
                xytext=(21.6, ytxt), fontsize=7, color='0.25', va='center',
                arrowprops=dict(arrowstyle='-', color='0.55', lw=0.7))

clabel(13.0, 'EJ-200 scintillator', '20 mm thick\n$120\\times120$ mm$^2$ FOV')

# ---- per-photon data product ----------------------------------------------
ax.add_patch(FancyArrowPatch((mcx + 1.5, 9.6), (27.2, 9.6), arrowstyle='-|>',
                             mutation_scale=11, color=REF, lw=1.2))
ax.add_patch(FancyBboxPatch((27.4, 8.75), 2.1, 1.7,
                            boxstyle='round,pad=0.12', fc='white', ec='0.45',
                            lw=0.9))
ax.text(28.45, 9.98, 'per photon', ha='center', fontsize=7, color='0.40')
ax.text(28.45, 9.28, r'$(x,\,y,\,t)$', ha='center', fontsize=9.5,
        color='0.15')

fig.tight_layout()
save(fig, 'setup_schematic')

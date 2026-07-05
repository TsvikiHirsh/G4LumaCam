#!/usr/bin/env python
"""Publication figure: schematic of the fast-neutron resonance imaging setup.

The full LumaCam detection chain at the PTB fast-neutron beamline, with the
system components that make FNRI demanding highlighted: a thick scintillator
for MeV-neutron efficiency, a fast large-aperture lens, a single-photon MCP
image intensifier, and an event-mode Timepix3 camera with nanosecond
timestamps, over a large field of view.

Run from anywhere:  python fig_setup.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (Rectangle, Circle, FancyArrowPatch, Ellipse,
                                FancyBboxPatch, Polygon)
from pubstyle import EXP, SIM, OPT, REF, save

fig, ax = plt.subplots(figsize=(9.6, 3.2))
ax.set_xlim(0, 30)
ax.set_ylim(-2.2, 9.8)
ax.set_aspect('equal')
ax.axis('off')

YB = 4.6                       # beam / optical axis height
SHADOW = dict(fc='0.82', ec='none')


def slab(x, y, w, h, fc, ec='0.25', lw=0.8, dz=0.10, **kw):
    """Component slab with a soft drop shadow."""
    ax.add_patch(Rectangle((x + dz, y - dz), w, h, **SHADOW))
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, **kw))


def clabel(x, name, spec, y=-0.15):
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
    ax.plot([xb - 0.16, xb + 0.16], [YB - 0.55, YB + 0.55], color='white', lw=4.0,
            zorder=3)
    ax.plot([xb - 0.16, xb + 0.16], [YB - 0.55, YB + 0.55], color='0.45', lw=1.0,
            zorder=4)
ax.text(5.75, 5.55, 'time of flight over  $L = 10.85$ m', ha='center',
        fontsize=7.5, color='0.30')
ax.text(5.75, 3.55, r'$E_n \leftarrow$ TOF', ha='center', fontsize=7.5,
        color='0.30', style='italic')

# ---- graphite knife edge (upper half of the beam) ---------------------------
slab(9.3, YB - 0.05, 1.5, 3.3, fc='0.60')
ax.text(10.05, 9.6, 'graphite', ha='center', va='top', fontsize=8,
        fontweight='bold', color='0.15')
ax.text(10.05, 8.95, 'knife edge', ha='center', va='top', fontsize=7,
        color='0.40')

# ---- scintillator ------------------------------------------------------------
slab(12.3, 0.9, 1.4, 7.2, fc='#BBD3E8')
ax.add_patch(Rectangle((12.3, 0.9), 0.28, 7.2, fc='#9FC0DC', ec='none'))
clabel(13.0, 'EJ-200 scintillator', '20 mm thick\n$120\\times120$ mm$^2$ FOV')
# neutron interaction: glow + star
ix, iy = 13.0, YB - 0.65
for r, al in [(0.55, 0.18), (0.38, 0.30), (0.22, 0.55)]:
    ax.add_patch(Circle((ix, iy), r, fc=SIM, ec='none', alpha=al))
ax.plot(ix, iy, marker='*', ms=8, color='white', zorder=6)

# ---- light cone to the lens --------------------------------------------------
lx = 16.8
cone = Polygon([(ix + 0.25, iy), (lx, iy + 1.75), (lx, iy - 1.75)],
               closed=True, fc=OPT, ec='none', alpha=0.16)
ax.add_patch(cone)
for s in (+1, -1):
    ax.plot([ix + 0.25, lx], [iy, iy + 1.75 * s], color=OPT, lw=1.0, alpha=0.8)

# ---- lens ---------------------------------------------------------------------
ax.add_patch(Ellipse((lx + 0.1, iy - 0.1), 0.70, 3.9, fc='0.82', ec='none'))
ax.add_patch(Ellipse((lx, iy), 0.70, 3.9, fc='#D9EAD9', ec=OPT, lw=1.3))
ax.add_patch(Ellipse((lx, iy), 0.30, 3.9, fc='white', ec='none', alpha=0.45))
clabel(lx, 'fast lens', '50 mm\n$f/0.95$')

# ---- converging cone to intensifier -------------------------------------------
mx = 19.8
cone2 = Polygon([(lx + 0.3, iy + 1.70), (lx + 0.3, iy - 1.70),
                 (mx, iy - 0.30), (mx, iy + 0.30)],
                closed=True, fc=OPT, ec='none', alpha=0.16)
ax.add_patch(cone2)
for s in (+1, -1):
    ax.plot([lx + 0.3, mx], [iy + 1.70 * s, iy + 0.30 * s], color=OPT, lw=1.0,
            alpha=0.8)

# ---- MCP image intensifier (photocathode | MCP | phosphor) ---------------------
slab(mx, iy - 2.0, 1.95, 4.0, fc='#EDEDED')
ax.add_patch(Rectangle((mx + 0.10, iy - 1.75), 0.20, 3.5, fc='#9FC0DC', ec='0.4',
                       lw=0.5))
ax.add_patch(Rectangle((mx + 0.55, iy - 1.75), 0.80, 3.5, fc='0.94', ec='0.4',
                       lw=0.5, hatch='/////'))
ax.add_patch(Rectangle((mx + 1.60, iy - 1.75), 0.20, 3.5, fc='#F2DFA9', ec='0.4',
                       lw=0.5))
clabel(mx + 0.85, 'intensifier', 'single-photon\nMCP + P47')

# ---- relay + TPX3 camera -------------------------------------------------------
ax.add_patch(FancyArrowPatch((mx + 2.15, iy), (23.1, iy), arrowstyle='-|>',
                             mutation_scale=11, color=REF, lw=1.2))
slab(23.2, iy - 1.65, 2.9, 3.3, fc='0.28')
ax.add_patch(Rectangle((23.55, iy - 1.10), 2.2, 2.2, fc='#3F5E77', ec='none'))
ax.text(24.65, iy, 'TPX3', ha='center', va='center', fontsize=9, color='white',
        fontweight='bold')
clabel(24.85, 'event camera', '$256\\times256$ px\n1.5625 ns bins')

# ---- per-photon data product ----------------------------------------------------
ax.add_patch(FancyArrowPatch((26.35, iy), (27.4, iy), arrowstyle='-|>',
                             mutation_scale=11, color=REF, lw=1.2))
ax.add_patch(FancyBboxPatch((27.6, iy - 0.85), 2.1, 1.7,
                            boxstyle='round,pad=0.12',
                            fc='white', ec='0.45', lw=0.9))
ax.text(28.65, iy + 0.38, 'per photon', ha='center', fontsize=7, color='0.40')
ax.text(28.65, iy - 0.32, r'$(x,\,y,\,t)$', ha='center', fontsize=9.5,
        color='0.15')

fig.tight_layout()
save(fig, 'setup_schematic')

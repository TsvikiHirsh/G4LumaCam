"""Shared publication style for the manuscript figures.

Palette (Okabe-Ito subset, CVD-validated all-pairs):
  EXP  #0072B2  experiment (filled step)
  SIM  #D55E00  simulation / detector quantities (solid line)
  OPT  #009E73  simulation optical truth (line)
  REF  #555555  reference / bound lines (dashed, neutral)
Conventions: ticks in, no chartjunk titles (captions carry the message),
panel letters "(a)" top-left, recessive grid, vector PDF + 300 dpi PNG.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = '#0072B2'
SIM = '#D55E00'
OPT = '#009E73'
REF = '#555555'

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      '..', '..', '..', 'figures', 'pub')

plt.rcParams.update({
    'font.size': 8.5,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.6,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
    'legend.frameon': False,
    'legend.fontsize': 8,
    'grid.alpha': 0.22,
    'grid.linewidth': 0.6,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
})


def panel_label(ax, s, dx=0.02, dy=0.97):
    ax.text(dx, dy, s, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, fontweight='bold')


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext, kw in (('pdf', {}), ('png', {'dpi': 300})):
        fig.savefig(os.path.join(OUTDIR, f'{name}.{ext}'), **kw)
    print(f'saved {os.path.join(OUTDIR, name)}.pdf/.png')

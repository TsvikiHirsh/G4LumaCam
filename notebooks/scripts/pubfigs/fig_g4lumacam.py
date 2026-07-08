#!/usr/bin/env python
"""Publication figure: composite of the G4LumaCam visualization panels.

(a) Geant4 event display: a neutron interaction in the EJ-200 slab and the
    resulting scintillation-photon cascade, auto-cropped and annotated.
(b) ray-optics trace of the 50 mm f/0.95 objective, auto-cropped and
    annotated.

Run from notebooks/:  python scripts/pubfigs/fig_g4lumacam.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pubstyle import panel_label, save

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   '..', '..', '..', '..', '..', 'papers', 'PTB_paper_overleaf')


def autocrop(path, pad=6, thresh=245):
    im = Image.open(path).convert('RGB')
    a = np.asarray(im)
    mask = (a.min(axis=2) < thresh)
    ys, xs = np.where(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, a.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, a.shape[1])
    return a[y0:y1, x0:x1]


sim = autocrop(os.path.join(SRC, 'lumacam_simulation.png'))
lens = autocrop(os.path.join(SRC, 'lens.png'))

fig, (a, b) = plt.subplots(
    2, 1, figsize=(7.0, 4.6),
    gridspec_kw={'height_ratios': [sim.shape[0] / sim.shape[1],
                                   1.55 * lens.shape[0] / lens.shape[1]]})

a.imshow(sim)
a.axis('off')
AN = dict(fontsize=7.5, color='0.15',
          arrowprops=dict(arrowstyle='-', color='0.45', lw=0.7))
h, w = sim.shape[:2]
a.annotate('incident neutron', xy=(0.115 * w, 0.20 * h),
           xytext=(0.26 * w, 0.06 * h), **AN)
a.annotate('EJ-200 scintillator', xy=(0.245 * w, 0.42 * h),
           xytext=(0.03 * w, 0.68 * h), **AN)
a.annotate('scintillation photons', xy=(0.50 * w, 0.34 * h),
           xytext=(0.60 * w, 0.14 * h), **AN)
a.annotate('mirror', xy=(0.42 * w, 0.62 * h),
           xytext=(0.53 * w, 0.80 * h), **AN)
panel_label(a, '(a)', dx=-0.03, dy=1.0)

b.imshow(lens)
b.axis('off')
h, w = lens.shape[:2]
b.annotate('rays from three field points\non the scintillator',
           xy=(0.16 * w, 0.30 * h), xytext=(0.30 * w, 0.12 * h), **AN)
b.annotate('50 mm $f/0.95$ objective', xy=(0.80 * w, 0.86 * h),
           xytext=(0.56 * w, 0.90 * h), **AN)
b.annotate('image plane', xy=(0.968 * w, 0.36 * h),
           xytext=(0.845 * w, 0.10 * h), **AN)
panel_label(b, '(b)', dx=-0.03, dy=1.0)

fig.tight_layout(h_pad=1.0)
save(fig, 'g4lumacam_panels')

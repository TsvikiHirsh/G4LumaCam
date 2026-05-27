"""
Where in the optical path are rays getting blocked, and is the 1.000 mm
'placeholder' max_aperture from the .zmx really acting as the dominant stop?

For each kind/fnumber we:
  1. Build the lens (now with do_apertures=True).
  2. Dump per-surface max_aperture to spot any anomalously small surface.
  3. Trace a 5 deg half-cone of rays from on-axis with check_apertures=True
     and tally where each blocked ray died (TraceRayBlockedError surface idx).
  4. Compare to the geometric expectation: solid angle subtended by entrance
     pupil from the object point should set the upper bound on transmission.
"""
import sys, math
sys.path.insert(0, "/work/nuclear/G4LumaCam/src")
import numpy as np
from collections import Counter
from lumacam.optics import Lens
from rayoptics.raytr import analyses
from rayoptics.raytr import trace as rt_trace

def cone(px, py, half_ang, n=2000, wvl=587.5618):
    rng = np.random.default_rng(0)
    s = math.sin(half_ang)
    r = s * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2*np.pi, n)
    dx = r*np.cos(th); dy = r*np.sin(th)
    dz = np.sqrt(np.maximum(1 - dx*dx - dy*dy, 1e-12))
    return [(np.array([px, py, 0.0]),
             np.array([dx[i], dy[i], dz[i]]),
             float(wvl)) for i in range(n)]

def diag(kind, fnumbers, n_rays=2000, half_deg=5.0):
    print(f"\n{'='*78}\n{kind}\n{'='*78}")
    for fn in fnumbers:
        L = Lens(data=None_df, kind=kind, fnumber=fn, zfine=0)
        sm = L.opm.seq_model
        # surface apertures
        aps = []
        for i, ifc in enumerate(sm.ifcs):
            try: aps.append((i, float(ifc.max_aperture)))
            except: pass
        aps_sorted = sorted(aps, key=lambda x: x[1])
        small5 = aps_sorted[:5]

        rays = cone(0.0, 0.0, math.radians(half_deg), n=n_rays)
        result = analyses.trace_list_of_rays(
            L.opm, rays, output_filter=None, rayerr_filter="full",
            check_apertures=True,
        )
        # tally pass/fail and where failures occurred
        n_pass = 0; fail_surf = Counter(); fail_kind = Counter()
        for r in result:
            if r is None: continue
            # success → tuple (ray, op_delta, wvl)
            # full failure → (input_ray, error) where error has .surf attr
            if hasattr(r, "__len__") and len(r) == 3 and not hasattr(r[1], "surf"):
                n_pass += 1
            else:
                err = r[1]
                surf = getattr(err, "surf", -1)
                fail_surf[surf] += 1
                fail_kind[type(err).__name__] += 1
        eff = 100.0 * n_pass / n_rays
        print(f"\n  f/{fn:.2f}  -> {n_pass}/{n_rays} rays pass ({eff:.1f}%)")
        print(f"  5 smallest surface clear apertures: {small5}")
        # top failure surfaces
        top_fails = fail_surf.most_common(3)
        print(f"  top blocking surfaces: {top_fails}")
        print(f"  failure kinds: {dict(fail_kind)}")


None_df = None  # Lens accepts data=None for the convenience constructor below

# Need at least an empty DataFrame to satisfy the Lens constructor
import pandas as pd
None_df = pd.DataFrame({c: [] for c in
    ['x','y','z','dx','dy','dz','wavelength','nz','pz','id','neutron_id','pulse_id','toa']})

diag("nikkor_58mm", [0.98, 1.1, 2.0, 4.0, 8.0, 16.0])
diag("microscope",  [2.0, 4.0, 8.0, 16.0])


# Also compute the geometric upper bound: the entrance pupil subtends a solid
# angle from the object point of roughly omega = pi * (EPD/2 / dist)^2.
# If a point source emits N rays into our 5 deg test cone, the fraction that
# *geometrically* can hit the entrance pupil is omega_pupil / omega_cone.
print("\n" + "="*78)
print("Geometric ceiling on transmission for our 5 deg test cone")
print("="*78)
for kind, dist, foc_len_guess in [("nikkor_58mm", 461.535, 58.0),
                                   ("microscope",  41.0,    50.0)]:
    print(f"\n{kind}  obj_dist={dist} mm  est focal length~{foc_len_guess} mm")
    cone_omega = 2*math.pi*(1 - math.cos(math.radians(5.0)))
    for fn in [0.98, 1.1, 2.0, 4.0, 8.0, 16.0]:
        epd = foc_len_guess / fn
        # solid angle subtended by entrance pupil of radius epd/2 at distance dist
        # small-angle: omega ~ pi*r^2/dist^2
        omega_pup = math.pi * (epd/2)**2 / dist**2
        ratio = min(omega_pup / cone_omega, 1.0)
        print(f"  f/{fn:5.2f}  EPD={epd:6.2f} mm  "
              f"omega_pupil/omega_cone = {ratio*100:5.1f}%  (rough upper bound)")

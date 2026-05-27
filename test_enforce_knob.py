"""Sanity-check the enforce_aperture knob: True = new behavior with fnumber DoF,
False = pre-branch behavior where fnumber is cosmetic and all rays pass."""
import sys, math, time
sys.path.insert(0, "/work/nuclear/G4LumaCam/src")
import numpy as np, pandas as pd
from lumacam.optics import Lens
from rayoptics.raytr import analyses

empty = pd.DataFrame({c: [] for c in
    ['x','y','z','dx','dy','dz','wavelength','nz','pz','id','neutron_id','pulse_id','toa']})

def cone(n, ang_deg=5.0, wvl=587.5618):
    rng = np.random.default_rng(0)
    s = math.sin(math.radians(ang_deg))
    r = s * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2*np.pi, n)
    dx = r*np.cos(th); dy = r*np.sin(th)
    dz = np.sqrt(np.maximum(1 - dx*dx - dy*dy, 1e-12))
    return [(np.array([0.0, 0.0, 0.0]),
             np.array([dx[i], dy[i], dz[i]]), float(wvl)) for i in range(n)]

n = 1000
print(f"Tracing {n} rays in a 5 deg cone, on-axis source, "
      "comparing enforce_aperture True vs False")
print()
print(f"{'kind':<12} {'fnumber':>8} {'mode':>10} {'check_ap':>10} "
      f"{'pass':>6} {'time_ms':>9}")
print("-"*72)

for kind in ["nikkor_58mm", "microscope"]:
    for fn in [1.1, 2.0, 8.0]:
        for enforce in [True, False]:
            L = Lens(data=empty, kind=kind, fnumber=fn, zfine=0,
                     enforce_aperture=enforce)
            rays = cone(n)
            t = time.perf_counter()
            out = analyses.trace_list_of_rays(
                L.opm, rays, output_filter="last", rayerr_filter="summary",
                # Mirror what the worker now does (check_apertures driven by flag)
                check_apertures=L.enforce_aperture,
            )
            dt = time.perf_counter() - t
            n_pass = sum(1 for r in out if r is not None
                         and hasattr(r, "__len__") and len(r)==3
                         and not hasattr(r[1], "surf"))
            print(f"{kind:<12} f/{fn:<5.2f}  {'enforce' if enforce else 'legacy':>10}  "
                  f"{str(L.enforce_aperture):>10}  {n_pass:>6} {dt*1000:>9.1f}")
    print()

# Quick check: legacy mode at all fnumbers should give same result
print("Sanity: legacy mode (enforce_aperture=False) should be fnumber-insensitive")
for kind in ["nikkor_58mm", "microscope"]:
    rates = []
    for fn in [1.1, 2.0, 8.0]:
        L = Lens(data=empty, kind=kind, fnumber=fn, zfine=0, enforce_aperture=False)
        out = analyses.trace_list_of_rays(L.opm, cone(n),
            output_filter="last", rayerr_filter="summary",
            check_apertures=False)
        npass = sum(1 for r in out if r is not None
                    and hasattr(r, "__len__") and len(r)==3
                    and not hasattr(r[1], "surf"))
        rates.append((fn, npass))
    pass_set = set(p for _, p in rates)
    print(f"  {kind:<12}: {rates}  "
          f"{'OK (fnumber cosmetic)' if len(pass_set)==1 else 'WARN: legacy not invariant'}")

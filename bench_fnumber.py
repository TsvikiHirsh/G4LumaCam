"""Small benchmark with flushing so we can see progress."""
import sys, time, math
sys.path.insert(0, "/work/nuclear/G4LumaCam/src")
import numpy as np, pandas as pd
from lumacam.optics import Lens
from rayoptics.raytr import analyses

def rays(n, ang=5.0, wvl=587.5618, seed=0):
    rng = np.random.default_rng(seed)
    s = math.sin(math.radians(ang))
    r = s * np.sqrt(rng.uniform(0, 1, n))
    th = rng.uniform(0, 2*np.pi, n)
    dx = r*np.cos(th); dy = r*np.sin(th)
    dz = np.sqrt(np.maximum(1 - dx*dx - dy*dy, 1e-12))
    px = rng.uniform(-2, 2, n); py = rng.uniform(-2, 2, n)
    return [(np.array([px[i], py[i], 0.0]),
             np.array([dx[i], dy[i], dz[i]]),
             float(wvl)) for i in range(n)]

empty = pd.DataFrame({c: [] for c in
    ['x','y','z','dx','dy','dz','wavelength','nz','pz','id','neutron_id','pulse_id','toa']})

# Build once
print("building lenses…", flush=True)
L_n = Lens(data=empty, kind="nikkor_58mm", fnumber=2.0, zfine=0)
L_m = Lens(data=empty, kind="microscope",  fnumber=8.0, zfine=0)
print("built. starting trace timing.", flush=True)

# Try just 200 rays under both conditions, both lenses
for label, L in [("nikkor", L_n), ("microscope", L_m)]:
    R = rays(200)
    for ck in [False, True]:
        t0 = time.perf_counter()
        out = analyses.trace_list_of_rays(L.opm, R,
            output_filter="last", rayerr_filter="summary",
            check_apertures=ck)
        dt = time.perf_counter() - t0
        npass = sum(1 for x in out if x is not None
                    and hasattr(x, "__len__") and len(x)==3
                    and not hasattr(x[1], "surf"))
        print(f"  {label:<10} check_apertures={str(ck):<5}  "
              f"{dt*1000:7.1f} ms  ({200/dt:>6.0f} rays/s)  pass={npass}",
              flush=True)

# Now scale to 2000 to see if it scales
print("\nScaling test, 2000 rays:", flush=True)
for label, L in [("nikkor", L_n), ("microscope", L_m)]:
    R = rays(2000)
    for ck in [False, True]:
        t0 = time.perf_counter()
        out = analyses.trace_list_of_rays(L.opm, R,
            output_filter="last", rayerr_filter="summary",
            check_apertures=ck)
        dt = time.perf_counter() - t0
        print(f"  {label:<10} check_apertures={str(ck):<5}  "
              f"{dt*1000:7.1f} ms  ({2000/dt:>6.0f} rays/s)",
              flush=True)

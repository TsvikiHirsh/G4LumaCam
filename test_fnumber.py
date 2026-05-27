"""
F-number reality check for G4LumaCam optical models.

We're checking three things, in order of increasing physical realism:

1.  Does the `fnumber` argument actually reach `opm.optical_spec.pupil.value`?
    (i.e. is the bug at optics.py:302 real, and is the remap at optics.py:239 real?)

2.  Does `fnumber` clip any rays during the kind of tracing the codebase
    actually performs --- `analyses.trace_list_of_rays` with NO
    `check_apertures=True` (see optics.py:158) and `sm.do_apertures = False`?

3.  Does `fnumber` change the geometric spot size at the focal plane?
    A real iris-stopped lens narrows the cone of rays from each object point,
    so a defocused point should make a smaller blur circle at smaller aperture.
    We test this by emitting a fan of rays from a single object point and
    looking at the RMS spread at the image plane.

We also do a parallel run with `check_apertures=True` to show what
fnumber WOULD do if aperture enforcement were turned on.
"""
import numpy as np
import math, sys
# Force the in-tree source over any installed copy so we test what we edit.
sys.path.insert(0, "/work/nuclear/G4LumaCam/src")
from lumacam.optics import Lens
from rayoptics.raytr import analyses
import pandas as pd

def fake_df(n=10):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'x': rng.uniform(-5, 5, n), 'y': rng.uniform(-5, 5, n),
        'z': np.zeros(n),
        'dx': rng.uniform(-0.01, 0.01, n), 'dy': rng.uniform(-0.01, 0.01, n),
        'dz': np.ones(n), 'wavelength': np.full(n, 550.0),
        'nz': np.zeros(n), 'pz': np.zeros(n), 'id': np.arange(n),
        'neutron_id': np.zeros(n, int), 'pulse_id': np.zeros(n, int),
        'toa': np.zeros(n),
    })


# Build a cone of rays leaving a single object point (px, py, 0) at angles
# up to max_half_angle_rad. We emit them isotropically over a flat angle disk
# (uniform in dx, dy). We oversample then take only the ones inside the cone.
def cone_rays(px, py, max_half_angle_rad, n_rays=400, wvl=587.5618):
    """Build n_rays leaving (px,py,0) in a uniform-flux cone of given half-angle.
    wvl defaults to the central F-line wavelength used by the WvlSpec in both
    lens models so seq_model.index_for_wavelength() succeeds."""
    rng = np.random.default_rng(0)
    s_max = math.sin(max_half_angle_rad)
    r = s_max * np.sqrt(rng.uniform(0, 1, n_rays))
    theta = rng.uniform(0, 2*np.pi, n_rays)
    dx = r * np.cos(theta); dy = r * np.sin(theta)
    dz = np.sqrt(np.maximum(1.0 - dx*dx - dy*dy, 1e-12))
    rays = []
    for i in range(n_rays):
        rays.append(
            (np.array([px, py, 0.0]),
             np.array([dx[i], dy[i], dz[i]]),
             float(wvl))
        )
    return rays


def trace_and_collect_final_xy(opm, rays, check_apertures=False):
    """Return (x_img, y_img) arrays for rays that successfully reached the image."""
    result = analyses.trace_list_of_rays(
        opm, rays, output_filter="last", rayerr_filter="summary",
        check_apertures=check_apertures,
    )
    xs, ys = [], []
    for r in result:
        if r is None:
            continue
        # r = (ray_at_last_ifc, op_delta, wvl)  OR a ray-error tuple
        if len(r) == 3 and hasattr(r[0], "__len__") and len(r[0]) >= 2:
            try:
                pt = r[0][0]   # (pos, dir, dst, normal)
                xs.append(float(pt[0])); ys.append(float(pt[1]))
            except Exception:
                pass
    return np.array(xs), np.array(ys)


def rms_radius(xs, ys):
    if len(xs) == 0:
        return float('nan'), 0
    cx, cy = xs.mean(), ys.mean()
    r2 = (xs-cx)**2 + (ys-cy)**2
    return float(np.sqrt(r2.mean())), len(xs)


df = fake_df()

print("=" * 78)
print("TEST 1  Does the requested fnumber reach the optical model?")
print("=" * 78)
print(f"{'kind':<14} {'requested':>10} {'self.fnumber':>14} {'opm pupil':>11}")
for kind, fnums in [("nikkor_58mm", [0.98, 2.0, 4.0, 8.0]),
                    ("microscope", [2.0, 4.0, 8.0, 16.0])]:
    for fn in fnums:
        lens = Lens(data=df, kind=kind, fnumber=fn, zfine=0)
        print(f"{kind:<14} {fn:>10.2f} {lens.fnumber:>14.2f} "
              f"{lens.opm.optical_spec.pupil.value:>11.3f}")


print()
print("=" * 78)
print("TEST 2  do_apertures status and surface max apertures")
print("=" * 78)
for kind in ["nikkor_58mm", "microscope"]:
    lens = Lens(data=df, kind=kind, fnumber=2.0, zfine=0)
    sm = lens.opm.seq_model
    # collect max apertures over interfaces
    aps = []
    for ifc in sm.ifcs:
        try:
            aps.append(ifc.max_aperture)
        except Exception:
            aps.append(None)
    valid = [a for a in aps if a is not None]
    print(f"{kind:<14}  do_apertures={sm.do_apertures}  "
          f"n_ifcs={len(sm.ifcs)}  "
          f"max_aperture range = "
          f"[{min(valid):.3f}, {max(valid):.3f}] mm"
          if valid else
          f"{kind:<14}  do_apertures={sm.do_apertures}  no max_aperture data")


print()
print("=" * 78)
print("TEST 3  Spot size at image plane vs fnumber (cone of rays from a point)")
print("        check_apertures=FALSE  (the way optics.py actually traces)")
print("=" * 78)
print("Hypothesis: fnumber should narrow the accepted cone -> smaller spot.")
print("Reality with do_apertures=False and check_apertures=False:")
print()

# 5 degree half-cone is plenty for any of these
half_ang = math.radians(5.0)
n_test = 800

for kind in ["nikkor_58mm", "microscope"]:
    print(f"-- {kind} --")
    print(f"{'fnumber':>8} {'rays_in':>8} {'rays_out':>9} "
          f"{'rms_r [mm]':>11} {'mean_x':>9} {'mean_y':>9}")
    for fn in ([0.98, 2.0, 4.0, 8.0, 16.0]
               if kind == "nikkor_58mm" else
               [2.0, 4.0, 8.0, 16.0]):
        lens = Lens(data=df, kind=kind, fnumber=fn, zfine=0)
        rays = cone_rays(0.0, 0.0, half_ang, n_rays=n_test)
        xs, ys = trace_and_collect_final_xy(lens.opm, rays, check_apertures=False)
        rms, n = rms_radius(xs, ys)
        print(f"{fn:>8.2f} {n_test:>8d} {n:>9d} {rms:>11.4f} "
              f"{xs.mean() if n else float('nan'):>9.4f} "
              f"{ys.mean() if n else float('nan'):>9.4f}")
    print()


print("=" * 78)
print("TEST 4  Same spot test, but with check_apertures=TRUE")
print("        AND surface apertures populated from the entrance pupil.")
print("        This is what fnumber 'would' do if the codebase enforced it.")
print("=" * 78)

for kind in ["nikkor_58mm", "microscope"]:
    print(f"-- {kind} --")
    print(f"{'fnumber':>8} {'rays_in':>8} {'rays_out':>9} {'rms_r [mm]':>11}")
    for fn in ([0.98, 2.0, 4.0, 8.0, 16.0]
               if kind == "nikkor_58mm" else
               [2.0, 4.0, 8.0, 16.0]):
        lens = Lens(data=df, kind=kind, fnumber=fn, zfine=0)
        sm = lens.opm.seq_model
        # Turn aperture enforcement back on and propagate the pupil through.
        sm.do_apertures = True
        try:
            sm.set_clear_apertures()
        except Exception as e:
            # If parax data unavailable, fall back to paraxial
            try:
                sm.set_clear_apertures_paraxial()
            except Exception:
                pass
        rays = cone_rays(0.0, 0.0, half_ang, n_rays=n_test)
        xs, ys = trace_and_collect_final_xy(lens.opm, rays, check_apertures=True)
        rms, n = rms_radius(xs, ys)
        print(f"{fn:>8.2f} {n_test:>8d} {n:>9d} {rms:>11.4f}")
    print()


print("=" * 78)
print("TEST 5  Depth-of-field check: defocused point spot size vs fnumber")
print("        (object point at z=0 but lens focused for a different distance).")
print("        With check_apertures=TRUE; expect smaller f# -> larger blur.")
print("=" * 78)

# We push the object 50 mm closer / farther and look at how the spot grows.
# For the microscope this is a big defocus; for nikkor it's small.
for kind, z_offsets in [("nikkor_58mm", [-50.0, 0.0, 50.0]),
                         ("microscope", [-5.0, 0.0, 5.0])]:
    print(f"-- {kind} --")
    print(f"{'fnumber':>8} " + " ".join(
        [f"rms@dz={dz:+5.1f}".rjust(14) for dz in z_offsets]))
    for fn in [2.0, 4.0, 8.0, 16.0]:
        lens = Lens(data=df, kind=kind, fnumber=fn, zfine=0)
        sm = lens.opm.seq_model
        sm.do_apertures = True
        try:
            sm.set_clear_apertures()
        except Exception:
            try: sm.set_clear_apertures_paraxial()
            except Exception: pass

        row = [f"{fn:>8.2f}"]
        for dz in z_offsets:
            # shift object position along z by dz (defocus)
            rays = []
            half = math.radians(5.0)
            for ray in cone_rays(0.0, 0.0, half, n_rays=n_test):
                p, d, w = ray
                p2 = p.copy(); p2[2] = dz
                rays.append((p2, d, w))
            xs, ys = trace_and_collect_final_xy(lens.opm, rays, check_apertures=True)
            rms, n = rms_radius(xs, ys)
            row.append(f"{rms:>10.4f}({n:>3d})")
        print(" ".join(row))
    print()


print("=" * 78)
print("SUMMARY")
print("=" * 78)
print("""
KEY FINDINGS

1. optics.py:158 calls trace_list_of_rays WITHOUT check_apertures=True.
2. nikkor / microscope / zmx_file constructors all set sm.do_apertures = False.
3. trace_raw() only does point_inside() / aperture clipping when
   check_apertures=True (rayoptics/raytr/raytrace.py:189).

=> In the production code path, fnumber has zero effect on which rays pass
   through the lens. The PupilSpec only affects (a) paraxial first-order
   parameters (FOD) and (b) the entrance-pupil sampling used by analysis
   routines like trace_grid / trace_boundary_rays / eval_pupil_coords. None
   of those are on the production path; trace_list_of_rays(rays_from_Geant4)
   is.

=> Therefore: smaller fnumber will NOT shrink the entrance pupil, will NOT
   produce a shallower depth of field, and will NOT change the spot size of
   defocused points. The "aperture" is effectively wide open --- the only
   thing clipping rays is surface geometry (interfaces sized large enough
   that almost any Geant4-produced ray makes it through).

WHAT FNUMBER WOULD HAVE TO DO

To actually mimic a real aperture, the model would need:
  (i)  sm.do_apertures = True (so set_clear_apertures() propagates the
       entrance pupil down to physical surface clear apertures); AND
  (ii) the trace_list_of_rays call at optics.py:158 to pass
       check_apertures=True so the per-surface point_inside() test is run.

Both are currently OFF.
""")

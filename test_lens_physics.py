#!/usr/bin/env python3
"""
Synthetic point-source tests for the lumacam Lens model.

Goal: prove the optics work as expected, with no G4 simulation, no empindex
clustering, no Bragg-track depth spread. We build photon cones aimed at the
lens entrance pupil from known source coordinates, trace them through the
lens at chosen (zfine, zscan, fnumber), and check four physical predictions:

  Test 1  Focal-plane shift with zscan.
          A point source at (0,0,z0) should image to a SINGLE point when the
          lens is correctly conjugated. RMS spot vs zscan should have a sharp
          minimum at zscan ≈ z0 (with zfine fixed at its calibrated value).

  Test 2  Magnification linearity.
          With the lens in focus, sources at (x_obj, 0, z0) should map to
          image positions x_img = m * x_obj where m = 1/reduction_ratio
          (with the lens's sign convention). The fit slope must agree with
          rayoptics' get_first_order_parameters() to <1%.

  Test 3  Depth-focus mapping.
          For each zscan, identify which physical depth z is in focus by
          finding the z whose image RMS is smallest. The relation should be
          monotonic and approximately z_focus ≈ zscan (Python z=0 anchored).

  Test 4  fnumber and depth-of-field.
          At f/0.95 the DOF should be narrow (a few mm); at f/8 it should
          cover the whole scintillator. Verify by sweeping zscan around the
          known focal depth and looking at the FWHM of the RMS curve.

If any of these fail, the lens model is misconfigured and tuning empindex
output is hopeless. If they all pass, the noise you see in the empindex
plots is downstream (reconstruction floor / source extent), not the optics.

Usage:
  python test_lens_physics.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lumacam.optics import Lens, VerbosityLevel
from rayoptics.environment import WvlSpec
from rayoptics.raytr import analyses


# ---------------------------------------------------------------------------
# constants matching the G4LumaCam geometry
# ---------------------------------------------------------------------------

LENS_KIND = "nikkor_58mm"
DIST_FROM_OBJ_MM = 461.535          # lens first surface position (Python frame, mm)
EPD_RADIUS_MM = 30.5                # ~EFL/2/fnumber at f/0.95 — full lens aperture
WAVELENGTH_NM = 500.0               # narrow-band test wavelength


# ---------------------------------------------------------------------------
# synthetic photon generation
# ---------------------------------------------------------------------------

def emit_cone_at_pupil(x0: float, y0: float, z0: float,
                        n_rays: int, lens_z_mm: float = DIST_FROM_OBJ_MM,
                        pupil_radius_mm: float = EPD_RADIUS_MM,
                        rng: np.random.Generator | None = None
                       ) -> pd.DataFrame:
    """Emit photons from a point source at (x0, y0, z0), each ray aimed at a
    uniform random point on a disk of radius `pupil_radius_mm` centered on
    the lens optical axis at z = `lens_z_mm`.

    This guarantees every ray reaches the lens (no acceptance loss) and that
    the angular cone fills the full aperture — exactly what you'd want for a
    "perfect emitter" feeding the lens.

    Returns a DataFrame in the same column layout the lumacam tracer expects:
        x, y, z, dx, dy, dz, wavelength, id, neutron_id, pulse_id, toa, nz, pz
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Uniform sampling on the lens-aperture disk
    r = pupil_radius_mm * np.sqrt(rng.uniform(0, 1, n_rays))
    phi = rng.uniform(0, 2 * np.pi, n_rays)
    x_aim = r * np.cos(phi)
    y_aim = r * np.sin(phi)
    z_aim = lens_z_mm

    # Direction vector from source to aim point on the aperture disk
    dxv = x_aim - x0
    dyv = y_aim - y0
    dzv = z_aim - z0
    norm = np.sqrt(dxv * dxv + dyv * dyv + dzv * dzv)
    dx = dxv / norm
    dy = dyv / norm
    dz = dzv / norm

    df = pd.DataFrame({
        "x":  np.full(n_rays, x0),
        "y":  np.full(n_rays, y0),
        "z":  np.full(n_rays, z0),
        "dx": dx, "dy": dy, "dz": dz,
        "wavelength": np.full(n_rays, WAVELENGTH_NM),
        "id":         np.arange(n_rays),
        "neutron_id": np.zeros(n_rays, dtype=int),
        "pulse_id":   np.zeros(n_rays, dtype=int),
        "toa":        np.zeros(n_rays),
        "nz":         np.full(n_rays, z0),
        "pz":         np.full(n_rays, z0),
        "px":         np.full(n_rays, x0),
        "py":         np.full(n_rays, y0),
    })
    return df


# ---------------------------------------------------------------------------
# lens helpers
# ---------------------------------------------------------------------------

def make_lens(zfine: float, fnumber: float) -> Lens:
    empty = pd.DataFrame({c: [] for c in
        ["x", "y", "z", "dx", "dy", "dz", "wavelength",
         "nz", "pz", "id", "neutron_id", "pulse_id", "toa"]})
    return Lens(data=empty, kind=LENS_KIND, fnumber=fnumber, zfine=zfine,
                verbosity=VerbosityLevel.QUIET)


def install_wavelength(lens: Lens, wavelength_nm: float) -> None:
    """Snap the spec to a single wavelength and propagate to seq_model."""
    wvl_values = np.array([[float(round(wavelength_nm, 1)), 1.0]])
    for opm in (lens.opm0, lens.opm):
        if opm is None:
            continue
        opm.optical_spec.spectral_region = WvlSpec(wvl_values, ref_wl=0)
        opm.update_model()


def trace(lens: Lens, photons: pd.DataFrame) -> np.ndarray:
    """Trace rays and return (n, 2) array of (x_image, y_image) at the sensor.
    Missing rows are NaN."""
    wvl = float(round(WAVELENGTH_NM, 1))
    rays = [(np.array([r.x, r.y, r.z], dtype=float),
             np.array([r.dx, r.dy, r.dz], dtype=float),
             np.array([wvl], dtype=float))
            for r in photons.itertuples()]
    out = np.full((len(photons), 2), np.nan)
    res = analyses.trace_list_of_rays(
        lens.opm, rays,
        output_filter="last", rayerr_filter="summary",
        check_apertures=lens.enforce_aperture,
    )
    for i, r in enumerate(res):
        if r is None:
            continue
        try:
            xi, yi, _ = r[0][0]
            if np.isfinite(xi) and np.isfinite(yi):
                out[i] = (float(xi), float(yi))
        except Exception:
            pass
    return out


def image_rms(image_xy: np.ndarray) -> tuple[float, int]:
    valid = ~np.isnan(image_xy[:, 0])
    n = int(valid.sum())
    if n < 3:
        return float("inf"), n
    xy = image_xy[valid]
    cx, cy = xy.mean(axis=0)
    return float(np.sqrt(np.mean((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2))), n


# ---------------------------------------------------------------------------
# Test 1 — focal-plane shift with zscan (axial source at z=10)
# ---------------------------------------------------------------------------

def test_focal_plane_shift(zfine: float, fnumber: float,
                            z_target: float = 10.0, n_rays: int = 400):
    print(f"\n=== TEST 1: focal-plane shift "
          f"(source on axis, z={z_target} mm, zfine={zfine}, f/{fnumber}) ===")
    lens = make_lens(zfine=zfine, fnumber=fnumber)
    install_wavelength(lens, WAVELENGTH_NM)
    photons = emit_cone_at_pupil(0, 0, z_target, n_rays)

    zscans = np.arange(-5, 25.01, 1.0)
    rms = np.empty_like(zscans, dtype=float)
    for i, zs in enumerate(zscans):
        lens.refocus(zscan=zs, zfine=zfine)
        install_wavelength(lens, WAVELENGTH_NM)  # spec is wiped by refocus
        img = trace(lens, photons)
        rms[i], _ = image_rms(img)

    i_min = int(np.argmin(rms))
    print(f"{'zscan':>6}  {'RMS (um)':>10}")
    for zs, r in zip(zscans, rms):
        flag = "  <-- min" if zs == zscans[i_min] else ""
        if np.isfinite(r):
            print(f"{zs:>6.1f}  {r*1000:>10.2f}{flag}")
        else:
            print(f"{zs:>6.1f}  {'inf':>10}{flag}")
    print(f"\nFocal-plane minimum at zscan = {zscans[i_min]:.1f} mm "
          f"(expected ≈ {z_target} mm if lens conjugate convention holds)")
    print(f"Min spot RMS = {rms[i_min]*1000:.2f} um, "
          f"max spot RMS in sweep = {np.nanmax(rms)*1000:.2f} um  "
          f"(ratio = {np.nanmax(rms)/rms[i_min]:.1f}x)")
    return zscans, rms


# ---------------------------------------------------------------------------
# Test 2 — magnification linearity
# ---------------------------------------------------------------------------

def test_magnification(zfine: float, fnumber: float,
                        z_target: float, zscan: float,
                        n_rays_per_source: int = 200,
                        x_grid_mm = (-50, -30, -10, 0, 10, 30, 50)):
    print(f"\n=== TEST 2: magnification linearity "
          f"(z={z_target}, zscan={zscan}, zfine={zfine}, f/{fnumber}) ===")
    lens = make_lens(zfine=zfine, fnumber=fnumber)
    lens.refocus(zscan=zscan, zfine=zfine)
    install_wavelength(lens, WAVELENGTH_NM)
    R = lens.reduction_ratio
    print(f"Lens reports reduction_ratio = {R:.4f}  =>  expected slope m = 1/R = {1/R:+.5f}")

    rng = np.random.default_rng(7)
    rows = []
    for x in x_grid_mm:
        photons = emit_cone_at_pupil(x, 0, z_target, n_rays_per_source, rng=rng)
        img = trace(lens, photons)
        valid = ~np.isnan(img[:, 0])
        if valid.sum() < 5:
            print(f"  x={x:+.1f}  trace failed")
            continue
        cx = float(img[valid, 0].mean())
        cy = float(img[valid, 1].mean())
        spot = float(np.sqrt(np.mean(
            (img[valid, 0] - cx) ** 2 + (img[valid, 1] - cy) ** 2)))
        rows.append((x, cx, cy, spot, int(valid.sum())))
    df = pd.DataFrame(rows, columns=["x_obj_mm", "x_img_mm", "y_img_mm",
                                     "spot_um", "n"])
    df["spot_um"] *= 1000.0
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if len(df) >= 2:
        slope, intercept = np.polyfit(df["x_obj_mm"], df["x_img_mm"], 1)
        m_expected = 1.0 / R
        print(f"\nFit: x_img = {slope:+.5f} * x_obj + {intercept:+.4f}")
        print(f"Predicted m = 1/R = {m_expected:+.5f}; "
              f"discrepancy {(slope - m_expected) / m_expected * 100:+.2f}%")
        return df, slope, m_expected
    return df, None, None


# ---------------------------------------------------------------------------
# Test 3 — depth-focus mapping (which z is sharp at each zscan?)
# ---------------------------------------------------------------------------

def test_depth_focus_map(zfine: float, fnumber: float,
                          z_targets=(0.5, 5, 10, 15, 19.5),
                          zscans=(-5, 0, 5, 10, 15, 20, 25),
                          n_rays: int = 300):
    print(f"\n=== TEST 3: depth-focus map (zfine={zfine}, f/{fnumber}) ===")
    lens = make_lens(zfine=zfine, fnumber=fnumber)

    rng = np.random.default_rng(11)
    sources = {z: emit_cone_at_pupil(0, 0, z, n_rays, rng=rng) for z in z_targets}

    print(f"\nRMS spot size (um), rows=zscan, cols=source z (where lens is focused = smallest in row):")
    header = "    " + "  ".join(f"z={z:>4.1f}" for z in z_targets)
    print(f"{'zscan':>6}  " + header)
    focal_z_at_zs = {}
    for zs in zscans:
        lens.refocus(zscan=zs, zfine=zfine)
        install_wavelength(lens, WAVELENGTH_NM)
        row_rms = {}
        for z in z_targets:
            img = trace(lens, sources[z])
            row_rms[z], _ = image_rms(img)
        line_vals = "  ".join(f"{row_rms[z]*1000:>7.1f}" for z in z_targets)
        z_best = min(row_rms, key=lambda k: row_rms[k])
        focal_z_at_zs[zs] = (z_best, row_rms[z_best])
        print(f"{zs:>6.1f}    {line_vals}    -> min @ z={z_best:.1f} "
              f"({row_rms[z_best]*1000:.1f} um)")
    print()
    print("Focal depth vs zscan:")
    print(f"  {'zscan':>6}  {'focal_z':>8}  {'RMS_um':>8}")
    for zs, (z_best, r) in focal_z_at_zs.items():
        print(f"  {zs:>6.1f}  {z_best:>8.1f}  {r*1000:>8.1f}")
    return focal_z_at_zs


# ---------------------------------------------------------------------------
# Test 4 — fnumber and depth of field
# ---------------------------------------------------------------------------

def test_fnumber_dof(zfine: float, z_target: float = 10.0, n_rays: int = 400,
                      fnumbers=(0.95, 2.0, 4.0, 8.0)):
    print(f"\n=== TEST 4: fnumber / depth-of-field (z={z_target}, zfine={zfine}) ===")
    for fn in fnumbers:
        lens = make_lens(zfine=zfine, fnumber=fn)
        install_wavelength(lens, WAVELENGTH_NM)
        # narrow sweep around the focal zscan
        zscans = np.arange(z_target - 8, z_target + 8.01, 1.0)
        rms = []
        for zs in zscans:
            lens.refocus(zscan=zs, zfine=zfine)
            install_wavelength(lens, WAVELENGTH_NM)
            photons = emit_cone_at_pupil(0, 0, z_target, n_rays)
            img = trace(lens, photons)
            r, _ = image_rms(img)
            rms.append(r)
        rms = np.array(rms)
        r_min = float(np.nanmin(rms))
        # FWHM in zscan where RMS doubles from minimum
        thresh = 2.0 * r_min
        above = rms > thresh
        if above.any() and (~above).any():
            in_focus = ~above
            ones = np.where(in_focus)[0]
            fwhm = zscans[ones.max()] - zscans[ones.min()]
        else:
            fwhm = float("nan")
        print(f"  f/{fn:>4}:  min RMS = {r_min*1000:>6.2f} um at zscan = "
              f"{zscans[int(np.argmin(rms))]:.1f}, "
              f"FWHM where RMS<2x_min = {fwhm:.1f} mm")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="Fewer rays per test")
    ap.add_argument("--zfine", type=float, default=12.275)
    ap.add_argument("--fnumber", type=float, default=0.95)
    ap.add_argument("--z-target", type=float, default=10.0)
    args = ap.parse_args()

    n_rays = 120 if args.quick else 400

    # Test 1
    zscans, rms = test_focal_plane_shift(args.zfine, args.fnumber,
                                          z_target=args.z_target, n_rays=n_rays)

    # Test 2 — magnification linearity at whatever zscan minimized Test 1
    zs_focus = float(zscans[int(np.argmin(rms))])
    test_magnification(args.zfine, args.fnumber, args.z_target, zs_focus,
                        n_rays_per_source=n_rays // 2)

    # Test 3 — depth-focus mapping
    test_depth_focus_map(args.zfine, args.fnumber, n_rays=n_rays)

    # Test 4 — fnumber DOF
    test_fnumber_dof(args.zfine, z_target=args.z_target, n_rays=n_rays)

    print("\nDone.  Interpretation:")
    print("  * Test 1 should show a sharp minimum somewhere in zscan ∈ [-5,25].")
    print("  * Test 2 fit slope should match 1/reduction_ratio to <1%.")
    print("  * Test 3 should show a monotonic focal_z vs zscan relation.")
    print("  * Test 4 should show FWHM growing with fnumber (wider DOF at higher f/#).")


if __name__ == "__main__":
    main()

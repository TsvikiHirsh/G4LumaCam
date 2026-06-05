#!/usr/bin/env python3
"""
Deep physics audit of the lumacam Lens model.

What we test, in order of severity:

  A. focal-depth map. For each zfine on a dense grid, find the zscan that
     brings z=0, 5, 10, 15, 20 mm into focus. This gives a direct lookup
     table you can use to pick (zfine, zscan) for any depth.

  B. paraxial magnification. Use tiny off-axis source displacements (1 mm
     and smaller) so aberrations don't pollute the fit. Compare actual
     image slope vs rayoptics' first-order reduction_ratio. If they
     disagree, the pixel-mm calibration in the analysis is wrong.

  C. pixel coordinate landing. With a known on-axis source, compute what
     pixel the trace says it lands on. Compare against (127.5, 127.5)
     (sensor center). This catches sign/origin convention bugs.

  D. ray transmission and vignetting. For each source position, count how
     many rays survive the lens trace. Off-axis sources may lose rays to
     internal apertures even when aimed correctly. If transmission is
     wildly different between sources, the centroid is biased.

  E. depth of field. For each fnumber, measure the FWHM in zscan over
     which the spot stays below 2x its minimum. Compare to the formula
     DOF ≈ 2*N*c*(M+1)/M².

  F. realistic G4 photons. Load actual sim_data, restrict to a thin z0
     slice, trace through the lens. See whether the per-source image
     positions agree with the paraxial prediction.

Usage:
  python test_lens_deep.py [--archive PATH]
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


LENS_KIND = "nikkor_58mm"
DIST_FROM_OBJ_MM = 461.535
EPD_RADIUS_MM = 30.5
WAVELENGTH_NM = 500.0
PIXEL_PITCH_MM = 0.055
SENSOR_HALF_PIX = 127.5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_lens(zfine: float, fnumber: float) -> Lens:
    empty = pd.DataFrame({c: [] for c in
        ["x", "y", "z", "dx", "dy", "dz", "wavelength",
         "nz", "pz", "id", "neutron_id", "pulse_id", "toa"]})
    return Lens(data=empty, kind=LENS_KIND, fnumber=fnumber, zfine=zfine,
                verbosity=VerbosityLevel.QUIET)


def install_wavelength(lens: Lens) -> None:
    wvl_values = np.array([[float(round(WAVELENGTH_NM, 1)), 1.0]])
    for opm in (lens.opm0, lens.opm):
        if opm is None:
            continue
        opm.optical_spec.spectral_region = WvlSpec(wvl_values, ref_wl=0)
        opm.update_model()


def emit_cone(x0, y0, z0, n_rays, pupil_r=EPD_RADIUS_MM,
              rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    r = pupil_r * np.sqrt(rng.uniform(0, 1, n_rays))
    phi = rng.uniform(0, 2 * np.pi, n_rays)
    xa = r * np.cos(phi)
    ya = r * np.sin(phi)
    dxv = xa - x0
    dyv = ya - y0
    dzv = DIST_FROM_OBJ_MM - z0
    n = np.sqrt(dxv**2 + dyv**2 + dzv**2)
    return (np.full(n_rays, x0), np.full(n_rays, y0), np.full(n_rays, z0),
            dxv / n, dyv / n, dzv / n)


def trace_xy(lens, xs, ys, zs, dxs, dys, dzs):
    wvl = float(round(WAVELENGTH_NM, 1))
    rays = [(np.array([x, y, z]), np.array([dx, dy, dz]),
             np.array([wvl]))
            for x, y, z, dx, dy, dz in zip(xs, ys, zs, dxs, dys, dzs)]
    out = np.full((len(rays), 2), np.nan)
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
                out[i] = (xi, yi)
        except Exception:
            pass
    return out


def spot_stats(img):
    valid = ~np.isnan(img[:, 0])
    n = int(valid.sum())
    if n < 3:
        return float("nan"), float("nan"), float("nan"), 0
    xy = img[valid]
    cx, cy = xy.mean(axis=0)
    rms = float(np.sqrt(np.mean((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)))
    return float(cx), float(cy), rms, n


# ---------------------------------------------------------------------------
# A. focal map: (zfine, target z) -> zscan
# ---------------------------------------------------------------------------

def test_focal_map(zfines, z_targets, fnumber=0.95, n_rays=200,
                    zscan_lo=-40, zscan_hi=50, zscan_step=2.0):
    print(f"\n=== A. focal-depth map (fnumber={fnumber}) ===")
    print(f"For each zfine and each target depth z, the zscan that minimises")
    print(f"the image RMS — i.e. the lens focuses on z.\n")
    print(f"{'zfine':>6}  " + "  ".join(f"z={z:>4.1f}" for z in z_targets))
    rows = []
    for zfine in zfines:
        lens = make_lens(zfine=zfine, fnumber=fnumber)
        line = [f"{zfine:>6.2f}"]
        for z in z_targets:
            xs, ys, zs0, dxs, dys, dzs = emit_cone(0, 0, z, n_rays)
            zscans = np.arange(zscan_lo, zscan_hi + 1e-6, zscan_step)
            rms = []
            for zs_v in zscans:
                lens.refocus(zscan=float(zs_v), zfine=zfine)
                install_wavelength(lens)
                img = trace_xy(lens, xs, ys, zs0, dxs, dys, dzs)
                _, _, r, _ = spot_stats(img)
                rms.append(r if np.isfinite(r) else 1e9)
            rms = np.array(rms)
            i = int(np.argmin(rms))
            best_zs = float(zscans[i])
            edge = "*" if i == 0 or i == len(rms) - 1 else " "
            line.append(f"{best_zs:>5.1f}{edge}")
            rows.append({"zfine": zfine, "target_z": z, "best_zscan": best_zs,
                          "edge_hit": edge == "*"})
        print("   ".join(line))
    print(f"\n* = best zscan hit the search boundary [{zscan_lo}, {zscan_hi}] — "
          f"true optimum may be outside")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B. paraxial magnification using TINY off-axis offsets
# ---------------------------------------------------------------------------

def test_paraxial_magnification(zfine, zscan, fnumber=0.95, z_target=10.0,
                                  n_rays=400):
    print(f"\n=== B. paraxial magnification "
          f"(zfine={zfine}, zscan={zscan}, fnumber={fnumber}, z={z_target}) ===")
    lens = make_lens(zfine=zfine, fnumber=fnumber)
    lens.refocus(zscan=zscan, zfine=zfine)
    install_wavelength(lens)
    R_paraxial = lens.reduction_ratio
    print(f"rayoptics reduction_ratio  = {R_paraxial:.4f}  (paraxial first-order)")
    print(f"=> predicted slope m = 1/R = {1/R_paraxial:+.5f}\n")

    # Tiny offsets so aberrations don't dominate
    rng = np.random.default_rng(3)
    rows = []
    for x_obj in [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]:
        xs, ys, zs0, dxs, dys, dzs = emit_cone(x_obj, 0, z_target, n_rays,
                                                rng=rng)
        img = trace_xy(lens, xs, ys, zs0, dxs, dys, dzs)
        cx, cy, rms, n = spot_stats(img)
        rows.append({"x_obj": x_obj, "x_img": cx, "y_img": cy,
                      "spot_um": rms * 1000, "n_traced": n})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # Fit slope from finite-but-small offsets
    finite = df.dropna(subset=["x_img"])
    if len(finite) >= 3:
        slope, intercept = np.polyfit(finite["x_obj"], finite["x_img"], 1)
        print(f"\nLinear fit: x_img = {slope:+.5f} * x_obj + {intercept:+.4f}")
        print(f"Predicted from rayoptics: {1/R_paraxial:+.5f}")
        ratio = slope / (1/R_paraxial)
        print(f"Ratio actual/predicted = {ratio:+.4f}  "
              f"(should be +1.000 if rayoptics is right)")
        return df, slope, 1.0 / R_paraxial
    return df, None, None


# ---------------------------------------------------------------------------
# C. pixel coordinate of an on-axis source
# ---------------------------------------------------------------------------

def test_pixel_landing(zfine, zscan, fnumber=0.95, z_target=10.0,
                        n_rays=200):
    print(f"\n=== C. pixel landing (on-axis source at z={z_target}, "
          f"zfine={zfine}, zscan={zscan}) ===")
    lens = make_lens(zfine=zfine, fnumber=fnumber)
    lens.refocus(zscan=zscan, zfine=zfine)
    install_wavelength(lens)
    R = lens.reduction_ratio

    xs, ys, zs0, dxs, dys, dzs = emit_cone(0, 0, z_target, n_rays)
    img = trace_xy(lens, xs, ys, zs0, dxs, dys, dzs)
    cx, cy, rms, n = spot_stats(img)
    pix_x = cx / PIXEL_PITCH_MM + SENSOR_HALF_PIX
    pix_y = cy / PIXEL_PITCH_MM + SENSOR_HALF_PIX
    print(f"Image position at sensor:  x = {cx*1000:+.2f} um, "
          f"y = {cy*1000:+.2f} um")
    print(f"Pixel coords (mid=127.5):  px = {pix_x:.3f}, py = {pix_y:.3f}")
    print(f"=> distance from sensor center: {np.hypot(pix_x - SENSOR_HALF_PIX, pix_y - SENSOR_HALF_PIX):.3f} pix")
    print(f"   (on-axis source should land at 127.5, 127.5)")
    return cx, cy, pix_x, pix_y


# ---------------------------------------------------------------------------
# D. vignetting / ray transmission at off-axis sources
# ---------------------------------------------------------------------------

def test_vignetting(zfine, zscan, fnumber=0.95, z_target=10.0,
                     n_rays=400):
    print(f"\n=== D. ray transmission across the field "
          f"(zfine={zfine}, zscan={zscan}, fnumber={fnumber}) ===")
    lens = make_lens(zfine=zfine, fnumber=fnumber)
    lens.refocus(zscan=zscan, zfine=zfine)
    install_wavelength(lens)
    rows = []
    for x_obj in [0, 5, 10, 15, 20, 30, 40, 50, 60]:
        xs, ys, zs0, dxs, dys, dzs = emit_cone(x_obj, 0, z_target, n_rays)
        img = trace_xy(lens, xs, ys, zs0, dxs, dys, dzs)
        valid = ~np.isnan(img[:, 0])
        rows.append({"x_obj_mm": x_obj, "n_input": n_rays,
                      "n_traced": int(valid.sum()),
                      "transmission_pct": 100.0 * valid.sum() / n_rays})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


# ---------------------------------------------------------------------------
# E. depth of field at varying fnumber
# ---------------------------------------------------------------------------

def test_dof(zfine, z_target=10.0, fnumbers=(0.95, 1.4, 2.0, 4.0, 8.0),
              n_rays=300, half_window=20.0):
    print(f"\n=== E. depth of field vs fnumber (z={z_target}, zfine={zfine}) ===")
    rng = np.random.default_rng(2)
    for fn in fnumbers:
        lens = make_lens(zfine=zfine, fnumber=fn)
        # Find best zscan over a wide window
        zscans = np.arange(-half_window, half_window + 0.001, 1.0)
        rms = []
        for zs_v in zscans:
            lens.refocus(zscan=float(zs_v), zfine=zfine)
            install_wavelength(lens)
            xs, ys, zs0, dxs, dys, dzs = emit_cone(0, 0, z_target, n_rays, rng=rng)
            img = trace_xy(lens, xs, ys, zs0, dxs, dys, dzs)
            _, _, r, _ = spot_stats(img)
            rms.append(r if np.isfinite(r) else float("inf"))
        rms = np.array(rms)
        i = int(np.argmin(rms))
        r_min = rms[i]
        # FWHM where spot stays < 2x min
        above = rms > 2 * r_min
        if (~above).any():
            in_focus_idx = np.where(~above)[0]
            fwhm = zscans[in_focus_idx.max()] - zscans[in_focus_idx.min()]
        else:
            fwhm = float("nan")
        # Also predict from formula DOF ≈ 2 N c (M+1)/M^2
        lens.refocus(zscan=float(zscans[i]), zfine=zfine)
        install_wavelength(lens)
        R = lens.reduction_ratio
        M = abs(1.0 / R)
        c_mm = PIXEL_PITCH_MM
        dof_pred = 2 * fn * c_mm * (M + 1) / (M * M)
        print(f"  f/{fn:>4}: min_RMS = {r_min*1000:>6.2f} um at zscan={zscans[i]:+.1f}, "
              f"FWHM<2x = {fwhm:>5.1f} mm   "
              f"(formula DOF ≈ {dof_pred:.1f} mm with M={M:.4f})")


# ---------------------------------------------------------------------------
# F. real G4 photons: per-source image positions
# ---------------------------------------------------------------------------

def test_real_photons(archive, zfine, zscan, fnumber=0.95):
    print(f"\n=== F. real G4 photons in thin z0 slice "
          f"(archive={archive}, zfine={zfine}, zscan={zscan}, fnumber={fnumber}) ===")
    sim = Path(archive) / "SimPhotons"
    csvs = sorted(sim.glob("sim_data_*.csv"))
    if not csvs:
        print(f"  no sim_data in {sim}; skipping")
        return
    df = pd.concat([pd.read_csv(p) for p in csvs[:3]], ignore_index=True)
    if "z" not in df.columns:
        print("  CSV has no 'z' column; abort")
        return
    print(f"  loaded {len(df)} photons")
    lens = make_lens(zfine=zfine, fnumber=fnumber)
    lens.refocus(zscan=zscan, zfine=zfine)
    install_wavelength(lens)

    # take photons within 0.5 mm of z=10 (or whatever's available)
    for z_slice in [5.0, 10.0, 15.0]:
        sub = df[(df.z > z_slice - 0.5) & (df.z < z_slice + 0.5)].copy()
        if len(sub) < 30:
            print(f"  z={z_slice}: only {len(sub)} photons in slice — skip")
            continue
        rays = [(np.array([r.x, r.y, r.z]), np.array([r.dx, r.dy, r.dz]),
                 np.array([float(round(r.wavelength, 1))]))
                for r in sub.itertuples()]
        res = analyses.trace_list_of_rays(
            lens.opm, rays,
            output_filter="last", rayerr_filter="summary",
            check_apertures=lens.enforce_aperture,
        )
        x_img = []; y_img = []; x_src = []; y_src = []
        for r, row in zip(res, sub.itertuples()):
            if r is None: continue
            try:
                xi, yi, _ = r[0][0]
                if np.isfinite(xi) and np.isfinite(yi):
                    x_img.append(xi); y_img.append(yi)
                    x_src.append(row.x); y_src.append(row.y)
            except Exception: pass
        if len(x_img) < 20:
            print(f"  z={z_slice}: only {len(x_img)} traced — skip"); continue
        x_img = np.array(x_img); y_img = np.array(y_img)
        x_src = np.array(x_src); y_src = np.array(y_src)
        # Fit slope per axis
        slope_x, intercept_x = np.polyfit(x_src, x_img, 1)
        slope_y, intercept_y = np.polyfit(y_src, y_img, 1)
        resid_x = x_img - (slope_x * x_src + intercept_x)
        resid_y = y_img - (slope_y * y_src + intercept_y)
        R = lens.reduction_ratio
        print(f"  z={z_slice}: n={len(x_img)}, "
              f"slope_x={slope_x:+.5f}, slope_y={slope_y:+.5f}, "
              f"predicted 1/R={1/R:+.5f}, "
              f"resid std_x={resid_x.std()*1000:.1f}um, std_y={resid_y.std()*1000:.1f}um")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="notebooks/archive/openbeam_ptb_1e5")
    args = ap.parse_args()

    # A. focal map across a fine zfine grid
    focal_map = test_focal_map(
        zfines=[11.5, 12.0, 12.275, 12.5, 12.75, 13.0, 13.5],
        z_targets=[0.0, 5.0, 10.0, 15.0, 20.0],
        fnumber=0.95,
        n_rays=200,
        zscan_lo=-30, zscan_hi=50, zscan_step=2.0,
    )

    # Use the map to derive (zfine=12.275, target=10) -> zscan
    row = focal_map[(focal_map.zfine == 12.275) & (focal_map.target_z == 10.0)]
    zs_for_center_12275 = float(row.best_zscan.iloc[0])
    row2 = focal_map[(focal_map.zfine == 12.5) & (focal_map.target_z == 10.0)]
    zs_for_center_125 = float(row2.best_zscan.iloc[0])

    print(f"\n--- Headline values ---")
    print(f"  zfine=12.275 -> z=10 focus at zscan={zs_for_center_12275:.1f}")
    print(f"  zfine=12.5   -> z=10 focus at zscan={zs_for_center_125:.1f}")

    # B. paraxial magnification at the calibrated config
    test_paraxial_magnification(zfine=12.275, zscan=zs_for_center_12275,
                                fnumber=0.95)

    # C. pixel landing for on-axis source
    test_pixel_landing(zfine=12.275, zscan=zs_for_center_12275, fnumber=0.95)

    # D. vignetting
    test_vignetting(zfine=12.275, zscan=zs_for_center_12275, fnumber=0.95)

    # E. DOF
    test_dof(zfine=12.275)

    # F. realistic photons (if archive exists)
    if Path(args.archive).exists():
        test_real_photons(args.archive, zfine=12.275,
                          zscan=zs_for_center_12275, fnumber=0.95)


if __name__ == "__main__":
    main()

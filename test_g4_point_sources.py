#!/usr/bin/env python3
"""
End-to-end physics audit using G4 point sources.

Generates a G4 simulation with a forward-emitting point source of optical
photons at a chosen position inside the scintillator, then traces the
resulting sim_data through the lumacam Lens. By varying source position
and lens parameters we can directly check four physical claims:

  T1: a point source at (0, 0, z) imaged in focus produces a sharp spot
      at sensor centre (pixel 127.5, 127.5).
  T2: image position scales linearly with source x: x_img = m * x_obj.
      The measured slope = the actual magnification.
  T3: the focal z (smallest spot) walks with zscan one-to-one.
  T4: at fnumber=0.95 the DOF is ~7 mm; at fnumber=8 it covers the
      whole scintillator.

Each test runs a G4 sim, reads sim_data_*.csv, traces through the lens
(same code path the empindex pipeline uses), and reports image-plane
statistics. No empindex, no clustering, no blob: just G4 + lens.
"""

from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import lumacam
from lumacam.optics import Lens, VerbosityLevel
from rayoptics.environment import WvlSpec
from rayoptics.raytr import analyses


ARCHIVE_BASE = Path("archive/point_source_test")
PIXEL_PITCH_MM = 0.055
SENSOR_HALF_PIX = 127.5


# ---------------------------------------------------------------------------
# G4 point source generation
# ---------------------------------------------------------------------------

def make_point_source_config(x_mm: float, y_mm: float, z_mm: float,
                              num_events: int = 5000,
                              max_theta_deg: float = 5.0) -> lumacam.Config:
    """Forward-emitting point source of 500 nm photons at (x, y, z) inside
    the scintillator. theta in [0, max_theta] from +z = a narrow forward cone
    aimed at the lens."""
    cfg = lumacam.Config.opticalphoton_point()
    cfg.position_x = x_mm
    cfg.position_y = y_mm
    cfg.position_z = z_mm
    cfg.position_unit = "mm"
    cfg.halfx = 0.0001
    cfg.halfy = 0.0001
    cfg.shape_unit = "um"
    cfg.direction_x = 0.0
    cfg.direction_y = 0.0
    cfg.direction_z = 1.0
    # GPS theta is measured from the direction OPPOSITE to /gps/direction.
    # So to emit FORWARD (along +z), we need theta near 180.
    cfg.angle_type = "iso"
    cfg.min_theta = 180.0 - max_theta_deg
    cfg.max_theta = 180.0
    cfg.angle_unit = "deg"
    cfg.num_events = int(num_events)
    cfg.progress_interval = max(num_events // 5, 1)
    cfg.csv_batch_size = int(num_events)
    # CRITICAL: opticalphoton_point inherits the dataclass default
    # scintillator_thickness=20, which the macro writes as "cm" → 200 mm slab.
    # The real geometry is 20 mm = 2 cm. Override here so the simulation
    # matches the standard scintillator the lens is calibrated for.
    cfg.scintillator_thickness = 2  # cm  (= 20 mm)
    return cfg


def run_g4_point(x_mm: float, y_mm: float, z_mm: float,
                  num_events: int = 5000, suffix: str = "default") -> Path:
    """Run a G4 simulation, return path to the archive directory."""
    archive = ARCHIVE_BASE / f"x{x_mm:+05.1f}_y{y_mm:+05.1f}_z{z_mm:+05.1f}_{suffix}"
    archive = Path(str(archive).replace("+", "p"))
    if archive.exists():
        shutil.rmtree(archive)
    sim = lumacam.Simulate(str(archive))
    cfg = make_point_source_config(x_mm, y_mm, z_mm, num_events=num_events)
    sim.run(cfg, verbosity=VerbosityLevel.QUIET)
    return archive


def load_sim_data(archive: Path) -> pd.DataFrame:
    csvs = sorted((archive / "SimPhotons").glob("sim_data_*.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)


# ---------------------------------------------------------------------------
# Lens trace
# ---------------------------------------------------------------------------

def trace_through_lens(df: pd.DataFrame, lens: Lens) -> tuple[np.ndarray, np.ndarray]:
    """Trace df rows through lens. Returns (image_xy, valid_source_xy)."""
    df = df.copy()
    df["wavelength"] = np.round(df["wavelength"], 1)
    # Install full set of wavelengths in the spec
    uniq = np.sort(df["wavelength"].unique())
    wvl_values = np.column_stack([uniq, np.ones(len(uniq))])
    for opm in (lens.opm0, lens.opm):
        opm.optical_spec.spectral_region = WvlSpec(wvl_values, ref_wl=0)
        opm.update_model()

    rays = [(np.array([r.x, r.y, r.z], dtype=float),
             np.array([r.dx, r.dy, r.dz], dtype=float),
             np.array([float(r.wavelength)], dtype=float))
            for r in df.itertuples()]
    res = analyses.trace_list_of_rays(
        lens.opm, rays,
        output_filter="last", rayerr_filter="summary",
        check_apertures=lens.enforce_aperture,
    )
    image_xy = []
    source_xy = []
    for r, row in zip(res, df.itertuples()):
        if r is None:
            continue
        try:
            xi, yi, _ = r[0][0]
            if np.isfinite(xi) and np.isfinite(yi):
                image_xy.append((xi, yi))
                source_xy.append((row.x, row.y))
        except Exception:
            pass
    return np.array(image_xy), np.array(source_xy)


def stat_block(image_xy: np.ndarray, label: str = ""):
    if len(image_xy) < 3:
        print(f"  {label}  no traced rays")
        return None
    cx, cy = image_xy.mean(axis=0)
    rx = image_xy[:, 0] - cx
    ry = image_xy[:, 1] - cy
    rms = float(np.sqrt(np.mean(rx*rx + ry*ry)))
    px = cx / PIXEL_PITCH_MM + SENSOR_HALF_PIX
    py = cy / PIXEL_PITCH_MM + SENSOR_HALF_PIX
    n = len(image_xy)
    print(f"  {label}  n={n:>4}  centroid (x,y)=({cx*1000:+.2f}, {cy*1000:+.2f}) um   "
          f"pixel ({px:.2f}, {py:.2f})   spot_RMS={rms*1000:.2f} um")
    return {"n": n, "cx_mm": cx, "cy_mm": cy, "pix_x": px, "pix_y": py,
            "rms_mm": rms}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make_lens(zfine: float, fnumber: float) -> Lens:
    empty = pd.DataFrame({c: [] for c in
        ["x", "y", "z", "dx", "dy", "dz", "wavelength",
         "nz", "pz", "id", "neutron_id", "pulse_id", "toa"]})
    return Lens(data=empty, kind="nikkor_58mm", fnumber=fnumber, zfine=zfine,
                verbosity=VerbosityLevel.QUIET)


def test_focal_position_at_calibrated_settings(num_events: int):
    print("\n=== T1: where does an on-axis source at z=10 land at "
          "(zfine=12.275, zscan=?, f/0.95)? ===")
    archive = run_g4_point(0.0, 0.0, 10.0, num_events=num_events, suffix="axis_z10")
    df = load_sim_data(archive)
    print(f"  G4 produced {len(df)} photon records at MonitorPhys")
    if df.empty:
        print("  EMPTY DATA — config may emit backward; check max_theta convention")
        return
    print(f"  source position from CSV: x median={df.x.median():.4f}, "
          f"y={df.y.median():.4f}, z={df.z.median():.4f}")
    print(f"  direction stats: dx mean={df.dx.mean():+.4f}, |dx| mean={df.dx.abs().mean():.4f}")

    # Sweep zscan to find sharpest spot
    print("  Sweeping zscan to find focal plane...")
    zscans = np.arange(0, 41, 2.0)
    rows = []
    lens = make_lens(zfine=12.275, fnumber=0.95)
    for zs in zscans:
        lens.refocus(zscan=float(zs), zfine=12.275)
        img, src = trace_through_lens(df, lens)
        if len(img) < 10:
            continue
        cx, cy = img.mean(axis=0)
        rms = float(np.sqrt(np.mean((img[:,0]-cx)**2 + (img[:,1]-cy)**2)))
        rows.append({"zscan": zs, "n": len(img),
                      "cx_um": cx*1000, "cy_um": cy*1000, "rms_um": rms*1000})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if not out.empty:
        i = int(out.rms_um.idxmin())
        print(f"\n  Sharpest at zscan = {out.zscan.iloc[i]:.1f}, "
              f"spot RMS = {out.rms_um.iloc[i]:.2f} um, "
              f"centroid pixel = ({out.cx_um.iloc[i]/1000/PIXEL_PITCH_MM + SENSOR_HALF_PIX:.2f}, "
              f"{out.cy_um.iloc[i]/1000/PIXEL_PITCH_MM + SENSOR_HALF_PIX:.2f})")


def test_magnification(num_events: int, zscan: float):
    print(f"\n=== T2: magnification at zfine=12.275, zscan={zscan}, f/0.95 ===")
    lens = make_lens(zfine=12.275, fnumber=0.95)
    lens.refocus(zscan=float(zscan), zfine=12.275)
    R = lens.reduction_ratio
    print(f"  rayoptics paraxial reduction_ratio = {R:.4f}, predicted m = 1/R = {1/R:+.5f}")

    rows = []
    for x_obj in [-30.0, -10.0, 0.0, 10.0, 30.0]:
        archive = run_g4_point(x_obj, 0.0, 10.0, num_events=num_events,
                                suffix=f"mag_x{int(x_obj)}")
        df = load_sim_data(archive)
        if df.empty: continue
        img, src = trace_through_lens(df, lens)
        if len(img) < 10: continue
        cx, cy = img.mean(axis=0)
        rows.append({"x_obj_mm": x_obj, "n_traced": len(img),
                      "x_img_mm": cx, "y_img_mm": cy})
    df_m = pd.DataFrame(rows)
    print(df_m.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    if len(df_m) >= 2:
        slope, intercept = np.polyfit(df_m.x_obj_mm, df_m.x_img_mm, 1)
        print(f"\n  Linear fit: x_img = {slope:+.5f} * x_obj + {intercept:+.4f}")
        print(f"  predicted by rayoptics: {1/R:+.5f}   |slope|/|predicted|={abs(slope)/abs(1/R):.3f}")
        print(f"  sign of slope: {'POSITIVE' if slope > 0 else 'NEGATIVE'}; "
              f"sign of predicted: {'POSITIVE' if 1/R > 0 else 'NEGATIVE'}")
        print(f"  empirical pixel→mm: PIXEL_MM = {PIXEL_PITCH_MM/abs(slope):.4f}")


def test_focal_plane_walk_with_zscan(num_events: int):
    print("\n=== T3: does focal plane walk with zscan? ===")
    lens = make_lens(zfine=12.275, fnumber=0.95)
    # Generate point sources at multiple depths once
    sources = {}
    for z_src in [5.0, 10.0, 15.0]:
        archive = run_g4_point(0.0, 0.0, z_src, num_events=num_events,
                                suffix=f"walk_z{int(z_src)}")
        sources[z_src] = load_sim_data(archive)

    print(f"  {'zscan':>6}  " + "  ".join(f"z={z:>4.1f} RMS"
                                            for z in sources))
    rows = []
    for zs in np.arange(0, 41, 2.0):
        lens.refocus(zscan=float(zs), zfine=12.275)
        line = [f"{zs:>6.1f}"]
        results = {}
        for z_src, df in sources.items():
            img, _ = trace_through_lens(df, lens)
            if len(img) < 5:
                line.append("   - "); continue
            cx, cy = img.mean(axis=0)
            rms = float(np.sqrt(np.mean((img[:,0]-cx)**2 + (img[:,1]-cy)**2)))
            results[z_src] = rms
            line.append(f"{rms*1000:>8.1f}")
        print("  ".join(line))
        rows.append({"zscan": zs, **{f"rms_z{int(z)}": results.get(z, np.nan)
                                       for z in sources}})

    print("\n  For each z_source, which zscan minimises the spot:")
    df_walk = pd.DataFrame(rows)
    for z_src in sources:
        col = f"rms_z{int(z_src)}"
        i = int(df_walk[col].idxmin())
        print(f"    z_src={z_src} mm  →  best zscan = {df_walk.zscan.iloc[i]:.1f}  "
              f"(spot {df_walk[col].iloc[i]*1000:.2f} um)")


def test_dof_vs_fnumber(num_events: int):
    print("\n=== T4: depth of field vs fnumber ===")
    archive = run_g4_point(0.0, 0.0, 10.0, num_events=num_events, suffix="dof")
    df = load_sim_data(archive)
    for fn in [0.95, 1.4, 2.0, 4.0, 8.0]:
        lens = make_lens(zfine=12.275, fnumber=fn)
        zscans = np.arange(-10, 41, 1.0)
        rms = []
        for zs in zscans:
            lens.refocus(zscan=float(zs), zfine=12.275)
            img, _ = trace_through_lens(df, lens)
            if len(img) < 5:
                rms.append(np.nan); continue
            cx, cy = img.mean(axis=0)
            r = float(np.sqrt(np.mean((img[:,0]-cx)**2 + (img[:,1]-cy)**2)))
            rms.append(r)
        rms = np.array(rms)
        finite = ~np.isnan(rms)
        if finite.sum() < 3:
            continue
        r_min = float(np.nanmin(rms))
        thr = 2 * r_min
        good = (rms < thr) & finite
        if good.any():
            ix = np.where(good)[0]
            fwhm = zscans[ix.max()] - zscans[ix.min()]
        else:
            fwhm = float("nan")
        i_min = int(np.nanargmin(rms))
        print(f"  f/{fn:>4}: min RMS = {r_min*1000:>6.2f} um at zscan={zscans[i_min]:+.1f}, "
              f"FWHM<2x = {fwhm:.1f} mm")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=int, default=2000,
                    help="G4 events per source (each event produces 1 photon)")
    ap.add_argument("--zscan-for-mag", type=float, default=26.0)
    ap.add_argument("--skip", type=str, default="",
                    help="Comma list of tests to skip: T1,T2,T3,T4")
    args = ap.parse_args()

    skip = set(args.skip.split(","))

    if "T1" not in skip:
        test_focal_position_at_calibrated_settings(num_events=args.events)
    if "T2" not in skip:
        test_magnification(num_events=args.events, zscan=args.zscan_for_mag)
    if "T3" not in skip:
        test_focal_plane_walk_with_zscan(num_events=args.events)
    if "T4" not in skip:
        test_dof_vs_fnumber(num_events=args.events)


if __name__ == "__main__":
    main()

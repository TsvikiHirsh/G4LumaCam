#!/usr/bin/env python3
"""
Calibrate lens focus parameters (zfine, zscan) against the photons in
SimPhotons/sim_data_*.csv.

Why this script exists
----------------------
zfine and zscan are coupled through the lens conjugate relation: a joint
optimizer that just minimizes spot size will slide both variables along a
ridge in parameter space and never converge to a unique answer. To break
the degeneracy we anchor on a *known physical depth* — a single neutron
interaction inside the scintillator gives a tight cluster of optical
photons whose true origin (x0, y0, z0) is known from the simulation.

Workflow
--------
  1) calibrate-zfine: pick an anchor depth z0_anchor, set
        zscan = SCINT_THICKNESS - (SCINT_THICKNESS - z0_anchor) / n
     (the apparent depth seen by the lens through the EJ200 glass),
     then sweep zfine and report the value that minimises the RMS
     image-plane spot. This is your calibrated zfine.

  2) scan-zscan: with zfine fixed at the calibrated value, sweep zscan
     for each requested depth slice and confirm the optimum follows
     zscan = apparent_z(z0). Useful for verifying the formula and for
     mapping z0 -> best-focus zscan empirically.

  3) scan-2d: 2D grid of RMS(zscan, zfine) for one depth slice. Lets you
     see the degeneracy ridge directly and pick a sensible anchor.

Photon selection
----------------
A "point source" is one (neutron_id, parent_id) group: a single charged
particle (proton/alpha) emits scintillation along its track, so its
photons sit in a tight x0,y0 cluster near the Bragg peak. We pick the
group with the most photons whose z0 falls inside the requested slice
and whose (x0, y0) standard deviation is below `--max-xy-spread`.

Usage
-----
  python optimize_focus.py calibrate-zfine --archive ARCHIVE --z0 20
  python optimize_focus.py scan-zscan      --archive ARCHIVE --zfine 12.75
  python optimize_focus.py scan-2d         --archive ARCHIVE --z0 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from lumacam.optics import Lens, VerbosityLevel
from rayoptics.environment import WvlSpec
from rayoptics.raytr import analyses


N_GLASS = 1.58            # EJ200 refractive index
SCINT_THICKNESS = 20.0    # mm (must match /lumacam/scintThickness in macro)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def apparent_z(z0, scint_thickness=SCINT_THICKNESS, n=N_GLASS):
    """Depth the lens sees for a source at physical depth z0 inside glass."""
    return scint_thickness - (scint_thickness - z0) / n


def load_photons(archive):
    sim_dir = Path(archive) / "SimPhotons"
    csvs = sorted(sim_dir.glob("sim_data_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"no sim_data_*.csv in {sim_dir}")
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    needed = {"x", "y", "z", "dx", "dy", "dz", "x0", "y0", "z0",
              "wavelength", "neutron_id", "parent_id"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns {missing}; rebuild G4 and re-run sim.")
    return df


def select_point_source(df, z0_target, z0_tol=0.5, max_xy_spread=0.2,
                        min_count=8):
    """Pick the (neutron_id, parent_id) group that best approximates a point
    source at the requested depth. Returns the photon sub-DataFrame.
    """
    near = df[(df.z0 > z0_target - z0_tol) & (df.z0 < z0_target + z0_tol)]
    if near.empty:
        raise ValueError(f"no photons with z0 in [{z0_target-z0_tol}, {z0_target+z0_tol}]")

    grouped = near.groupby(["neutron_id", "parent_id"])
    stats = grouped.agg(
        n=("id", "size"),
        x0_std=("x0", "std"),
        y0_std=("y0", "std"),
        z0_med=("z0", "median"),
    ).reset_index()
    stats = stats[stats.n >= min_count]
    stats = stats[(stats.x0_std.fillna(0) < max_xy_spread)
                  & (stats.y0_std.fillna(0) < max_xy_spread)]
    if stats.empty:
        raise ValueError(
            f"no point source meeting (n>={min_count}, xy_std<{max_xy_spread}) "
            f"in z0 slice {z0_target}±{z0_tol}. Try a larger sim or relax filters."
        )
    # Prefer the most populated, then the tightest in xy.
    stats = stats.sort_values(["n", "x0_std", "y0_std"],
                              ascending=[False, True, True])
    pick = stats.iloc[0]
    grp = near[(near.neutron_id == pick.neutron_id)
               & (near.parent_id == pick.parent_id)]
    return grp, pick


def make_lens(archive, kind="nikkor_58mm", fnumber=None, zfine_init=12.75):
    return Lens(archive=archive, kind=kind, fnumber=fnumber,
                zfine=zfine_init, verbosity=VerbosityLevel.QUIET)


def set_lens_wavelength(lens, wavelengths):
    """Install the photon wavelengths into the lens model.

    Lens.refocus() rebuilds self.opm from a deepcopy of self.opm0 on every
    call, so any spectral_region change on self.opm is wiped. We mutate
    opm0 (and opm) so subsequent refocus() calls keep our wavelengths.
    seq_model.wvlns is only updated by opm.update_model(); each refocus
    call ends with one, so wvlns becomes our set once refocus runs.
    """
    rounded = np.round(np.asarray(wavelengths, dtype=float), 1)
    counts = pd.Series(rounded).value_counts().sort_index()
    wvl_values = np.column_stack([counts.index.to_numpy(),
                                  np.ones(len(counts))])
    for model in (lens.opm0, lens.opm):
        if model is None:
            continue
        model.optical_spec.spectral_region = WvlSpec(wvl_values, ref_wl=0)
        model.update_model()


def round_wavelength(w):
    return float(np.round(w, 1))


def trace_spot_rms(lens, ray_df):
    """Trace rays through the current lens.opm and return in-plane RMS spot.

    Returns
    -------
    (rms_mm, n_traced)
        rms_mm: sqrt(mean((x-cx)^2 + (y-cy)^2)) at the image plane
        n_traced: number of rays that actually produced a finite image position
    """
    rays = [
        (np.array([r.x, r.y, r.z], dtype=float),
         np.array([r.dx, r.dy, r.dz], dtype=float),
         np.array([round_wavelength(r.wavelength)], dtype=float))
        for r in ray_df.itertuples()
    ]
    results = analyses.trace_list_of_rays(
        lens.opm, rays,
        output_filter="last", rayerr_filter="summary",
        check_apertures=lens.enforce_aperture,
    )
    xs, ys = [], []
    for r in results:
        if r is None:
            continue
        try:
            ray, _path, _wvl = r
            px, py, _pz = ray[0]
            if np.isfinite(px) and np.isfinite(py):
                xs.append(float(px))
                ys.append(float(py))
        except Exception:
            pass
    if len(xs) < 3:
        return float("inf"), len(xs)
    xs, ys = np.asarray(xs), np.asarray(ys)
    cx, cy = xs.mean(), ys.mean()
    rms = float(np.sqrt(np.mean((xs - cx) ** 2 + (ys - cy) ** 2)))
    return rms, len(xs)


def sweep_1d(lens, ray_df, axis, values, fixed):
    """Evaluate spot RMS over a 1-D sweep. `axis` is 'zfine' or 'zscan';
    `fixed` is the value of the *other* knob to hold constant.
    """
    rows = []
    for v in values:
        if axis == "zfine":
            lens.refocus(zscan=fixed, zfine=v)
        elif axis == "zscan":
            lens.refocus(zscan=v, zfine=fixed)
        else:
            raise ValueError(axis)
        rms, n = trace_spot_rms(lens, ray_df)
        rows.append({axis: v, "fixed_other": fixed, "rms_mm": rms, "n_traced": n})
    return pd.DataFrame(rows)


def parabolic_min(xs, ys):
    """Fit a parabola through the three samples around the minimum and return
    the vertex. Falls back to the discrete argmin if the parabola is concave-up
    cannot be fit.
    """
    ys = np.asarray(ys, dtype=float)
    xs = np.asarray(xs, dtype=float)
    i = int(np.argmin(ys))
    if i == 0 or i == len(ys) - 1:
        return float(xs[i]), float(ys[i])
    x0, x1, x2 = xs[i - 1], xs[i], xs[i + 1]
    y0, y1, y2 = ys[i - 1], ys[i], ys[i + 1]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(denom) < 1e-12:
        return float(x1), float(y1)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a <= 0:
        return float(x1), float(y1)
    x_vert = -b / (2 * a)
    y_vert = y1 + a * (x_vert - x1) ** 2 + b * 0  # cheap evaluation; not critical
    return float(x_vert), float(y_vert)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_calibrate_zfine(args):
    df = load_photons(args.archive)
    print(f"Loaded {len(df)} photons from {args.archive}/SimPhotons/")

    grp, pick = select_point_source(df, args.z0, z0_tol=args.z0_tol,
                                    max_xy_spread=args.max_xy_spread,
                                    min_count=args.min_count)
    print(f"\nPoint source: neutron_id={int(pick.neutron_id)}, "
          f"parent_id={int(pick.parent_id)}, n_photons={int(pick.n)}, "
          f"z0_med={pick.z0_med:.3f} mm, x0_std={pick.x0_std:.4f} mm, "
          f"y0_std={pick.y0_std:.4f} mm")

    zscan_anchor = apparent_z(args.z0, args.scint_thickness, args.n)
    print(f"Anchor zscan = apparent_z({args.z0}) = {zscan_anchor:.4f} mm "
          f"(formula: T - (T - z0)/n, with T={args.scint_thickness}, n={args.n})")

    lens = make_lens(args.archive, kind=args.kind, fnumber=args.fnumber,
                    zfine_init=args.zfine_init)
    set_lens_wavelength(lens, grp["wavelength"].to_numpy())

    # Coarse sweep
    coarse = np.linspace(args.zfine_min, args.zfine_max, args.coarse_steps)
    print(f"\nCoarse zfine sweep over [{args.zfine_min}, {args.zfine_max}] "
          f"in {args.coarse_steps} steps...")
    coarse_df = sweep_1d(lens, grp, axis="zfine", values=coarse,
                         fixed=zscan_anchor)
    print(coarse_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    i = int(coarse_df.rms_mm.idxmin())
    z_best = float(coarse_df.zfine.iloc[i])
    span = (args.zfine_max - args.zfine_min) / max(args.coarse_steps - 1, 1)

    # Fine sweep around the coarse winner
    fine_lo = max(args.zfine_min, z_best - span)
    fine_hi = min(args.zfine_max, z_best + span)
    fine = np.linspace(fine_lo, fine_hi, args.fine_steps)
    print(f"\nFine zfine sweep over [{fine_lo:.3f}, {fine_hi:.3f}] "
          f"in {args.fine_steps} steps...")
    fine_df = sweep_1d(lens, grp, axis="zfine", values=fine, fixed=zscan_anchor)
    print(fine_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Parabolic refinement
    x_vert, _ = parabolic_min(fine_df.zfine.values, fine_df.rms_mm.values)

    # Evaluate at the parabolic vertex for a final RMS number
    lens.refocus(zscan=zscan_anchor, zfine=x_vert)
    final_rms, n_final = trace_spot_rms(lens, grp)

    print()
    print("=" * 60)
    print(f"Calibrated zfine  = {x_vert:.4f} mm")
    print(f"Anchor zscan      = {zscan_anchor:.4f} mm  (z0 = {args.z0} mm)")
    print(f"RMS spot at min   = {final_rms*1000:.2f} um   "
          f"({n_final}/{int(pick.n)} rays traced)")
    print("=" * 60)

    out = pd.concat([coarse_df.assign(stage="coarse"),
                     fine_df.assign(stage="fine")],
                    ignore_index=True)
    out_path = Path(args.archive) / "calibrate_zfine.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSweep saved to {out_path}")
    return x_vert


def cmd_scan_zscan(args):
    df = load_photons(args.archive)
    print(f"Loaded {len(df)} photons from {args.archive}/SimPhotons/")

    lens = make_lens(args.archive, kind=args.kind, fnumber=args.fnumber,
                    zfine_init=args.zfine)

    rows = []
    for z0 in args.z0:
        try:
            grp, pick = select_point_source(df, z0, z0_tol=args.z0_tol,
                                            max_xy_spread=args.max_xy_spread,
                                            min_count=args.min_count)
        except ValueError as e:
            print(f"  z0={z0}: SKIP ({e})")
            continue

        set_lens_wavelength(lens, grp["wavelength"].to_numpy())
        zscan_pred = apparent_z(z0, args.scint_thickness, args.n)

        # First sweep within the user's window. Boundary hits mean the true
        # minimum is outside; widen and re-sweep up to args.max_widen times.
        win_lo = args.zscan_window_lo
        win_hi = args.zscan_window_hi
        for _ in range(args.max_widen + 1):
            zvals = np.linspace(zscan_pred + win_lo, zscan_pred + win_hi,
                                args.steps)
            sweep = sweep_1d(lens, grp, axis="zscan", values=zvals,
                             fixed=args.zfine)
            i = int(sweep.rms_mm.idxmin())
            if 0 < i < len(sweep) - 1:
                break  # interior minimum — done
            # Hit a boundary: shift the window in that direction by its width.
            width = win_hi - win_lo
            if i == 0:
                win_lo -= width
            else:
                win_hi += width
        x_vert, _ = parabolic_min(sweep.zscan.values, sweep.rms_mm.values)

        lens.refocus(zscan=x_vert, zfine=args.zfine)
        rms_at_vert, _ = trace_spot_rms(lens, grp)
        lens.refocus(zscan=zscan_pred, zfine=args.zfine)
        rms_at_pred, _ = trace_spot_rms(lens, grp)

        rows.append({
            "z0_target": z0,
            "z0_actual_med": pick.z0_med,
            "n_photons": int(pick.n),
            "zscan_predicted": zscan_pred,
            "rms_um_at_predicted": rms_at_pred * 1000.0,
            "zscan_best": x_vert,
            "rms_um_at_best": rms_at_vert * 1000.0,
            "delta": x_vert - zscan_pred,
        })

    if not rows:
        print("No valid depth slices.")
        return

    res = pd.DataFrame(rows)
    print()
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    out_path = Path(args.archive) / "scan_zscan.csv"
    res.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


def cmd_scan_2d(args):
    df = load_photons(args.archive)
    print(f"Loaded {len(df)} photons from {args.archive}/SimPhotons/")

    grp, pick = select_point_source(df, args.z0, z0_tol=args.z0_tol,
                                    max_xy_spread=args.max_xy_spread,
                                    min_count=args.min_count)
    print(f"Point source: neutron_id={int(pick.neutron_id)}, "
          f"parent_id={int(pick.parent_id)}, n_photons={int(pick.n)}, "
          f"z0_med={pick.z0_med:.3f} mm")

    lens = make_lens(args.archive, kind=args.kind, fnumber=args.fnumber)
    set_lens_wavelength(lens, grp["wavelength"].to_numpy())

    zfines = np.linspace(args.zfine_min, args.zfine_max, args.zfine_steps)
    zscans = np.linspace(args.zscan_min, args.zscan_max, args.zscan_steps)
    grid = np.zeros((len(zfines), len(zscans)), dtype=float)
    print(f"Tracing {len(zfines)} x {len(zscans)} = "
          f"{len(zfines)*len(zscans)} grid points...")
    for i, zf in enumerate(zfines):
        for j, zs in enumerate(zscans):
            lens.refocus(zscan=zs, zfine=zf)
            grid[i, j], _ = trace_spot_rms(lens, grp)

    # Flatten to a long-form CSV that any plotter can ingest.
    rows = []
    for i, zf in enumerate(zfines):
        for j, zs in enumerate(zscans):
            rows.append({"zfine": zf, "zscan": zs, "rms_mm": grid[i, j]})
    res = pd.DataFrame(rows)
    out_path = Path(args.archive) / f"scan_2d_z0_{args.z0:.1f}.csv"
    res.to_csv(out_path, index=False)
    print(f"\nGrid saved to {out_path}")

    finite = np.isfinite(grid)
    if finite.any():
        ij = np.unravel_index(np.nanargmin(np.where(finite, grid, np.inf)),
                              grid.shape)
        print(f"\nDiscrete minimum: "
              f"zfine={zfines[ij[0]]:.4f}, zscan={zscans[ij[1]]:.4f}, "
              f"RMS={grid[ij]*1000:.2f} um")
        print(f"Formula-predicted zscan for z0={args.z0}: "
              f"{apparent_z(args.z0, args.scint_thickness, args.n):.4f} mm")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--archive", required=True,
                        help="Archive dir containing SimPhotons/sim_data_*.csv")
    common.add_argument("--kind", default="nikkor_58mm")
    common.add_argument("--fnumber", type=float, default=None,
                        help="Override lens fnumber (default: kind default)")
    common.add_argument("--n", type=float, default=N_GLASS,
                        help="Scintillator refractive index (default 1.58)")
    common.add_argument("--scint-thickness", type=float, default=SCINT_THICKNESS,
                        help="Scintillator thickness mm (default 20)")
    common.add_argument("--z0-tol", type=float, default=0.5,
                        help="Half-width of z0 slice in mm (default 0.5)")
    common.add_argument("--max-xy-spread", type=float, default=0.2,
                        help="Max x0/y0 std-dev in mm for a 'point source' (default 0.2)")
    common.add_argument("--min-count", type=int, default=8,
                        help="Min photons in a point-source group (default 8)")

    a = sub.add_parser("calibrate-zfine", parents=[common])
    a.add_argument("--z0", type=float, required=True,
                   help="Anchor depth in mm (the depth you want in focus)")
    a.add_argument("--zfine-init", type=float, default=12.75)
    a.add_argument("--zfine-min", type=float, default=0.0)
    a.add_argument("--zfine-max", type=float, default=25.0)
    a.add_argument("--coarse-steps", type=int, default=11)
    a.add_argument("--fine-steps", type=int, default=11)
    a.set_defaults(func=cmd_calibrate_zfine)

    b = sub.add_parser("scan-zscan", parents=[common])
    b.add_argument("--zfine", type=float, required=True,
                   help="Calibrated zfine to hold fixed")
    b.add_argument("--z0", type=float, nargs="+", required=True,
                   help="One or more depth targets, mm")
    b.add_argument("--zscan-window-lo", type=float, default=-5.0,
                   help="zscan search window low offset from predicted (mm)")
    b.add_argument("--zscan-window-hi", type=float, default=5.0)
    b.add_argument("--steps", type=int, default=15)
    b.add_argument("--max-widen", type=int, default=3,
                   help="If the min sits on a boundary, double the window up "
                        "to this many times before giving up.")
    b.set_defaults(func=cmd_scan_zscan)

    c = sub.add_parser("scan-2d", parents=[common])
    c.add_argument("--z0", type=float, required=True)
    c.add_argument("--zfine-min", type=float, default=8.0)
    c.add_argument("--zfine-max", type=float, default=17.0)
    c.add_argument("--zfine-steps", type=int, default=10)
    c.add_argument("--zscan-min", type=float, default=0.0)
    c.add_argument("--zscan-max", type=float, default=25.0)
    c.add_argument("--zscan-steps", type=int, default=10)
    c.set_defaults(func=cmd_scan_2d)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

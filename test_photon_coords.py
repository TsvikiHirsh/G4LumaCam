"""
Verify EventProcessor's photon-coordinate scheme against the underlying physics.

CSV schema after the Option-C switch:
  x,y,z,dx,dy,dz        : exit pos at monitor (z = SCINT_THICKNESS) + refracted in-air dir
                          (this is what Python rayoptics traces from)
  x0,y0,z0,dx0,dy0,dz0  : in-glass birth pos + pre-refraction direction
                          (analysis only; lets us check Snell + straight travel)

Three checks:
  1. Exit z is ~constant and matches the scintillator thickness.
  2. Lateral drift: x = x0 + (dx0/dz0)*(z-z0)   and same for y.
     This verifies the photon travels in a straight line inside the glass.
  3. Snell's law at the front face: sin(theta_air) = n * sin(theta_glass),
     where sin(theta) = sqrt(d_x^2 + d_y^2) since |d|=1.

Synthetic tests run with no inputs (math only).
The CSV tests run only if SimPhotons/sim_data_*.csv exists with the new schema.

Usage:
  python test_photon_coords.py
"""
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

N_GLASS = 1.58  # EJ200 refractive index, matches MaterialBuilder


def refract_through_front_face(dx_g, dy_g, dz_g, n=N_GLASS):
    """Snell's law for a ray exiting a +z face from index n into index 1.

    Returns the in-air unit direction (dx_a, dy_a, dz_a) or None for TIR.
    Lateral component is multiplied by n (no flip), z component is
    re-derived from the unit-length constraint.
    """
    dx_a = n * dx_g
    dy_a = n * dy_g
    lat2 = dx_a * dx_a + dy_a * dy_a
    if lat2 >= 1.0:
        return None  # total internal reflection
    dz_a = math.sqrt(1.0 - lat2)
    return dx_a, dy_a, dz_a


# ----------------------------- synthetic tests -----------------------------

def test_snell_synthetic():
    """sin(theta_air) / sin(theta_glass) must equal n for arbitrary in-glass rays."""
    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(1000):
        # Pick a forward-going in-glass direction with small enough lateral
        # component to actually exit (sin theta_glass < 1/n => no TIR).
        theta_g = rng.uniform(0.0, math.asin(1.0 / N_GLASS) * 0.99)
        phi = rng.uniform(0.0, 2 * math.pi)
        dx_g = math.sin(theta_g) * math.cos(phi)
        dy_g = math.sin(theta_g) * math.sin(phi)
        dz_g = math.cos(theta_g)
        out = refract_through_front_face(dx_g, dy_g, dz_g)
        assert out is not None, "spuriously rejected as TIR"
        dx_a, dy_a, dz_a = out
        sin_g = math.hypot(dx_g, dy_g)
        sin_a = math.hypot(dx_a, dy_a)
        assert math.isclose(sin_a, N_GLASS * sin_g, rel_tol=1e-12)
        assert math.isclose(dx_a * dx_a + dy_a * dy_a + dz_a * dz_a, 1.0, abs_tol=1e-12)
        n_ok += 1
    print(f"  [synthetic Snell] {n_ok} samples, all match sin_air = n * sin_glass")


def test_tir_synthetic():
    """Above the critical angle, refraction should return None (TIR)."""
    theta_c = math.asin(1.0 / N_GLASS)
    # 1% past the critical angle => certainly TIR
    theta_g = theta_c * 1.01
    dx_g = math.sin(theta_g)
    dy_g = 0.0
    dz_g = math.cos(theta_g)
    out = refract_through_front_face(dx_g, dy_g, dz_g)
    assert out is None, f"expected TIR at theta_g = {math.degrees(theta_g):.2f} deg"
    print(f"  [synthetic TIR] correctly flagged at theta_g = {math.degrees(theta_g):.3f} deg "
          f"(theta_c = {math.degrees(theta_c):.3f} deg)")


# ------------------------------ CSV tests ----------------------------------

def _load_latest_csv():
    """Find the most recent sim_data_*.csv with the new schema, or None."""
    repo = Path(__file__).resolve().parent
    csvs = sorted((repo / "SimPhotons").glob("sim_data_*.csv"))
    needed = {"x", "y", "z", "dx", "dy", "dz", "x0", "y0", "z0", "dx0", "dy0", "dz0"}
    for path in reversed(csvs):
        try:
            df = pd.read_csv(path, nrows=1)
        except Exception:
            continue
        if needed.issubset(df.columns):
            return path, pd.read_csv(path)
    return None, None


def test_exit_z_constant(df):
    """Every photon's exit z should equal the scintillator thickness."""
    z = df["z"].to_numpy()
    z_min, z_max = float(z.min()), float(z.max())
    spread = z_max - z_min
    # All photons enter MonitorPhys at the same plane, so the spread should be
    # at the level of single-precision rounding of one number, not millimeters.
    assert spread < 1e-3, f"z spread {spread:.3e} mm > 1um — exit plane is not flat"
    print(f"  [exit z] N={len(z)} photons, z in [{z_min:.4f}, {z_max:.4f}] mm, "
          f"spread {spread*1e3:.2f} um")


def test_drift_consistency(df, abs_tol_mm=2e-3):
    """x = x0 + (dx0/dz0)*(z-z0); same for y. Verifies straight travel in glass."""
    z_minus_z0 = df["z"].to_numpy() - df["z0"].to_numpy()
    pred_x = df["x0"].to_numpy() + (df["dx0"] / df["dz0"]).to_numpy() * z_minus_z0
    pred_y = df["y0"].to_numpy() + (df["dy0"] / df["dz0"]).to_numpy() * z_minus_z0
    res_x = df["x"].to_numpy() - pred_x
    res_y = df["y"].to_numpy() - pred_y
    max_x = float(np.abs(res_x).max())
    max_y = float(np.abs(res_y).max())
    # Loose tolerance: x0,y0,z0 are written at 4 decimal places (0.1 um),
    # and dx0/dz0 can amplify rounding by ~20 over (z-z0).
    assert max_x < abs_tol_mm, f"x drift residual {max_x:.3e} mm exceeds {abs_tol_mm}"
    assert max_y < abs_tol_mm, f"y drift residual {max_y:.3e} mm exceeds {abs_tol_mm}"
    print(f"  [drift] max |x - x_pred| = {max_x*1e3:.2f} um, "
          f"max |y - y_pred| = {max_y*1e3:.2f} um")
    print(f"  [drift] median |dx0/dz0|*(z-z0) = "
          f"{np.median(np.abs((df['dx0']/df['dz0']).to_numpy()*z_minus_z0))*1e3:.1f} um")


def test_snell_on_data(df, rel_tol=1e-3):
    """For each photon, sin(theta_air)/sin(theta_glass) ~= n."""
    sin_g = np.hypot(df["dx0"].to_numpy(), df["dy0"].to_numpy())
    sin_a = np.hypot(df["dx"].to_numpy(),  df["dy"].to_numpy())
    # Skip rays that are very close to axial — ratio numerically unstable.
    mask = sin_g > 1e-3
    if mask.sum() < 10:
        print("  [snell on data] WARN: not enough non-axial rays to test")
        return
    ratio = sin_a[mask] / sin_g[mask]
    med = float(np.median(ratio))
    p05, p95 = (float(np.percentile(ratio, p)) for p in (5, 95))
    # Median should be very close to n; tails widen because dx0 rounding
    # to 6 dp shows up in the denominator for small angles.
    assert abs(med - N_GLASS) / N_GLASS < rel_tol, \
        f"median Snell ratio {med:.5f} vs n={N_GLASS}"
    print(f"  [snell on data] N={mask.sum()} rays, "
          f"median sin_air/sin_glass = {med:.5f} (n = {N_GLASS})")
    print(f"  [snell on data] 5th/95th percentile = {p05:.4f} / {p95:.4f}")


def test_unit_direction(df, abs_tol=1e-4):
    """|d| = 1 for both in-air and in-glass direction columns."""
    mag_a = np.sqrt(df["dx"]**2 + df["dy"]**2 + df["dz"]**2).to_numpy()
    mag_g = np.sqrt(df["dx0"]**2 + df["dy0"]**2 + df["dz0"]**2).to_numpy()
    assert np.allclose(mag_a, 1.0, atol=abs_tol), f"|d_air| not unit: max dev {np.max(np.abs(mag_a-1)):.3e}"
    assert np.allclose(mag_g, 1.0, atol=abs_tol), f"|d_glass| not unit: max dev {np.max(np.abs(mag_g-1)):.3e}"
    print(f"  [unit dir] max |1 - |d_air||  = {np.max(np.abs(mag_a-1)):.2e}")
    print(f"  [unit dir] max |1 - |d_glass||= {np.max(np.abs(mag_g-1)):.2e}")


# ----------------------------- entry point ---------------------------------

def main():
    print("=== Synthetic tests (no G4 input needed) ===")
    test_snell_synthetic()
    test_tir_synthetic()

    print("\n=== Tests on the latest SimPhotons/sim_data_*.csv ===")
    path, df = _load_latest_csv()
    if df is None:
        print("  SKIPPED: no sim_data_*.csv with the new schema yet.")
        print("  Rebuild G4LumaCam and run a short simulation, then re-run this script.")
        return 0
    print(f"  source: {path.name}  ({len(df)} photons)")
    test_unit_direction(df)
    test_exit_z_constant(df)
    test_drift_consistency(df)
    test_snell_on_data(df)
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

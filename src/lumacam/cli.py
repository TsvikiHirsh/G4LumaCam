"""
lumacam-trace — CLI entry point for the trace_rays stage.

Converts SimPhotons CSV files into TracedPhotons CSVs and TPX3 binary files.

Usage:
    lumacam-trace DATA_ROOT [options]

Examples:
    lumacam-trace /data/run42 --blob 0.5 --gain 720 --decay-time 30 --deadtime 600 --zfine 12.7
    lumacam-trace /data/run42 --blob 0.5 --param gain=720 --param noise=0.05 --simulate-ccw
"""

import argparse
import sys
from pathlib import Path


def _coerce(value: str):
    """Convert a string to int, float, or leave as str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lumacam-trace",
        description="Trace rays from SimPhotons to TPX3 files and TracedPhotons CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "data_root",
        metavar="DATA_ROOT",
        help="Path to the experiment root directory (must contain a SimPhotons/ sub-directory).",
    )

    # --- optics ---
    optics = p.add_argument_group("optics")
    optics.add_argument("--zfine", type=float, default=12.7,
                        help="Lens fine-focus offset in mm.")
    optics.add_argument("--zscan", type=float, default=0.0,
                        help="Z-scan offset in mm.")
    optics.add_argument("--fnumber", type=float, default=None,
                        help="F-number override (default: use model value).")
    optics.add_argument("--calibrate", action="store_true", default=False,
                        help="Trace a virtual point-source grid through the lens "
                             "(once per run, a few percent overhead) and write each "
                             "photon's expected centroid pixel as x_opt/y_opt columns "
                             "in TracedPhotons. After 'empindex --sim-merge' they "
                             "appear as sim/x_opt, sim/y_opt; ev/x - sim/x_opt "
                             "isolates reconstruction effects from lens optics.")
    optics.add_argument("--no-enforce-aperture", action="store_true", default=False,
                        help="Disable physical aperture enforcement during tracing. "
                             "Restores pre-fnumber-enforce-apertures behavior: no iris "
                             "sizing, no per-surface clipping. fnumber becomes a "
                             "paraxial-only parameter (no DoF/throughput effect) but "
                             "all rays propagate. Use for fast/approximate runs.")

    # --- detector / saturation ---
    det = p.add_argument_group("detector / saturation")
    det.add_argument("--source", choices=["hits", "photons"], default=None,
                     help="Workflow: 'hits' applies saturation + writes TPX3; "
                          "'photons' exports directly. Auto-detected if omitted.")
    det.add_argument("--deadtime", type=float, default=600.0,
                     help="Pixel deadtime in ns.")
    det.add_argument("--blob", type=float, default=1.0,
                     help="Blob radius in pixels.")
    det.add_argument("--blob-variance", type=float, default=0.0,
                     help="Blob radius variance.")
    det.add_argument("--decay-time", type=float, default=30.0,
                     help="Scintillator decay time in ns.")
    det.add_argument("--gain", type=float, default=10000.0,
                     help="Detector gain.")
    det.add_argument("--detector-model", type=str, default="image_intensifier_gain",
                     help="Detector model name.")
    det.add_argument("--param", metavar="KEY=VALUE", action="append", default=[],
                     help="Extra model parameter (repeatable). "
                          "Example: --param noise=0.05")

    # --- CCW simulation ---
    p.add_argument("--simulate-ccw", action="store_true", default=False,
                   help="Simulate the -25 ns coarse-clock wraparound artefact in the "
                        "TPX3 output and add a coarse_clock_wrap column to TracedPhotons. "
                        "Off by default.")

    # --- output ---
    out = p.add_argument_group("output")
    out.add_argument("--split-method", choices=["auto", "pulse", "file"], default="auto",
                     help="TPX3 file split strategy.")
    out.add_argument("--suffix", type=str, default="",
                     help="Suffix appended to output file names.")
    out.add_argument("--sym-suffix", type=str, default=None, metavar="DIR",
                     help="After tracing, create a tpx3Files symlink inside DIR pointing "
                          "to the actual tpx3Files output. Useful when the empindex working "
                          "directory differs from the trace archive (e.g. "
                          "--suffix z30 --sym-suffix openbeam_ptb/z30).")

    # --- performance ---
    perf = p.add_argument_group("performance")
    perf.add_argument("--n-processes", type=int, default=None,
                      help="Number of worker processes (default: all CPUs).")
    perf.add_argument("--chunk-size", type=int, default=1000,
                      help="Ray-tracing chunk size per worker.")
    perf.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility.")
    perf.add_argument("--no-progress", action="store_true", default=False,
                      help="Suppress the progress bar.")

    # --- verbosity ---
    p.add_argument("--verbosity", type=int, choices=[0, 1, 2, 3], default=1,
                   help="0=quiet, 1=basic, 2=detailed, 3=debug.")

    return p


def trace_rays_main():
    parser = _build_parser()
    args = parser.parse_args()

    # Parse --param KEY=VALUE pairs into a dict; seed with --gain default
    model_params = {"gain": args.gain}
    for kv in args.param:
        if "=" not in kv:
            parser.error(f"--param must be in KEY=VALUE format, got: {kv!r}")
        k, _, v = kv.partition("=")
        model_params[k.strip()] = _coerce(v.strip())

    # Import here so that import errors surface only when the CLI is actually used
    try:
        from lumacam.optics import Lens
    except ImportError as exc:
        sys.exit(f"Failed to import lumacam: {exc}")

    lens = Lens(args.data_root, enforce_aperture=not args.no_enforce_aperture)
    lens.trace_rays(
        zfine=args.zfine,
        zscan=args.zscan,
        fnumber=args.fnumber,
        source=args.source,
        deadtime=args.deadtime,
        blob=args.blob,
        blob_variance=args.blob_variance,
        decay_time=args.decay_time,
        detector_model=args.detector_model,
        model_params=model_params,
        simulate_ccw=args.simulate_ccw,
        calibrate=args.calibrate,
        split_method=args.split_method,
        suffix=args.suffix,
        n_processes=args.n_processes,
        chunk_size=args.chunk_size,
        seed=args.seed,
        progress_bar=not args.no_progress,
        verbosity=args.verbosity,
    )

    if args.sym_suffix:
        data_root = Path(args.data_root).resolve()
        if args.suffix:
            tpx3_src = data_root / args.suffix / "tpx3Files"
        else:
            tpx3_src = data_root / "tpx3Files"
        sym_dir = Path(args.sym_suffix)
        if not sym_dir.is_absolute():
            sym_dir = Path.cwd() / sym_dir
        sym_dir = sym_dir.resolve()
        sym_dir.mkdir(parents=True, exist_ok=True)
        tpx3_link = sym_dir / "tpx3Files"
        if tpx3_link.is_symlink():
            tpx3_link.unlink()
        tpx3_link.symlink_to(tpx3_src)
        print(f"Symlink created: {tpx3_link} -> {tpx3_src}")

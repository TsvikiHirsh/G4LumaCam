"""Find where the actual aperture stop sits on each lens, so we know
which surface to size programmatically."""
import sys, math
sys.path.insert(0, "/work/nuclear/G4LumaCam/src")
import pandas as pd
from lumacam.optics import Lens

empty = pd.DataFrame({c: [] for c in
    ['x','y','z','dx','dy','dz','wavelength','nz','pz','id','neutron_id','pulse_id','toa']})

for kind in ["nikkor_58mm", "microscope"]:
    print("="*70); print(kind); print("="*70)
    L = Lens(data=empty, kind=kind, fnumber=2.0, zfine=0)
    sm = L.opm.seq_model
    print(f"  seq_model.stop_surface = {sm.stop_surface}")
    # First-order parameters
    pd_pkg = L.opm['analysis_results'].get('parax_data')
    if pd_pkg is not None:
        ax_ray, pr_ray, fod = pd_pkg
        print(f"  efl       = {fod.efl:.3f} mm")
        print(f"  fno       = {fod.fno:.3f}")
        print(f"  pp1 (front PP) = {fod.pp1:.3f} mm")
        print(f"  ppk (back PP)  = {fod.ppk:.3f} mm")
        print(f"  enp_dist  = {fod.enp_dist:.3f} mm")
        print(f"  enp_radius= {fod.enp_radius:.3f} mm")
        if sm.stop_surface is not None:
            si = sm.stop_surface
            print(f"  stop ax-ray height = {abs(ax_ray[si][0]):.3f} mm  "
                  f"(this is what the iris radius should be)")
            print(f"  stop existing max_aperture = {sm.ifcs[si].max_aperture:.3f} mm")
    # surfaces near where the stop would be sensible
    print(f"  total surfaces: {len(sm.ifcs)}")
    # Show max apertures of all surfaces
    aps = [(i, ifc.max_aperture) for i, ifc in enumerate(sm.ifcs)]
    print(f"  smallest 6: {sorted(aps, key=lambda x: x[1])[:6]}")
    print()

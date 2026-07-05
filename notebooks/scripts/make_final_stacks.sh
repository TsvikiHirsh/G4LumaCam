#!/bin/bash
# Produce the final TOF-resolved TIFF stacks (EventImages/{stack,sum}.tif) for
# the manuscript's imaging-performance demonstration (Sec. "Predicted Imaging
# Performance with Satellite-Aware Reconstruction"):
#
#   1. experimental graphite knife-edge (run 44) and open beam (run 45, air45)
#   2. high-statistics calibrated simulations of the same two geometries
#
# Each dataset is reconstructed with the four calibrated presets
# (best-inf / best-cog / best-first / best-largest). On the simulation
# archives the first preset traces with the calibrated detector model
# (lumacam-trace, gaussian_probabilistic optimum) and the later ones re-use
# the trace automatically.
#
# Usage:  bash notebooks/scripts/make_final_stacks.sh
set -e
export PATH=/root/micromamba/bin:$PATH

GRAPHITE44=/work/nuclear/PTB/data/PTB2024        # run 44, graphite knife edge
AIR45=/work/nuclear/PTB/data/air45               # run 45, open beam (~2 s)
SIM_GRAPHITE=/work/nuclear/G4LumaCam/notebooks/archive/graphite_ptb_1e7
SIM_OPENBEAM=/work/nuclear/G4LumaCam/notebooks/archive/openbeam_ptb_1e7

PRESETS="best-inf best-cog best-first best-largest"

for data in "$GRAPHITE44" "$AIR45" "$SIM_GRAPHITE" "$SIM_OPENBEAM"; do
    for p in $PRESETS; do
        echo "=== $data  [$p]"
        empindex run "$data" --params "$p" --events2image \
            --suffix "final_${p//-/_}" -q
    done
done

echo
echo "Done. Stacks under <dataset>/final_<preset>/EventImages/{stack,sum}.tif"

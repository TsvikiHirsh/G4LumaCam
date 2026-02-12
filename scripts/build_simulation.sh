#!/usr/bin/env bash
set -euo pipefail

# Build the Geant4 lumacam executable and install it into the current env.
# Requires CONDA_PREFIX (or equivalent) from the pixi/conda environment.

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
SRC_DIR="${ROOT_DIR}/src/G4LumaCam"
PREFIX="${CONDA_PREFIX:-}"

if [[ -z "${PREFIX}" ]]; then
  echo "Error: CONDA_PREFIX is not set; activate the pixi/conda environment first." >&2
  exit 1
fi

SYSROOT="${PREFIX}/x86_64-conda-linux-gnu/sysroot"
mkdir -p "${BUILD_DIR}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -DCMAKE_SYSROOT="${SYSROOT}"
cmake --build "${BUILD_DIR}"
install -m 755 "${BUILD_DIR}/lumacam" "${PREFIX}/bin/lumacam"

echo "Built and installed lumacam to ${PREFIX}/bin/lumacam"

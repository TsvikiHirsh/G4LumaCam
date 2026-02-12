#!/usr/bin/env sh
set -e

need="empir_export_events empir_export_photons empir_export_pixelActivations empir_pixel2photon_tpx3spidr empir_photon2event empir_event2image"

candidates=""
[ -n "${EMPIR_PATH:-}" ] && candidates="$candidates ${EMPIR_PATH}"
[ -n "${EMPIR_EXPORT_PATH:-}" ] && candidates="$candidates ${EMPIR_EXPORT_PATH}"
candidates="$candidates /opt/empir/v1.0.0/bin /opt/empir/v1.0.0/empir_export /opt/empir/bin"
if [ -n "${CONDA_PREFIX:-}" ]; then
  candidates="$candidates ${CONDA_PREFIX}/bin ${CONDA_PREFIX}/empir_export ${CONDA_PREFIX}/empir/bin"
fi

found=""
for d in $candidates; do
  [ -d "$d" ] || continue
  ok=1
  for b in $need; do
    if [ ! -x "$d/$b" ]; then ok=0; break; fi
  done
  if [ "$ok" -eq 1 ]; then
    found="$d"
    break
  fi
done

if [ -n "$found" ]; then
  export EMPIR_PATH="$found"
  export EMPIR_EXPORT_PATH="$found"
  export PATH="$found:$PATH"
  echo "EMPIR auto-detected at $found"
else
  echo "Warning: EMPIR binaries not found ($need). Set EMPIR_PATH manually." >&2
fi

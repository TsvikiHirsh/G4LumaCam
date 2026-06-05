# G4LumaCam: Optical Photon Coordinate Issue for Lens Tracing

## System Overview

A nuclear neutron detector that uses a **Geant4 simulation** (G4LumaCam) coupled to a **Python rayoptics lens trace** (lumacam). The pipeline is:

1. G4 simulates neutron → scintillation → optical photons in EJ200 scintillator
2. `EventProcessor.cc` records photons that exit the scintillator front face (z=20mm), filtering to only those aimed at the lens entrance pupil
3. Photon data written to `SimPhotons/sim_data*.csv`
4. Python (`lumacam/optics.py`) traces those photons through a Nikkor 58mm f/0.95 lens model using `rayoptics`
5. Image reconstructed on 256×256 pixel TPX3 sensor

## Geometry

```
Neutron beam → [back face z=0mm] --- EJ200 scintillator 20mm thick --- [front face z=20mm] → 441mm air → Lens → Sensor
               (beam entry)                                             (exit, toward lens)
```

- Scintillator: 120mm × 120mm × 20mm, refractive index n=1.58
- Back face (z=0mm): black coating (absorbing) — beam entry side
- Side faces: black coating (absorbing)
- Front face (z=20mm): thin monitor volume, photons pass through to lens
- Lens first surface: z = dist_from_obj + zscan = 461.535 + zscan mm from z=0

**Key consequence of black coatings**: all photons that reach the monitor were born traveling directly forward (no reflections). Therefore **birth direction = exit direction**.

## What Each Photon Has

For each optical photon recorded at the monitor, two positions are available:

| Name | Symbol | Value | How obtained |
|------|--------|-------|-------------|
| Birth position | (x₀, y₀, z₀) | Inside scintillator, z₀ ∈ [0, 20mm] | `tracks[tid].x0/y0/z0` — position at step 1 of optical photon |
| Exit position | (x_exit, y_exit, z_exit≈20mm) | At the monitor face | `prePos.x()/mm`, `prePos.y()/mm`, `prePos.z()/mm` |
| Direction | (dx, dy, dz) | Forward-going unit vector | Same at birth and exit (straight travel) |

The relationship between birth and exit:
```
x_exit = x0 + (dx/dz) × (20 - z0)   [up to ~2mm lateral drift for z0=0]
y_exit = y0 + (dy/dz) × (20 - z0)
```

## The Core Problem

**Which position to use for the Python ray trace?**

Python's `trace_list_of_rays` takes `(pt0, direction, wavelength)` where `pt0` is the 3D starting position of the ray, and the lens model has its first surface at `z = dist_from_obj + zscan`.

### Option A: Birth position (x0, y0, z0) + birth direction

- **Pro**: x0,y0 directly corresponds to the scintillation origin → image correctly maps to parent position → uniform good resolution across all depths
- **Pro**: z0 ∈ [0,20mm] enables depth scanning via `zscan` (focus at depth z0 when `zscan = 20 - (20-z0)/1.58`)
- **Con**: z0 is inside the scintillator glass (n=1.58), but Python traces in air — the refractive index correction is ignored

### Option B: Exit position (x_exit, y_exit, z=0) + exit direction

- **Pro**: Photon is now in air, so the air-trace is physically correct for the path from monitor to lens
- **Pro**: All photons start at z=0 → simple, consistent reference plane
- **Con**: x_exit ≠ x0 (lateral drift up to 2mm for photons born at z0=0mm) → image maps to exit position, not scintillation origin → **systematic position error that is depth-dependent** → resolution appears best at z0=20mm (no drift), worst at z0=0mm (~2mm drift ≈ 5 pixels)
- **Con**: The `zscan` needed to focus on depth z0 is `zscan = 20 - (20-z0)/1.58`, not simply `zscan = z0`

### Option C: Apparent position accounting for refraction

A photon born at z0 inside glass (n=1.58) appears to the lens (in air) as if it originated at:
```
z_apparent = 20 - (20 - z0) / 1.58
```
And its apparent lateral position is approximately x0 (same), but the direction in air is `n × dx0` (larger angle).

The "correct" physics would trace from `(x0, y0, z_apparent)` with refracted direction `(n×dx0, n×dy0, dz_corrected)`. This is not currently implemented.

## What Was Observed

| Config | Result |
|--------|--------|
| Birth pos + birth dir (Option A) | ✅ Uniform good resolution across all depths; zscan works to shift focal depth |
| Exit pos (z=0) + exit dir (Option B) | ❌ Resolution best at z0=20mm (front face), worst at z0=0mm — appears as "focus fixed at z=20mm" |
| Birth pos + birth dir, but re-tuning zfine simultaneously with zscan | ❌ Focus always returns to z=0 (zscan and zfine are coupled; optimizer cancels out zscan) |

## Analysis Columns Needed

The user needs the following data for analysis (separate from tracing):

| Column | Purpose |
|--------|---------|
| `x, y, z` → x0, y0, z0 | Python trace starting position AND scintillation origin for analysis |
| `dx, dy, dz` | Ray direction (birth = exit for black-coated crystal) |
| `px, py, pz` | Parent charged particle (proton/alpha) birth position = neutron interaction vertex |
| `nx, ny, nz` | Neutron first interaction position in scintillator |
| `toa` | Time of arrival |

Analysis uses:
- `|px - x0|` → track length (distance from parent birth to photon birth along track)
- `|nx - px|` → single scatter (≈0) vs multi-scatter (large)
- `z0` for depth reconstruction; `zscan = 20 - (20-z0)/1.58` to focus on depth z0

## Depth Scanning

**Correct approach**: fix `zfine` at the calibrated value, then **only vary `zscan`**:
```python
zscan = 20 - (20 - z0_target) / 1.58
# Examples (for 20mm scintillator, n=1.58):
# z0=0mm  → zscan ≈ 7.3mm
# z0=10mm → zscan ≈ 13.7mm  
# z0=20mm → zscan = 20mm
```

**Do NOT retune `zfine` when changing `zscan`** — they are coupled through the lens conjugate relation and will cancel each other out, always returning focus to z=0mm.

## Current State of EventProcessor.cc

The code has been going back and forth. The version that produced the working result used **birth position** for x,y,z and **birth direction** for dx,dy,dz. This needs to be the final decision and then locked in.

## Recommendation

Use **Option A** (birth position + birth direction) because:
1. It gives correct image reconstruction (no depth-dependent lateral error)
2. The refractive index correction for z0 is small compared to the 461mm object distance (~1.5% of total path)  
3. The black coating guarantees birth direction = exit direction
4. It restores the working behavior the user had before

The 20mm scintillator glass introduces a small apparent-depth shift (7.3–20mm range instead of 0–20mm), which means the actual scintillator depth range visible to the lens is compressed by 1/n. The `zscan = 20 - (20-z0)/1.58` formula correctly accounts for this when focusing on a specific depth.

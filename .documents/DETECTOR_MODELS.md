# Detector Models in G4LumaCam

This guide explains the different physical detector models available in G4LumaCam for simulating photon detection and sensor response.

## Overview

The `trace_rays()` and `saturate_photons()` methods support multiple physical models that can be selected via the `detector_model` parameter using lowercase strings. Each model simulates different aspects of photon detection physics:

```python
from lumacam import Lens

lens = Lens(archive="my_simulation")

# Typical workflow: use trace_rays() with detector model
lens.trace_rays(
    deadtime=600.0,
    blob=2.0,
    detector_model="gaussian_diffusion",
    model_params={'charge_coupling': 0.85}
)

# Or call saturate_photons() directly
result = lens.saturate_photons(
    detector_model="gaussian_diffusion",
    blob=1.5,
    deadtime=600.0
)
```

## Available Models

### 1. image_intensifier (Default)

**Physics:** Simulates a Micro-Channel Plate (MCP) based image intensifier coupled to an event camera.

**Behavior:**
- Photon hits intensifier phosphor screen
- Creates circular blob of secondary photons
- All pixels in blob activated simultaneously
- Exponential conversion delay
- Independent pixel deadtime

**Parameters:**
- `blob`: Maximum blob radius in pixels
- `blob_variance`: Radius variance (uniform distribution)
- `decay_time`: Exponential time constant for photon conversion (ns)
- `deadtime`: Pixel saturation window (ns)
- `min_tot`: Minimum time-over-threshold (ns)

**Example:**
```python
# Using trace_rays (typical workflow)
lens.trace_rays(
    detector_model="image_intensifier",
    blob=2.0,              # ~13 pixels per photon
    blob_variance=0.5,     # radius varies [1.5, 2.0]
    decay_time=100.0,      # 100ns phosphor decay
    deadtime=600.0         # 600ns pixel deadtime
)
```

**Best for:** Night vision cameras, photomultipliers with position sensitivity, scintillator imaging

---

### 2. gaussian_diffusion

**Physics:** Models charge diffusion in solid-state detectors with Gaussian point-spread function.

**Behavior:**
- Photon generates electron-hole pairs
- Charge diffuses with Gaussian distribution
- Charge collected in nearby pixels weighted by distance
- No conversion delay (direct detection)
- Supports charge coupling efficiency

**Parameters:**
- `blob`: Gaussian sigma (standard deviation) in pixels
- `deadtime`: Pixel saturation window (ns)
- `min_tot`: Minimum time-over-threshold (ns)
- `model_params`:
  - `charge_coupling` (0-1): Fraction of charge collected (default: 1.0)

**Example:**
```python
# Using trace_rays (typical workflow)
lens.trace_rays(
    detector_model="gaussian_diffusion",
    blob=1.5,              # σ = 1.5 pixels
    deadtime=400.0,
    model_params={'charge_coupling': 0.85}
)
```

**Best for:** CCDs, CMOS sensors, silicon photomultipliers (SiPM), semiconductor detectors

---

### 3. direct_detection

**Physics:** Simplest model - direct photon detection without spatial spreading.

**Behavior:**
- Each photon detected in single pixel
- No blob formation
- No conversion delay
- Fast computation

**Parameters:**
- `deadtime`: Pixel saturation window (ns)
- `min_tot`: Minimum time-over-threshold (ns)

**Example:**
```python
# Using trace_rays (typical workflow)
lens.trace_rays(
    detector_model="direct_detection",
    deadtime=300.0         # Shorter deadtime for fast sensors
)
```

**Best for:** Fast event cameras, APD arrays in Geiger mode, ideal point detectors

---

### 4. wavelength_dependent

**Physics:** Advanced intensifier model with wavelength-dependent quantum efficiency and gain.

**Behavior:**
- Photons filtered by quantum efficiency curve
- Blob size scales with wavelength (diffraction)
- Wavelength-dependent detection probability
- Exponential conversion delay

**Parameters:**
- `blob`: Base blob radius (scaled by wavelength)
- `decay_time`: Exponential time constant (ns)
- `deadtime`: Pixel saturation window (ns)
- `min_tot`: Minimum time-over-threshold (ns)
- `model_params`:
  - `qe_wavelength`: Array of wavelengths (nm)
  - `qe_values`: Quantum efficiency at each wavelength (0-1)
  - `wavelength_scaling`: Wavelength scaling factor (default: λ/500nm)

**Example:**
```python
# Typical bialkali photocathode response
lens.trace_rays(
    detector_model="wavelength_dependent",
    blob=2.0,
    decay_time=100.0,
    deadtime=600.0,
    model_params={
        'qe_wavelength': [300, 350, 400, 450, 500, 550, 600, 650],
        'qe_values':     [0.05, 0.15, 0.25, 0.30, 0.28, 0.20, 0.10, 0.02],
        'wavelength_scaling': 1.2
    }
)
```

**Note:** Requires `wavelength` column in input data. If not present, defaults to 500nm.

**Best for:** Multi-wavelength imaging, spectral response studies, realistic intensifier simulation

---

### 5. avalanche_gain

**Physics:** Models avalanche photodiodes (APDs) or photomultiplier tubes (PMTs) with stochastic gain.

**Behavior:**
- Poisson-distributed gain per photon (using Gamma approximation)
- Afterpulsing effects (delayed secondary pulses)
- Gain affects pixel TOT response
- Small spatial blob from avalanche region

**Parameters:**
- `blob`: Avalanche region radius (typically small, 0-1 pixel)
- `deadtime`: Pixel saturation window (ns)
- `min_tot`: Minimum time-over-threshold (ns)
- `model_params`:
  - `mean_gain`: Average gain (default: 100)
  - `gain_variance`: Gain fluctuation (default: 20)
  - `afterpulse_prob`: Probability of afterpulse (default: 0.01)
  - `afterpulse_delay`: Mean afterpulse delay (ns, default: 200)

**Example:**
```python
lens.trace_rays(
    detector_model="avalanche_gain",
    blob=0.5,              # Small avalanche region
    deadtime=500.0,
    model_params={
        'mean_gain': 150,
        'gain_variance': 30,
        'afterpulse_prob': 0.02,
        'afterpulse_delay': 250.0
    }
)
```

**Best for:** APDs, SiPMs, PMTs, any detector with avalanche multiplication

---

## Model Selection Guide

| Detector Type | Recommended Model | Key Parameters |
|---------------|-------------------|----------------|
| MCP Intensifier + Camera | `"image_intensifier"` | blob=2.0, decay_time=100 |
| CCD / CMOS Sensor | `"gaussian_diffusion"` | blob=1.0-2.0 (σ) |
| Fast Event Camera | `"direct_detection"` | deadtime=100-300 |
| Multi-color Intensifier | `"wavelength_dependent"` | QE curve + wavelength data |
| APD / SiPM / PMT | `"avalanche_gain"` | mean_gain=100-1000 |

## Common Parameters

All models support these common parameters:

### deadtime (ns)
Pixel saturation window. After a pixel activates, additional photons update TOT but don't create new events until deadtime expires.

**Typical values:**
- Event cameras: 100-600 ns
- CCDs: N/A (continuous integration)
- APDs: 50-500 ns

### min_tot (ns)
Minimum time-over-threshold. Ensures events have measurable duration.

**Typical values:** 10-50 ns

### seed (int)
Random seed for reproducibility. Use the same seed to get identical results across runs.

## Output Format

All models produce the same output DataFrame structure:

| Column | Description |
|--------|-------------|
| `pixel_x`, `pixel_y` | Integer pixel coordinates |
| `toa2` | Time of arrival (first activation) in ns |
| `time_diff` | Time-over-threshold (TOT) in ns |
| `photon_count` | Number of photon blobs during deadtime |
| `id`, `neutron_id` | Particle tracking IDs |
| `pulse_id`, `pulse_time_ns` | Pulse identification |
| `nz`, `pz` | Particle momentum components |

## Advanced Usage

### Comparing Models

```python
models = ["image_intensifier", "gaussian_diffusion", "direct_detection"]

results = {}
for model in models:
    results[model] = lens.saturate_photons(
        detector_model=model,
        blob=2.0,
        deadtime=600.0,
        seed=42  # Same seed for fair comparison
    )

# Compare photon counts
for name, df in results.items():
    print(f"{name}: {len(df)} events")
```

### Custom QE Curves

```python
import numpy as np

# Create custom quantum efficiency curve
wavelengths = np.linspace(300, 700, 50)
qe = np.exp(-((wavelengths - 450)**2) / (2 * 50**2))  # Gaussian centered at 450nm

lens.trace_rays(
    detector_model="wavelength_dependent",
    blob=2.0,
    deadtime=600.0,
    model_params={
        'qe_wavelength': wavelengths.tolist(),
        'qe_values': qe.tolist()
    }
)
```

### Optimizing Performance

For large datasets, choose models by computational cost:

**Fastest:** `"direct_detection"` (no blob calculation)
**Fast:** `"image_intensifier"` (simple circle test)
**Medium:** `"gaussian_diffusion"` (Gaussian evaluation)
**Slower:** `"wavelength_dependent"` (QE interpolation per photon)
**Slowest:** `"avalanche_gain"` (stochastic gain + afterpulses)

## Physics References

### Image Intensifier
- Circular blob approximates MCP pore multiplication + phosphor spreading
- Exponential decay models phosphor persistence
- Typical parameters from [Photonis Gen 3 specs]

### Gaussian Diffusion
- Models charge cloud spreading in semiconductors
- σ ≈ √(2Dt) where D is diffusion coefficient, t is drift time
- Typical σ = 1-3 pixels for CCDs

### Wavelength Dependence
- Photocathode QE from bialkali, multialkali, or GaAs curves
- Blob scaling accounts for diffraction: PSF ∝ λ/NA

### Avalanche Gain
- Gamma distribution approximates Poisson multiplication statistics
- Afterpulsing from trapped carriers in avalanche region
- Typical APD gain: 50-1000, SiPM gain: 10^5-10^6

## Troubleshooting

### "Missing wavelength column" warning
**Solution:** Add `wavelength` column to input data, or use a different model.

### Too many/few events
**Solution:** Adjust `blob` parameter or use `DIRECT_DETECTION` for 1:1 photon mapping.

### Unrealistic TOT values
**Solution:** Tune `deadtime` and `min_tot` parameters based on real sensor specs.

### Low quantum efficiency
**Solution:** Check `qe_values` in `WAVELENGTH_DEPENDENT` model, ensure values are 0-1 range.

---

## Version History

- **v0.5.0**: Added 5 detector models with selectable physics using lowercase string arguments
- **v0.4.0**: Original image_intensifier model only

## Contributing

To add a new detector model:

1. Add enum value to `DetectorModel` in `optics.py`
2. Add lowercase string mapping in `saturate_photons()` method
3. Create `_apply_<model_name>_model()` helper method
4. Add dispatch case in `saturate_photons()` main loop
5. Update `trace_rays()` documentation
6. Document parameters and physics in this guide
7. Add tests for validation

---

**Questions?** See the main README.md or open an issue on GitHub.

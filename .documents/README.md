# G4LumaCam Documentation

This folder contains detailed documentation for G4LumaCam detector models and physics.

## Contents

### [DETECTOR_MODELS.md](DETECTOR_MODELS.md)
Comprehensive guide to available detector models for photon detection simulation.

**Topics covered:**
- All 8 detector models with physics explanations
- Parameter descriptions and defaults
- Usage examples with `trace_rays()` and `saturate_photons()`
- Model selection guide
- Performance optimization tips
- Troubleshooting

**Models documented:**
1. `image_intensifier` - Default MCP intensifier
2. `gaussian_diffusion` - CCD/CMOS charge diffusion
3. `direct_detection` - Simple single-pixel
4. `wavelength_dependent` - Spectral QE response
5. `avalanche_gain` - APD/PMT with afterpulsing
6. **`image_intensifier_gain`** - Gain-dependent MCP (RECOMMENDED for TPX3)
7. **`timepix3_calibrated`** - TPX3-specific calibration
8. **`physical_mcp`** - Full physics simulation

### [PHYSICS_MODELS_DESIGN.md](PHYSICS_MODELS_DESIGN.md)
Literature review and physics-based design document for detector models.

**Topics covered:**
- Timepix3 detector specifications (Poikela et al. 2014)
- Image intensifier MCP physics
- Blob formation and gain scaling
- TOT calibration curves
- Literature references
- Implementation notes

**Key equations:**
- Blob size: σ ∝ (gain)^0.4
- TOT response: TOT = a + b × ln(Q/Q_ref)
- MCP gain: G = 10³-10⁴

## Quick Start

For typical TPX3 + image intensifier simulation:

```python
from lumacam import Lens

lens = Lens(archive="my_simulation")

# Recommended: Gain-dependent model
lens.trace_rays(
    deadtime=475,                          # TPX3 spec
    detector_model="image_intensifier_gain",
    gain=5000,                             # MCP voltage dependent
    sigma_0=1.0,                           # Base blob size
    decay_time=100                         # Phosphor persistence
)
```

## See Also

- Main README: `../README.md`
- Demo notebook: `../notebooks/detector_models_demo.ipynb`
- Test script: `../test_detector_models.py`

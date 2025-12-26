# Physical Models for Image Intensifier + Event Camera

## Literature Review

### Timepix3 Detector Characteristics
**Reference:** Poikela et al., "Timepix3: a 65K channel hybrid pixel readout chip with simultaneous ToA/ToT and sparse readout", *JINST* 9 C05013 (2014)

**Key Properties:**
- **Pixel size:** 55 × 55 μm²
- **Time resolution:** 1.56 ns (ToA), 25 ns (ToT)
- **Deadtime:** ~475 ns per pixel (configurable)
- **TOT response:** Logarithmic with deposited charge
  - TOT = a + b × ln(Q/Q₀)
  - Typical calibration: a ≈ 20-50 ns, b ≈ 40-80 ns
- **Dynamic range:** ~10-bit ToT, 10⁴ electrons to 10⁶ electrons

### Image Intensifier (Gen 2/Gen 3 MCP)
**References:**
- Photonis Technical Specs
- Csorba, "Image Tubes" (1985)
- Siegmund et al., "Performance of the Delay Line Detectors for the UVCS and SUMER Instruments on SOHO" (1997)

**Key Properties:**
- **Photocathode QE:** 10-40% (wavelength dependent)
- **MCP gain:** 10³-10⁴ (voltage controlled)
  - Gain formula: G = exp(V/V₀) where V₀ ~ 200V
  - Typical operating: 10³ @ 900V, 10⁴ @ 1100V
- **Phosphor screen:**
  - P20/P43/P46 phosphors
  - Decay time: 0.5-100 μs (phosphor dependent)
  - Spatial resolution: 30-70 lp/mm
- **PSF (Point Spread Function):**
  - Gaussian σ ≈ 20-100 μm (gain dependent)
  - Higher gain → larger spread
  - σ ∝ √(G × d) where d is MCP-phosphor gap

### Blob Formation Physics

**Electron multiplication in MCP:**
1. Photoelectron enters MCP channel
2. Gain G = 10³-10⁴ electrons out
3. Electrons accelerate to phosphor (~5-8 kV)
4. Phosphor emits N_photons ∝ G photons

**Blob size scaling:**
- **Theoretical:** R_blob ∝ √(G × E_electron)
- **Empirical fit:** σ_pixels = σ₀ × (G/G₀)^α
  - α ≈ 0.3-0.5 (from MCP datasheets)
  - σ₀ ≈ 0.5-1.5 pixels (at reference gain)

**Photon distribution in blob:**
- **Model:** Gaussian I(r) = (N_total/2πσ²) × exp(-r²/2σ²)
- N_total depends on:
  - MCP gain G
  - Photocathode QE
  - Phosphor efficiency (~20 photons/keV)

## Proposed Physical Models

### Model 1: `"image_intensifier"` (Current - Simple)
**Use case:** Fast simulation, educational

**Physics:**
- Fixed circular blob
- Uniform photon distribution
- Simple exponential delay
- Basic deadtime

**Parameters:**
- `blob`: Radius in pixels
- `decay_time`: Phosphor decay constant
- `deadtime`: Pixel saturation time

---

### Model 2: `"image_intensifier_gain"` (NEW - Gain-dependent)
**Use case:** Realistic simulation with variable gain

**Physics:**
- **Blob size:** σ = σ₀ × (gain/1000)^0.4
- **Photon distribution:** Gaussian I(r) ∝ exp(-r²/2σ²)
- **TOT:** Charge-weighted based on photon count
- **Deadtime:** Per-pixel (default 475 ns for TPX3)

**Parameters:**
- `gain`: MCP gain (100-50000, default 5000)
- `sigma_0`: Base blob sigma at gain=1000 (default 1.0 pixels)
- `gain_exponent`: Blob scaling exponent (default 0.4)
- `deadtime`: Pixel deadtime (default 475 ns)
- `tot_mode`: "linear" or "logarithmic"

**TOT Calculation:**
```python
if tot_mode == "logarithmic":
    # Timepix3-like response
    charge = photon_count * weight  # weighted sum
    TOT = tot_a + tot_b * ln(charge / charge_ref)
else:  # linear
    TOT = min_tot + photon_count * tot_slope
```

---

### Model 3: `"timepix3_calibrated"` (NEW - TPX3-specific)
**Use case:** Accurate Timepix3 simulation

**Physics:**
- Uses **Timepix3 calibration curve** from literature
- **Surrogate model** for charge-to-TOT conversion
- **Energy deposition** from photon statistics
- **Per-pixel calibration** variation (optional)

**Parameters:**
- `gain`: MCP gain
- `tot_a`: TOT calibration offset (default 30 ns)
- `tot_b`: TOT calibration slope (default 50 ns)
- `charge_ref`: Reference charge (default 1000 e⁻)
- `pixel_variation`: TOT calibration spread (default 0.05 = 5%)
- `deadtime`: 475 ns (TPX3 spec)

**Calibration curve (from Poikela et al. 2014):**
```
TOT[ns] = a + b × ln(Q[e⁻] / Q_ref)

Typical values:
a = 20-50 ns (varies per pixel)
b = 40-80 ns (varies per pixel)
Q_ref = 1000 e⁻
```

---

### Model 4: `"physical_mcp"` (NEW - Full physics)
**Use case:** High-fidelity simulation, research

**Physics:**
- **Poisson gain statistics:** G ~ Gamma(mean=gain, variance=gain×f)
- **Energy-dependent QE:** η(λ)
- **Phosphor decay:** Multi-exponential (fast + slow components)
- **MCP pore statistics:** Discrete channel simulation
- **PSF modeling:** Includes MCP-phosphor gap effects

**Parameters:**
- `gain`: Mean MCP gain
- `gain_noise_factor`: Excess noise factor (default 1.3)
- `mcp_voltage`: MCP voltage (800-1200 V)
- `phosphor_type`: "p20", "p43", "p46"
- `decay_fast`: Fast decay component (default 50 ns)
- `decay_slow`: Slow decay component (default 500 ns)
- `fast_fraction`: Fraction in fast component (default 0.7)
- `mcp_phosphor_gap`: Gap in mm (default 0.5)

---

## Recommended Model Selection

| Use Case | Model | Rationale |
|----------|-------|-----------|
| Quick testing | `"image_intensifier"` | Fast, simple |
| Standard simulation | `"image_intensifier_gain"` | Good balance of speed & physics |
| TPX3 experiments | `"timepix3_calibrated"` | Matches real detector |
| Publication-quality | `"physical_mcp"` | Full physics simulation |
| CCD/CMOS (no intensifier) | `"gaussian_diffusion"` | Direct detection |

## Implementation Notes

### Kwargs Support
Allow users to pass model parameters as kwargs:
```python
lens.trace_rays(
    deadtime=475,
    detector_model="image_intensifier_gain",
    gain=8000,                    # kwargs become model_params
    sigma_0=1.2,
    tot_mode="logarithmic"
)
```

### Default Values
- `detector_model=None` → `"image_intensifier_gain"` (new default)
- `gain=5000` (mid-range MCP operation)
- `deadtime=475` (TPX3 spec)

### TOT Improvements
Current implementation: `TOT = last_toa - first_toa`

New implementations:
1. **Charge-weighted:** `TOT = f(Σ charge_i)`
2. **Logarithmic:** `TOT = a + b × ln(Σ charge)`
3. **Energy-dependent:** `TOT = calibration_curve(E_deposited)`

## References

1. **Poikela, T. et al.** (2014). "Timepix3: a 65K channel hybrid pixel readout chip with simultaneous ToA/ToT and sparse readout." *Journal of Instrumentation*, 9(05), C05013.

2. **Llopart, X. et al.** (2007). "Timepix, a 65k programmable pixel readout chip for arrival time, energy and/or photon counting measurements." *Nuclear Instruments and Methods in Physics Research A*, 581(1-2), 485-494.

3. **Photonis** (2020). "Image Intensifier Tubes Technical Information." Photonis USA.

4. **Siegmund, O. H. W. et al.** (1997). "Performance of the Delay Line Detectors for the UVCS and SUMER Instruments on SOHO." *SPIE Proceedings*, 3114, 283-294.

5. **Csorba, I. P.** (1985). *Image Tubes*. Howard W. Sams & Co.

6. **Vallerga, J. et al.** (2009). "High speed multi-anode microchannel array detector system." *Nuclear Instruments and Methods A*, 606(3), 651-660.

## Validation

Recommended tests:
1. **Gain scaling:** Verify σ_blob ∝ gain^0.4
2. **TOT linearity:** Check TOT vs. photon count matches Timepix3 data
3. **Deadtime:** Confirm 475 ns pixel recovery
4. **Blob integral:** Verify ∫ I(r) dr = total photons

## Future Enhancements

- [ ] Add afterpulsing in MCP (ion feedback)
- [ ] Implement electron backscatter from phosphor
- [ ] Add halo around blob (low-energy electrons)
- [ ] Support multi-hit per pixel (TOT extension mode)
- [ ] Magnetic field effects on blob shape

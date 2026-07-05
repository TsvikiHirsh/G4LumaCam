# Changelog

All notable changes to G4LumaCam will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-07-04

### Changed
- **`gaussian_probabilistic` is now the default detector model**, with all
  defaults set to the values calibrated event-by-event against PTB fast-neutron
  data (air45, Sigma-chi2 = 0.52 +/- 0.02 over four per-event observables x two
  reconstruction modes):
  - `blob` = 0.405 px (intensifier point-spread sigma)
  - `decay_time` = 16.5 ns (P47 phosphor)
  - `n_secondaries` = 9 (detected pixels per photon gain spot)
  - `photon_keep_fraction` = 0.241 (effective optical yield ~ photocathode QE)
  - `ap_prob` = 0.012, `ap_rmax` = 5.5 px, `ap_secondaries` = 8 (afterpulse
    satellites at the literature rate, Mahon et al. 2024, NIM-A 1059 168816)
  - `zfine` = 12.6 mm (calibrated fine focus)
  These apply to both the Python API (`Lens.trace_rays()`) and the
  `lumacam-trace` CLI; every value can still be overridden per call.
  Set `ap_prob=0` to disable afterpulse satellites.

### Added
- **Afterpulse satellite component** in `gaussian_probabilistic`: each detected
  photon spawns Poisson(`ap_prob`) satellite mini-clusters displaced
  isotropically by a uniform-disc radius up to `ap_rmax` (~2x the
  photocathode-MCP proximity gap), modeling photoelectron backscattering in
  the MCP intensifier; optional `ap_delay` emission delay.
- **`photon_keep_fraction`** continuous effective-optical-yield parameter in
  the trace stage (replaces the discrete thinned-archive workflow).
- **`config.empir_params`**: `BEST_DETECTOR_MODEL` dict with the calibrated
  optimum, and `position_mode: "largest"` in the out-of-focus parameter set
  (largest-cluster event positioning - robust against afterpulse satellites).
- **Publication figure suite** (`notebooks/scripts/pubfigs/`) generating the
  manuscript figures from the calibration archives.
- **New notebook** `notebooks/calibrated_model_v06.ipynb`: the calibrated
  model, the afterpulse mechanism, and the `empindex --params best` /
  `--events2image` workflow.

### Companion (empindex)
- `empindex run` accepts built-in presets `best` / `best-inf` / `best-cog` /
  `best-first` / `best-largest` (calibrated reconstruction + trace sections),
  and a new `--events2image` flag writing TOF-resolved TIFF stacks
  (`EventImages/stack.tif`, `sum.tif`, `tof_bins.csv`).

## [0.5.0] - 2025-12-30

### Added
- **Advanced Detector Models**: 3 new physics-based detector models for high-fidelity simulation
  - `image_intensifier_gain`: Gain-dependent blob sizing (σ ∝ gain^0.4) - **RECOMMENDED for Timepix3+MCP**
  - `timepix3_calibrated`: Logarithmic TOT response with per-pixel variation (TOT = a + b × ln(Q))
  - `physical_mcp`: Full MCP physics with Poisson gain statistics and bi-exponential phosphor decay
- **Phosphor Screen Database**: Support for 4 phosphor types with auto-configuration
  - P47 (YAG:Ce): 70-100ns decay, modern Chevron MCPs (default)
  - P46 (Y₂SiO₅:Ce): ~70ns, high frame rate applications
  - P43 (Gd₂O₂S:Tb): ~1ms, traditional Gen 2/3 intensifiers
  - P20 (ZnCdS:Ag): 100ns + 1ms tail, legacy systems
- **Export Pixels Functionality**: `export_pixels()` method for exporting pixel activation data
  - Generates CSV files in `ExportedPixels` subfolder
  - Complementary to `export_photons()` method
  - Configurable via `export_pixels=False` parameter in Analysis
- **Comprehensive Documentation**:
  - [DETECTOR_MODELS.md](.documents/DETECTOR_MODELS.md): Complete guide to all 8 detector models
  - [DETECTOR_MODELS_SUMMARY.md](.documents/DETECTOR_MODELS_SUMMARY.md): Quick reference guide
  - [BLOB_VS_GAIN.md](.documents/BLOB_VS_GAIN.md): Explains difference between blob and gain parameters
  - [detector_models_comparison.ipynb](notebooks/detector_models_comparison.ipynb): Interactive demo notebook
- **Visualization**: Single event visualization using [visualize_pixel_map](https://github.com/TsvikiHirsh/visualize_pixel_map)

### Changed
- **Default Detector Model**: Changed from `image_intensifier` to `image_intensifier_gain`
  - Provides physics-based gain control and automatic blob sizing
  - Better matches real MCP+Timepix3 detector behavior
- **Default Parameters**:
  - `decay_time`: Changed from 10ns to 100ns (P47 phosphor standard)
  - Models now use gain-dependent blob calculation by default (blob=0)
- **Timepix3 Specifications**: Updated deadtime from 600ns to accurate 475ns spec
- **Model Interface**: Enhanced parameter passing through trace_rays() call chain
  - Added `detector_model` and `model_params` to all internal methods
  - Consistent parameter handling across single and grouped tracing

### Fixed
- **Gain Scaling Bug**: Fixed issue where `blob > 0` was overriding gain-dependent calculation
  - Setting `blob=0` now correctly enables automatic gain-based blob sizing
  - Critical for `image_intensifier_gain` and `physical_mcp` models
- **NameError Fix**: Resolved `detector_model` and `model_params` not being passed to internal methods
- **Simulation Workflow**: Corrected notebook examples to use `Simulate(archive).run(config)` pattern

## [0.4.0] - 2025-10-28

### Added
- **Tutorial**: New comprehensive [G4LumaCam Tutorial](notebooks/G4LumaCam_Tutorial.ipynb) covering simulation setup, ray tracing, and data analysis
- **Optical System Enhancements**:
  - Support for multiple lens types and improved focus management
  - ZMX lens file support for accurate optical modeling
  - `zfine` parameter for improved clarity in focus adjustments (replaces `focus` parameter)
  - Lens data files for JP1987-249119 and JP2000-019398 examples
  - Method to calculate and export first-order optical parameters
- **Analysis Features**:
  - MTF (Modulation Transfer Function) analysis capability
  - ROI (Region of Interest) analysis functionality with statistical outputs
  - `collect_analysis_results` method to gather MTF and ROI statistics from archive
  - Event-by-event processing with sum image option
  - `neutron_id` column for pixel-to-photon analysis traceability
- **TPX3 File Generation**:
  - Complete TPX3 file export functionality compatible with EMPIR and other reconstruction tools
  - Support for multiple TPX3 files per simulation
  - Event-based TPX3 file organization
  - `in_tpx` column to track pixels surviving saturation
- **Detector Effects**:
  - Saturation modeling with configurable blob size
  - Deadtime simulation (configurable, default 600ns)
  - Circle blob generation for realistic hit patterns
- **Neutron Source Options**:
  - Pulsed neutron source configuration
  - Uniform time spread for neutrons
  - Flux and frequency configuration options
  - Time spread parameters
  - Point ion configuration for radioactive decay simulations
  - Gamma line configuration with histogram support
- **Scintillator Support**:
  - LYSO scintillator material
  - Configurable scintillator thickness
  - Default scintillation properties for EJ-200
  - Sample thickness configuration
- **Data Processing**:
  - `groupby` functionality for organizing simulation data
  - Batch file support (CSV format)
  - Suffix parameter for subfolder organization
  - Optional event export to CSV
  - Enhanced chunk handling with parallel processing
- **Configuration Options**:
  - Sample width modification
  - Aperture adjustment
  - TDC and TOA precision improvements (260 ps TDC time)
  - Default `csv_batch_size` set to 1000, `num_events` reduced to 10000

### Changed
- **EMPIR Integration**: Three flexible methods for specifying EMPIR executable path (environment variable, runtime parameter, or default path)
- **Analysis Class**: Refactored to load default parameters from `empir_params.py`
- **Coordinate System**: Fixed TPX3 coordinate writing (resolves upper-half only issue)
- **Time Calculations**: Corrected trigger difference in TOF calculation, improved time precision
- **File Organization**:
  - Photon files now saved in ImportedPhotons directory
  - Improved directory creation logic
  - Enhanced archive structure for grouped data
- **Lens Class**:
  - Refactored initialization and focus adjustment logic
  - Improved optical model loading
  - Enhanced plotting capabilities with detailed docstrings
- **Performance**: Optimized ray processing with improved chunk handling and index preservation

### Fixed
- ZMX file loading issues
- Focus calculation with `zfine` parameter
- Coordinate translation in TPX3 files (y-axis 1-128 pixels issue)
- L-shaped geometry dimensions
- Scintillator thickness updating entire world geometry
- Event processor with deadtime enabled
- Sample thickness units and material definitions
- Pulse structure and trigger ID information
- TOA (Time of Arrival) calculation precision (changed to G4double)
- Merge of reconstructed event-by-event results
- File naming conventions for TPX3 files
- Verbosity level print statements
- Resource cleanup in GeometryConstructor
- Boolean value for TDC1 in pixel2photon configuration
- README links and formatting

### Removed
- Neutron Event Analyzer (NEA) dependency from setup.py
- Unused methods from codebase
- Debug output statements
- TriggerTimes class (trigger now provided through sim_data CSV files)

## [0.3.0] - Initial Public Release

### Added
- Initial public release of G4LumaCam
- Geant4-based neutron interaction simulation
- Basic optical ray tracing through LumaCam lens system
- EMPIR workflow integration
- Configurable neutron sources
- Multi-process support for simulations
- MIT License

---

[0.5.0]: https://github.com/TsvikiHirsh/G4LumaCam/releases/tag/v0.5.0
[0.4.0]: https://github.com/TsvikiHirsh/G4LumaCam/releases/tag/v0.4.0
[0.3.0]: https://github.com/TsvikiHirsh/G4LumaCam/releases/tag/v0.3.0

# G4LumaCam

A Geant4-based Simulator for LumaCam Event Camera

![LumaCam Simulation](https://github.com/TsvikiHirsh/G4LumaCam/blob/master/notebooks/lumacam_simulation.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

G4LumaCam is a Geant4-based simulation package for the LumaCam event camera that enables reconstruction of neutron events using the same analysis workflow as experimental data. The simulator generates standard Timepix-3 (TPX3) files that can be processed with various reconstruction tools, including the official EMPIR workflow. This flexibility allows researchers to simulate, validate, and optimize neutron detection setups before conducting physical experiments, reducing development time and costs.

## Key Features

- **High-Fidelity Physics**: Neutron interaction simulation based on Geant4 10.6 physics models
- **NCrystal Support**: Optional integration with NCrystal for accurate thermal neutron scattering in crystalline materials
- **Multiple Scintillators**: Support for EJ200, GS20, LYSO, and <sup>6</sup>LiF-ZnS:Ag scintillators with realistic optical properties
- **Realistic Optics**: Accurate optical ray tracing through the LumaCam lens system
- **Standard Output Format**: Generates TPX3 files compatible with multiple reconstruction tools
- **Flexible Reconstruction**: Use EMPIR for official workflow - just like in a real experiment!
- **Configurable Sources**: Customizable neutron source properties (energy, spatial distribution, flux, etc.)
- **Efficient Processing**: Multi-process support for large-scale simulations
- **End-to-End Workflow**: From particle generation to reconstructed images

## Quick Start

Check out our new [detailed tutorial](https://github.com/TsvikiHirsh/G4LumaCam/blob/master/notebooks/G4LumaCam_Tutorial.ipynb) for a comprehensive guide covering simulation setup, ray tracing, and data analysis.

### Basic Usage

```python
import lumacam

# 1. Run neutron source simulation
sim = lumacam.Simulate("openbeam")
config = lumacam.Config.neutrons_uniform_energy()
df = sim.run(config)

# 2. Trace rays through the optical system
lens = lumacam.Lens(archive="openbeam")
lens.trace_rays(blob=1.0, deadtime=600)  # 1px blob, 600ns deadtime
# This generates TPX3 files compatible with various reconstruction tools

# 3. Reconstruct using EMPIR (requires EMPIR license)
analysis = lumacam.Analysis(archive="archive/test/openbeam")
analysis.process(params="hitmap", event2image=True)
```

## Installation

### Prerequisites

**Geant4 via Docker** (recommended):
```bash
docker pull jeffersonlab/geant4:g4v10.6.2-ubuntu24
```

**Python Dependencies**:
- Python 3.7+
- [ray-optics](https://github.com/mjhoptics/ray-optics) - Optical ray tracing (instsalled automatically)
- NumPy, Pandas, Matplotlib (installed automatically)

**EMPIR** (optional - for official analysis workflow):

EMPIR is a proprietary reconstruction code for Timepix-3 detector data, available from [LoskoVision Ltd.](https://amscins.com/product/chronos-series/neutron-imaging/). **Note**: EMPIR is only required if you want to use the `lumacam.Analysis` workflow. The simulation generates standard TPX3 files that can be processed with alternative, open-source Timepix-3 reconstruction tools

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TsvikiHirsh/G4LumaCam.git
   cd G4LumaCam
   ```

2. **(Optional) Configure EMPIR path** before installation:
   ```bash
   export EMPIR_PATH=/path/to/empir/executables
   ```

3. **Install G4LumaCam**:

   **Standard installation** (without NCrystal):
   ```bash
   pip install .
   ```

   **With optional NCrystal support** (for crystalline materials):
   ```bash
   pip install .[ncrystal]
   ```

### NCrystal Support (Optional)

G4LumaCam supports **NCrystal** for simulating thermal neutron scattering in crystalline materials. This enables high-fidelity Bragg edge imaging and other crystallographic neutron techniques.

#### Installing with NCrystal

The easiest way to install G4LumaCam with NCrystal support is using the optional dependency syntax:

```bash
pip install .[ncrystal]
```

This will automatically install NCrystal-Geant4 and configure the build system to enable NCrystal support. During installation, you'll see:
```
-- NCrystal-Geant4 found - NCrystal support enabled
```

If you've already installed G4LumaCam without NCrystal, you can add it later:

```bash
pip install ncrystal-geant4
pip install --force-reinstall --no-deps .
```

Alternatively, install both NCrystal and Geant4 via conda:

```bash
conda install -c conda-forge ncrystal geant4
```

For manual installation, see the [NCrystal-Geant4 repository](https://github.com/mctools/ncrystal-geant4).

**Note**: If NCrystal is not installed, G4LumaCam will build successfully with standard NIST materials only.

#### Using NCrystal Materials

With NCrystal support enabled, you can define crystalline sample materials using NCrystal cfg-strings. The easiest way is to use the Python API with the pre-configured `neutrons_bragg_edge()` method:

```python
import lumacam

# Use the pre-configured Bragg edge configuration (Fe sample at 293K)
config = lumacam.Config.neutrons_bragg_edge()
sim = lumacam.Simulate("bragg_edge_test")
df = sim.run(config)
```

**Customize the material and parameters:**

```python
# Custom NCrystal material with different parameters
config = lumacam.Config.neutrons_bragg_edge(
    energy_min=0.001,  # eV (corresponds to ~9 Å)
    energy_max=0.5     # eV (corresponds to ~0.4 Å)
)
config.sample_material = "Al_sg225.ncmat;temp=80K"  # Aluminum FCC at 80K
config.sample_thickness = 1.0  # 10 mm thickness
config.num_events = 100000

sim = lumacam.Simulate("al_bragg_edge")
df = sim.run(config)
```

**Available NCrystal cfg-string examples:**
- `Fe_sg229.ncmat;temp=293.15K` - Iron (BCC) at room temperature
- `Al_sg225.ncmat;temp=80K` - Aluminum (FCC) at 80K
- `Si_sg227.ncmat;temp=300K` - Silicon (diamond cubic)
- `C_sg194_pyrolytic_graphite.ncmat` - Pyrolytic graphite

For macro files, use the `/lumacam/sampleMaterial` command:
```bash
/lumacam/sampleMaterial Fe_sg229.ncmat;temp=293.15K
```

#### NCrystal Physics

NCrystal provides:
- Accurate crystalline structure factors for Bragg scattering
- Inelastic scattering (phonons)
- Temperature-dependent cross sections
- Support for 100+ crystalline materials from the NCrystal database

The physics is automatically installed after `runMgr->Initialize()` and handles thermal neutrons (<5 eV). Higher energy neutrons and other particles continue to use standard Geant4 physics.

## Scintillator Materials

G4LumaCam supports multiple scintillator materials for neutron detection. You can select the scintillator type using the Python API or macro commands.

### Available Scintillators

- **EJ200** - Plastic scintillator (default)
  - Fast decay time (~2.1 ns)
  - High light output for charged particles
  - Suitable for fast neutron detection via recoil protons

- **GS20** - <sup>6</sup>Li-glass scintillator
  - ~8,000 photons/neutron for thermal neutrons
  - Good for thermal neutron imaging
  - Contains enriched <sup>6</sup>Li for neutron capture

- **LYSO** - Lutetium-yttrium oxyorthosilicate
  - High density (7.1 g/cm³)
  - Excellent for gamma-ray detection
  - Also sensitive to neutrons

- **ZnS** - <sup>6</sup>LiF-ZnS:Ag composite scintillator
  - ~160,000 photons/neutron for thermal neutrons
  - Highest light output among neutron scintillators
  - Optimized for thermal neutron imaging
  - Material composition: 27.8% <sup>6</sup>LiF, 55.6% ZnS, 16.6% binder
  - Optical properties from experimental measurements

### Setting Scintillator Type

**Using Python API:**
```python
import lumacam

config = lumacam.Config.neutrons_uniform_energy()
config.scintillator_material = "ZnS"  # Options: "EJ200", "GS20", "LYSO", "ZnS"
sim = lumacam.Simulate("my_simulation")
df = sim.run(config)
```

**Using macro commands:**
```bash
/lumacam/scintillatorMaterial ZnS
```

## Simulation Output & Reconstruction with EMPIR

G4LumaCam generates standard **TPX3 files** from the simulation, which are compatible with various Timepix-3 reconstruction tools.
### EMPIR (Official Workflow)
The `lumacam.Analysis` class provides seamless integration with EMPIR for the complete LumaCam reconstruction pipeline. This requires EMPIR licensing (see EMPIR Configuration below).

## EMPIR Configuration

G4LumaCam offers three methods to specify the EMPIR executable path:

### 1. Environment Variable (Global Configuration)
Set before installation for system-wide configuration:
```bash
export EMPIR_PATH=/path/to/empir/executables
pip install .
```

### 2. Runtime Parameter (Per-Session)
Specify when creating an Analysis object:
```python
analysis = lumacam.Analysis(
    archive="your_archive",
    empir_dirpath="/path/to/empir/executables"
)
```

### 3. Default Path (Fallback)
If unspecified, G4LumaCam searches for EMPIR in `./empir` relative to your working directory.

## Documentation

- **[Tutorial Notebook](__notebooks/tutorial.ipynb__)**: Step-by-step guide with examples

For additional support, please [open an issue](https://github.com/TsvikiHirsh/G4LumaCam/issues).

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## Citation

If you use G4LumaCam in your research, please cite:

```bibtex
@software{g4lumacam,
  author = {Hirsh, Tsviki Y.},
  title = {G4LumaCam: A Geant4-based Simulator for LumaCam Event Camera},
  url = {https://github.com/TsvikiHirsh/G4LumaCam},
  year = {2025},
}
```

## License

G4LumaCam is released under the MIT License. See [LICENSE](__LICENSE.md__) for details.

## Contact

- **Author**: Tsviki Y. Hirsh
- **Repository**: [https://github.com/TsvikiHirsh/G4LumaCam](https://github.com/TsvikiHirsh/G4LumaCam)
- **Issues**: [GitHub Issues](https://github.com/TsvikiHirsh/G4LumaCam/issues)

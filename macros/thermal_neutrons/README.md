# Thermal Neutron Bragg Edge Imaging Configuration

This directory contains macro configurations for simulating thermal neutron Bragg edge imaging of crystalline Fe (iron) samples using NCrystal.

## Macros

### 1. `fe_bragg_edge.mac`
Polychromatic thermal neutron beam covering the wavelength range from 0.5 Å to 8 Å (energy range: ~1.28 meV to ~327 meV). This configuration is ideal for observing Bragg edge patterns in the transmitted neutron spectrum.

**Key Parameters:**
- Sample: Fe gamma phase (BCC, space group 229) via NCrystal
- Sample thickness: 20 mm
- Scintillator: GS20 (optimized for thermal neutron detection)
- Energy range: 0.00128 eV to 0.327 eV (linear distribution)
- Events: 50,000 neutrons

### 2. `fe_monochromatic_05A.mac`
Monochromatic neutron beam at λ = 0.5 Å (E ≈ 0.327 eV).

**Key Parameters:**
- Sample: Fe gamma phase via NCrystal
- Sample thickness: 20 mm
- Scintillator: GS20
- Energy: 0.327 eV (monochromatic)
- Events: 50,000 neutrons

### 3. `fe_monochromatic_8A.mac`
Monochromatic neutron beam at λ = 8 Å (E ≈ 0.00128 eV).

**Key Parameters:**
- Sample: Fe gamma phase via NCrystal
- Sample thickness: 20 mm
- Scintillator: GS20
- Energy: 0.00128 eV (monochromatic)
- Events: 50,000 neutrons

## Running the Simulations

To run these macros, make sure you have NCrystal-Geant4 installed:

```bash
pip install ncrystal-geant4
```

Then rebuild G4LumaCam with NCrystal support:

```bash
cd /path/to/G4LumaCam/build
cmake ../src/G4LumaCam
make
```

Run the simulation:

```bash
cd /path/to/G4LumaCam/build/lib
./lumacam ../../macros/thermal_neutrons/fe_bragg_edge.mac
```

## Understanding Bragg Edges

Bragg edges occur when the neutron wavelength matches the lattice spacing in the crystalline material:

λ = 2d sinθ

For Fe (BCC structure):
- Major Bragg edges occur at specific wavelengths corresponding to different crystal planes
- The (110) reflection has the largest d-spacing and produces the first Bragg edge
- Observing transmission vs. wavelength reveals these characteristic edges

## Output

The simulation produces CSV files containing:
- Optical photon positions and energies at the scintillator
- Parent particle information (neutron interactions)
- Timing information for time-of-flight analysis
- Pulse structure data

Analyze the output to reconstruct the transmitted neutron spectrum and identify Bragg edges.

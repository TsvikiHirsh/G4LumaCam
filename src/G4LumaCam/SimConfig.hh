#ifndef SIM_CONFIG_HH
#define SIM_CONFIG_HH

#include "G4SystemOfUnits.hh"
#include "G4String.hh"
#include <random>
#include <vector>

namespace Sim {
    extern G4String outputFileName;
    extern G4int batchSize;
    extern std::default_random_engine randomEngine;
    extern G4double WORLD_SIZE;
    extern G4double SCINT_THICKNESS;
    extern G4double SAMPLE_THICKNESS;
    extern G4double SCINT_SIZE;
    extern G4double SAMPLE_WIDTH; // Full width
    extern G4double COATING_THICKNESS;
    static G4String sampleMaterial = "G4_GRAPHITE";
    static G4String scintillatorMaterial = "ScintillatorPVT";
    extern G4double TMIN;
    extern G4double TMAX;
    extern G4double FLUX; // Neutron flux in n/cm²/s
    extern G4double FREQ; // Pulse frequency in Hz
    extern std::vector<G4double> pulseTimes; // Trigger times for pulses in ns
    extern std::vector<G4int> neutronsPerPulse; // Neutrons per pulse

    void SetScintThickness(G4double thickness);
    void SetSampleThickness(G4double thickness);
    void SetSampleWidth(G4double width);
    void ComputePulseStructure(G4int totalNeutrons); // Compute pulse times and neutrons per pulse
}

#endif
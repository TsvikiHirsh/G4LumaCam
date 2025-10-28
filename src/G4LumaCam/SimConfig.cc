#include "SimConfig.hh"
#include <ctime>
#include "G4ios.hh"
#include <cmath>
#include <filesystem>
#include "Randomize.hh"

namespace Sim {
    G4String outputFileName = "sim_data.csv";
    G4int batchSize = 0;
    std::default_random_engine randomEngine(time(nullptr));
    G4double WORLD_SIZE = 50.0 * m;
    G4double SCINT_THICKNESS = 2.0 * cm;
    G4double SAMPLE_THICKNESS = 7.5 * cm;
    G4double SCINT_SIZE = 12.0 * cm;
    G4double SAMPLE_WIDTH = 12.0 * cm; // Full width
    G4double COATING_THICKNESS = 0.01 * cm;
    G4double TMIN = 0.0 * ns;
    G4double TMAX = 0.0 * ns;
    G4double FLUX = 0.0; // Default: no pulsed structure
    G4double FREQ = 0.0; // Default: no pulsed structure
    std::vector<G4double> pulseTimes; // Trigger times in ns
    std::vector<G4int> neutronsPerPulse; // Neutrons per pulse

    void SetScintThickness(G4double thickness) {
        if (thickness > 0) {
            SCINT_THICKNESS = thickness;
            G4cout << "Scintillator thickness set to: " << thickness / cm << " cm" << G4endl;
        } else {
            G4cerr << "ERROR: Scintillator thickness must be positive!" << G4endl;
        }
    }

    void SetSampleThickness(G4double thickness) {
        if (thickness > 0) {
            SAMPLE_THICKNESS = thickness;
            G4cout << "Sample thickness set to: " << thickness / cm << " cm" << G4endl;
        } else {
            G4cerr << "ERROR: Sample thickness must be positive!" << G4endl;
        }
    }

    void SetSampleWidth(G4double width) {
        if (width > 0) {
            SAMPLE_WIDTH = width;
            G4cout << "Sample width set to: " << width / cm << " cm" << G4endl;
        } else {
            G4cerr << "ERROR: Sample width must be positive!" << G4endl;
        }
    }

    void ComputePulseStructure(G4int totalNeutrons) {
        pulseTimes.clear();
        neutronsPerPulse.clear();
        
        if (FLUX <= 0 || FREQ <= 0 || totalNeutrons <= 0) {
            G4cout << "Pulse structure not computed: FLUX=" << FLUX << ", FREQ=" << FREQ
                   << ", totalNeutrons=" << totalNeutrons << G4endl;
            return;
        }
        
        // Calculate FOV area (cm²)
        G4double fovWidthCm = SCINT_SIZE / cm;  // Convert to cm
        G4double fovArea = fovWidthCm * fovWidthCm;  // Area in cm²
        
        G4cout << "\n=== FOV Calculation ===" << G4endl;
        G4cout << "SCINT_SIZE = " << SCINT_SIZE / cm << " cm" << G4endl;
        G4cout << "FOV width = " << fovWidthCm << " cm" << G4endl;
        G4cout << "FOV area = " << fovArea << " cm²" << G4endl;
        G4cout << "======================" << G4endl;
        
        G4double neutronsPerSecond = FLUX * fovArea; // Total neutrons/s
        G4double pulsePeriod = 1.0 / FREQ; // Seconds per pulse
        G4double avgNeutronsPerPulse = neutronsPerSecond / FREQ; // Neutrons per pulse
        G4int numPulses = std::ceil(totalNeutrons / avgNeutronsPerPulse);
        
        G4cout << "\n=== Computing Pulse Structure ===" << G4endl;
        G4cout << "FOV area: " << fovArea << " cm²" << G4endl;
        G4cout << "Flux: " << FLUX << " n/cm²/s" << G4endl;
        G4cout << "Frequency: " << FREQ << " Hz" << G4endl;
        G4cout << "Total neutrons/s: " << neutronsPerSecond << G4endl;
        G4cout << "Pulse period: " << pulsePeriod * 1e6 << " us" << G4endl;
        G4cout << "Avg neutrons/pulse: " << avgNeutronsPerPulse << G4endl;
        G4cout << "Total neutrons requested: " << totalNeutrons << G4endl;
        G4cout << "Number of pulses needed: " << numPulses << G4endl;
        
        G4int remainingNeutrons = totalNeutrons;
        G4int totalAssigned = 0;
        
        // Handle low neutron rates (< 1 neutron per pulse on average)
        if (avgNeutronsPerPulse < 1.0) {
            G4cout << "Low neutron rate detected: distributing neutrons across pulses" << G4endl;
            
            G4double pulsesPerNeutron = 1.0 / avgNeutronsPerPulse;
            
            for (G4int n = 0; n < totalNeutrons; ++n) {
                G4int pulseIndex = std::floor(n * pulsesPerNeutron);
                G4double pulseTime = pulseIndex * pulsePeriod * 1e9; // Seconds to ns
                
                pulseTimes.push_back(pulseTime);
                neutronsPerPulse.push_back(1);
                totalAssigned++;
                
                if (n < 5) {
                    G4cout << "Neutron " << n << " -> Pulse " << pulseIndex 
                           << ": t=" << pulseTime << " ns" << G4endl;
                }
            }
        } else {
            // Original logic for >= 1 neutron per pulse
            for (G4int i = 0; i < numPulses && remainingNeutrons > 0; ++i) {
                G4double rand = G4UniformRand();
                G4int neutronsThisPulse = std::floor(avgNeutronsPerPulse + rand);
                
                if (neutronsThisPulse > remainingNeutrons) {
                    neutronsThisPulse = remainingNeutrons;
                }
                if (neutronsThisPulse < 0) neutronsThisPulse = 0;
                
                neutronsPerPulse.push_back(neutronsThisPulse);
                pulseTimes.push_back(i * pulsePeriod * 1e9); // Seconds to ns
                
                totalAssigned += neutronsThisPulse;
                remainingNeutrons -= neutronsThisPulse;
                
                if (i < 5) {
                    G4cout << "Pulse " << i << ": t=" << (i * pulsePeriod * 1e6) 
                           << " us (" << pulseTimes.back() << " ns), n=" 
                           << neutronsThisPulse << G4endl;
                }
            }
        }
        
        G4cout << "\nPulse structure computed:" << G4endl;
        G4cout << "  Total pulses: " << pulseTimes.size() << G4endl;
        G4cout << "  Total neutrons assigned: " << totalAssigned << G4endl;
        G4cout << "  Remaining: " << remainingNeutrons << G4endl;
        
        if (totalAssigned != totalNeutrons) {
            G4cerr << "WARNING: Neutron count mismatch! Expected " << totalNeutrons 
                   << " but assigned " << totalAssigned << G4endl;
        }
        G4cout << "=================================\n" << G4endl;
    }
} // namespace Sim
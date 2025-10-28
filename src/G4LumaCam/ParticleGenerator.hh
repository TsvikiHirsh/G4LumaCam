#ifndef PARTICLE_GENERATOR_HH
#define PARTICLE_GENERATOR_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4GeneralParticleSource.hh"

class ParticleGenerator : public G4VUserPrimaryGeneratorAction {
public:
    ParticleGenerator();
    ~ParticleGenerator() override;
    
    void GeneratePrimaries(G4Event* anEvent) override;
    G4double getParticleEnergy() const { return lastEnergy; }
    void SetTotalNeutrons(G4int totalNeutrons);
    G4int getCurrentPulseIndex() const { return currentPulseIndex; } 

private:
    G4GeneralParticleSource* source;
    G4double lastEnergy;
    G4int currentPulseIndex;
    G4int neutronsInCurrentPulse;
};

#endif
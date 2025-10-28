#ifndef EVENT_PROCESSOR_HH
#define EVENT_PROCESSOR_HH
#include "G4VSensitiveDetector.hh"
#include "G4SystemOfUnits.hh"
#include <vector>
#include <map>
#include <fstream>

class ParticleGenerator;

class EventProcessor : public G4VSensitiveDetector {
public:
    EventProcessor(const G4String& name, ParticleGenerator* gen = nullptr);
    ~EventProcessor() override;
    void Initialize(G4HCofThisEvent*) override;
    G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
    void EndOfEvent(G4HCofThisEvent*) override;

private:
    struct PhotonRecord {
        G4int id, parentId, neutronId;
        G4double x, y, z, dx, dy, dz;  // Position and direction at monitor
        G4double x0, y0, z0, dx0, dy0, dz0;  // Position and direction at generation
        G4double timeOfArrival;
        G4double wavelength, parentEnergy, neutronEnergy;
        G4String parentType;
        G4double px, py, pz, nx, ny, nz;
        G4int pulseId;
        G4double pulseTime;
    };

    struct TrackData {
        G4String type;
        G4double x, y, z, energy;
        G4bool isLightProducer;
        // Store generation info for optical photons
        G4double x0, y0, z0, dx0, dy0, dz0;
    };

    std::vector<PhotonRecord> photons;
    std::map<G4int, TrackData> tracks;
    G4double neutronPos[3], neutronEnergy, protonEnergy;
    G4double lensPos[2];
    G4int neutronCount, batchCount, eventCount;
    std::ofstream dataFile;
    ParticleGenerator* particleGen;
    G4bool neutronRecorded;
    G4double currentEventTriggerTime;

    void resetData();
    void writeData();
    void openOutputFile();
};
#endif
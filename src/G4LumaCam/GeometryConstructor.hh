#ifndef GEOMETRY_CONSTRUCTOR_HH
#define GEOMETRY_CONSTRUCTOR_HH

#include "G4VUserDetectorConstruction.hh"
#include "MaterialBuilder.hh"
#include "LumaCamMessenger.hh"
#include "G4LogicalVolume.hh"
#include "SimConfig.hh"
#include "G4GenericMessenger.hh"
#include "G4SDManager.hh"
#include "G4ios.hh" // Added for G4cout/G4cerr

class EventProcessor;
class ParticleGenerator;
class LumaCamMessenger;

class GeometryConstructor : public G4VUserDetectorConstruction {
public:
    GeometryConstructor(ParticleGenerator* gen = nullptr);
    ~GeometryConstructor() override;
    G4VPhysicalVolume* Construct() override;
    G4LogicalVolume* getScintillatorLog() const { return scintLog; }

private:
    MaterialBuilder* matBuilder;
    EventProcessor* eventProc;
    G4LogicalVolume* sampleLog;
    G4LogicalVolume* scintLog;
    LumaCamMessenger* lumaCamMessenger;

    G4VPhysicalVolume* createWorld();
    G4LogicalVolume* buildLShape(G4LogicalVolume* worldLog);
    void addComponents(G4LogicalVolume* lShapeLog);
};

#endif
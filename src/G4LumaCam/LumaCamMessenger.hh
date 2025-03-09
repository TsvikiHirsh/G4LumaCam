#ifndef LUMACAM_MESSENGER_HH
#define LUMACAM_MESSENGER_HH
#include "G4GenericMessenger.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4ios.hh"

class MaterialBuilder; // Forward declaration

class LumaCamMessenger {
public:
    LumaCamMessenger(G4String* filename = nullptr, 
                     G4LogicalVolume* sampleLogVolume = nullptr, 
                     G4LogicalVolume* scintLogVolume = nullptr, // New parameter for scintillator volume
                     G4int batch = 10000);
    ~LumaCamMessenger();
    void SetMaterial(const G4String& materialName);
    void SetScintillator(const G4String& scintCode); // New method for scintillator selection

private:
    G4String* csvFilename;
    G4LogicalVolume* sampleLog;
    G4LogicalVolume* scintLog; // New member for scintillator logical volume
    G4int batchSize;
    G4GenericMessenger* messenger;
    MaterialBuilder* matBuilder; // To access SSLG4 scintillators
};
#endif
#include "LumaCamMessenger.hh"
#include "MaterialBuilder.hh"

LumaCamMessenger::LumaCamMessenger(G4String* filename, G4LogicalVolume* sampleLogVolume, 
                                   G4LogicalVolume* scintLogVolume, G4int batch)
    : csvFilename(filename), sampleLog(sampleLogVolume), scintLog(scintLogVolume), 
      batchSize(batch), scintillatorCode("PVT") {
    matBuilder = new MaterialBuilder();
    messenger = new G4GenericMessenger(this, "/lumacam/", "lumacam control commands");

    if (csvFilename) {
        messenger->DeclareProperty("csvFilename", *csvFilename)
            .SetGuidance("Set the CSV filename")
            .SetParameterName("filename", false)
            .SetDefaultValue("sim_data.csv");
    }

    if (sampleLog) {
        messenger->DeclareMethod("sampleMaterial", &LumaCamMessenger::SetMaterial)
            .SetGuidance("Set the material of the sample_log")
            .SetParameterName("material", false)
            .SetDefaultValue("G4_GRAPHITE");
    }

    if (scintLog) {
        messenger->DeclareMethod("scintillator", &LumaCamMessenger::SetScintillator)
            .SetGuidance("Set the scintillator material (e.g., OPSC-100, ISC-1000)")
            .SetParameterName("scintCode", false)
            .SetDefaultValue("PVT");
    }

    messenger->DeclareProperty("batchSize", batchSize)
        .SetGuidance("Set the number of events per CSV file (0 for single file)")
        .SetParameterName("size", false)
        .SetDefaultValue("10000");
}

LumaCamMessenger::~LumaCamMessenger() {
    delete messenger;
    delete matBuilder;
}

void LumaCamMessenger::SetMaterial(const G4String& materialName) {
    if (!sampleLog) return;
    G4NistManager* nistManager = G4NistManager::Instance();
    G4Material* material = nistManager->FindOrBuildMaterial(materialName);
    if (material) {
        sampleLog->SetMaterial(material);
        G4cout << "Sample material set to: " << materialName << G4endl;
    } else {
        G4cerr << "Material " << materialName << " not found!" << G4endl;
    }
}

void LumaCamMessenger::SetScintillator(const G4String& scintCode) {
    if (!scintLog) return;
    scintillatorCode = scintCode;
    G4Material* scintillator = nullptr;
    if (scintCode == "PVT") {
        scintillator = matBuilder->getPVT();
    } else {
        scintillator = matBuilder->getScintillator(scintCode, false); // MPT off initially
    }
    if (scintillator) {
        scintLog->SetMaterial(scintillator);
        G4cout << "Scintillator material set to: " << scintCode << " (MPT will be configured post-initialization)" << G4endl;
    } else {
        G4cerr << "Scintillator " << scintCode << " not found!" << G4endl;
    }
}
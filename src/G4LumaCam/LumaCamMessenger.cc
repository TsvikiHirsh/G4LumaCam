#include "LumaCamMessenger.hh"
#include "GeometryConstructor.hh"
#include "SimConfig.hh"
#include "G4RunManager.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"

LumaCamMessenger::LumaCamMessenger(G4String* filename, G4LogicalVolume* sampleLogVolume, 
                                   G4LogicalVolume* scintLogVolume, G4int batch)
    : csvFilename(filename), sampleLog(sampleLogVolume), scintLog(scintLogVolume),
      batchSize(batch), matBuilder(new MaterialBuilder()) {
    
    Sim::batchSize = batchSize; // Initialize Sim::batchSize
    G4cout << "LumaCamMessenger: Initializing with csvFilename=" 
           << (csvFilename ? *csvFilename : "null")
           << ", sampleLog=" << (sampleLog ? sampleLog->GetName() : "null")
           << ", scintLog=" << (scintLog ? scintLog->GetName() : "null")
           << ", batchSize=" << batchSize << G4endl;

    messenger = new G4GenericMessenger(this, "/lumacam/", "LumaCam control commands");

    // CSV filename
    if (csvFilename) {
        messenger->DeclareProperty("csvFilename", *csvFilename)
            .SetGuidance("Set the CSV filename")
            .SetParameterName("filename", false)
            .SetDefaultValue("sim_data.csv");
    }

    // Sample material
    messenger->DeclareMethod("sampleMaterial", &LumaCamMessenger::SetMaterial)
        .SetGuidance("Set the material of the sample_log")
        .SetParameterName("material", false)
        .SetDefaultValue("G4_GRAPHITE");

    // Scintillator material
    messenger->DeclareMethod("scintMaterial", &LumaCamMessenger::SetScintillatorMaterial)
        .SetGuidance("Set the scintillator material (EJ200, GS20, or LYSO)")
        .SetParameterName("material", false)
        .SetDefaultValue("EJ200");

    // Scintillator thickness
    messenger->DeclareMethod("scintThickness", &LumaCamMessenger::SetScintThickness)
        .SetGuidance("Set the scintillator thickness in cm")
        .SetParameterName("thickness", false)
        .SetDefaultValue("1.0");

    // Sample thickness
    messenger->DeclareMethod("sampleThickness", &LumaCamMessenger::SetSampleThickness)
        .SetGuidance("Set the sample thickness in cm")
        .SetParameterName("thickness", false)
        .SetDefaultValue("3.75");

    // Sample width
    messenger->DeclareMethod("sampleWidth", &LumaCamMessenger::SetSampleWidth)
        .SetGuidance("Set the sample half-width in cm (for knife edge from left to right)")
        .SetParameterName("width", false)
        .SetDefaultValue("12.0");

    // Batch size
    messenger->DeclareMethod("batchSize", &LumaCamMessenger::SetBatchSize)
        .SetGuidance("Set the number of events per CSV file (0 for single file)")
        .SetParameterName("size", false)
        .SetDefaultValue("10000");

    // Minimum emission time
    messenger->DeclarePropertyWithUnit("tmin", "ns", Sim::TMIN)
        .SetGuidance("Set minimum emission time (ns)")
        .SetParameterName("tmin", false)
        .SetDefaultValue("0.0");

    // Maximum emission time
    messenger->DeclarePropertyWithUnit("tmax", "ns", Sim::TMAX)
        .SetGuidance("Set maximum emission time (ns). If > tmin, uniform distribution is applied.")
        .SetParameterName("tmax", false)
        .SetDefaultValue("0.0");

    // Neutron flux
    messenger->DeclareMethod("flux", &LumaCamMessenger::SetFlux)
        .SetGuidance("Set neutron flux in n/cm²/s")
        .SetParameterName("flux", false)
        .SetDefaultValue("0.0");

    // Pulse frequency
    messenger->DeclareMethod("freq", &LumaCamMessenger::SetFrequency)
        .SetGuidance("Set pulse frequency in Hz")
        .SetParameterName("freq", false)
        .SetDefaultValue("0.0");
}

LumaCamMessenger::~LumaCamMessenger() {
    delete messenger;
    delete matBuilder;
}

void LumaCamMessenger::SetBatchSize(G4int size) {
    if (size < 0) {
        G4cerr << "ERROR: Batch size must be non-negative!" << G4endl;
        return;
    }
    batchSize = size;
    Sim::batchSize = size; // Synchronize with Sim::batchSize
    G4cout << "LumaCamMessenger: Batch size set to " << size << G4endl;
}

void LumaCamMessenger::SetSampleLog(G4LogicalVolume* log) {
    sampleLog = log;
    if (sampleLog) {
        G4cout << "LumaCamMessenger: sampleLog set to " << sampleLog->GetName() << G4endl;
    }
}

void LumaCamMessenger::SetScintLog(G4LogicalVolume* log) {
    scintLog = log;
    if (scintLog) {
        G4cout << "LumaCamMessenger: scintLog set to " << scintLog->GetName() << G4endl;
    }
}

void LumaCamMessenger::SetMaterial(const G4String& materialName) {
    if (!sampleLog) {
        G4cerr << "ERROR: sampleLog is nullptr, cannot set material to " << materialName << G4endl;
        return;
    }
    
    G4NistManager* nistManager = G4NistManager::Instance();
    G4Material* material = nistManager->FindOrBuildMaterial(materialName);
    
    if (material) {
        G4cout << "Current sample material: " 
               << sampleLog->GetMaterial()->GetName() << G4endl;
        Sim::sampleMaterial = materialName;
        GeometryConstructor* geom = dynamic_cast<GeometryConstructor*>(
            const_cast<G4VUserDetectorConstruction*>(
                G4RunManager::GetRunManager()->GetUserDetectorConstruction()));
        if (geom) {
            geom->UpdateSampleGeometry(Sim::SAMPLE_THICKNESS, material);
            G4RunManager::GetRunManager()->GeometryHasBeenModified();
            G4cout << "Sample material set to: " << materialName 
                   << ", Confirmed material: " 
                   << sampleLog->GetMaterial()->GetName() << G4endl;
        } else {
            G4cerr << "ERROR: Failed to cast to GeometryConstructor!" << G4endl;
        }
    } else {
        G4cerr << "Material " << materialName << " not found!" << G4endl;
        G4cout << "Available NIST materials:" << G4endl;
        const std::vector<G4String>& materialNames = nistManager->GetNistMaterialNames();
        for (const auto& name : materialNames) {
            G4cout << name << G4endl;
        }
    }
}

void LumaCamMessenger::SetScintillatorMaterial(const G4String& materialName) {
    if (!scintLog) {
        G4cerr << "ERROR: scintLog is nullptr, cannot set scintillator material to " << materialName << G4endl;
        return;
    }
    
    G4cout << "Setting scintillator material to: " << materialName << G4endl;
    matBuilder->setScintillatorType(materialName);
    G4Material* material = matBuilder->getScintillator();
    
    if (material) {
        G4cout << "Current scintillator material: " 
               << scintLog->GetMaterial()->GetName() << G4endl;
        Sim::scintillatorMaterial = materialName;
        scintLog->SetMaterial(material);
        G4RunManager::GetRunManager()->GeometryHasBeenModified();
        G4cout << "Scintillator material set to: " << materialName 
               << ", Confirmed material: " 
               << scintLog->GetMaterial()->GetName() << G4endl;
    } else {
        G4cerr << "ERROR: Scintillator material " << materialName << " not found!" << G4endl;
        G4cout << "Available scintillator materials: EJ200, GS20, LYSO" << G4endl;
    }
}

void LumaCamMessenger::SetScintThickness(G4double thickness) {
    if (thickness <= 0) {
        G4cerr << "ERROR: Scintillator thickness must be positive!" << G4endl;
        return;
    }
    G4cout << "Setting scintillator thickness to: " << thickness << " cm" << G4endl;
    Sim::SetScintThickness(thickness * cm);
    GeometryConstructor* geom = dynamic_cast<GeometryConstructor*>(
        const_cast<G4VUserDetectorConstruction*>(
            G4RunManager::GetRunManager()->GetUserDetectorConstruction()));
    if (geom && scintLog) {
        geom->UpdateScintillatorGeometry(Sim::SCINT_THICKNESS);
        G4RunManager::GetRunManager()->GeometryHasBeenModified();
    } else {
        G4cerr << "ERROR: Failed to cast to GeometryConstructor or scintLog is nullptr!" << G4endl;
    }
}

void LumaCamMessenger::SetSampleThickness(G4double thickness) {
    if (thickness <= 0) {
        G4cerr << "ERROR: Sample thickness must be positive!" << G4endl;
        return;
    }
    G4cout << "Setting sample thickness to: " << thickness << " cm" << G4endl;
    Sim::SetSampleThickness(thickness * cm);
    GeometryConstructor* geom = dynamic_cast<GeometryConstructor*>(
        const_cast<G4VUserDetectorConstruction*>(
            G4RunManager::GetRunManager()->GetUserDetectorConstruction()));
    if (geom && sampleLog) {
        G4Material* material = sampleLog->GetMaterial();
        geom->UpdateSampleGeometry(Sim::SAMPLE_THICKNESS, material);
        G4RunManager::GetRunManager()->GeometryHasBeenModified();
    } else {
        G4cerr << "ERROR: Failed to cast to GeometryConstructor or sampleLog is nullptr!" << G4endl;
    }
}

void LumaCamMessenger::SetSampleWidth(G4double width) {
    if (width <= 0) {
        G4cerr << "ERROR: Sample width must be positive!" << G4endl;
        return;
    }
    G4cout << "Setting sample width to: " << width << " cm" << G4endl;
    Sim::SetSampleWidth(width * cm);
    GeometryConstructor* geom = dynamic_cast<GeometryConstructor*>(
        const_cast<G4VUserDetectorConstruction*>(
            G4RunManager::GetRunManager()->GetUserDetectorConstruction()));
    if (geom && sampleLog) {
        G4Material* material = sampleLog->GetMaterial();
        geom->UpdateSampleGeometry(Sim::SAMPLE_THICKNESS, material, Sim::SAMPLE_WIDTH);
        G4RunManager::GetRunManager()->GeometryHasBeenModified();
    } else {
        G4cerr << "ERROR: Failed to cast to GeometryConstructor or sampleLog is nullptr!" << G4endl;
    }
}

void LumaCamMessenger::SetFlux(G4double flux) {
    if (flux < 0) {
        G4cerr << "ERROR: Neutron flux must be non-negative!" << G4endl;
        return;
    }
    Sim::FLUX = flux;
    G4cout << "Neutron flux set to: " << flux << " n/cm²/s" << G4endl;
}

void LumaCamMessenger::SetFrequency(G4double freq) {
    if (freq < 0) {
        G4cerr << "ERROR: Pulse frequency must be non-negative!" << G4endl;
        return;
    }
    Sim::FREQ = freq;
    G4cout << "Pulse frequency set to: " << freq / 1000 << " kHz" << G4endl;
}
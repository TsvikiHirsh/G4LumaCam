#include "MaterialBuilder.hh"
#include "G4Element.hh"
#include "G4SystemOfUnits.hh"
#include <vector>

MaterialBuilder::MaterialBuilder() {
    G4NistManager* nist = G4NistManager::Instance();
    
    // Elements
    G4Element* H = new G4Element("H", "H", 1., 1.01 * g/mole);
    G4Element* C = new G4Element("C", "C", 6., 12.01 * g/mole);
    G4Element* N = new G4Element("N", "N", 7., 14.01 * g/mole);
    G4Element* O = new G4Element("O", "O", 8., 16.00 * g/mole);
    G4Element* Si = new G4Element("Si", "Si", 14., 28.09 * g/mole);
    
    // Vacuum, Air, Graphite, Quartz, Absorber
    vacuum = new G4Material("Vacuum", 1., 1.01 * g/mole, CLHEP::universe_mean_density,
        kStateGas, 2.73 * kelvin, 3.e-18 * pascal);
    
    air = new G4Material("Air", 1.290 * mg/cm3, 2);
    air->AddElement(N, 70 * perCent);
    air->AddElement(O, 30 * perCent);
    G4double airEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double airRIndex[2] = {1.0, 1.0};
    G4double airAbs[2] = {100.0 * m, 100.0 * m};
    setupMaterialProperties(air, airEnergy, airRIndex, airAbs, 2);
    
    sampleMaterial = nist->FindOrBuildMaterial("G4_GRAPHITE");
    
    windowMaterial = new G4Material("Quartz", 2.20 * g/cm3, 2);
    windowMaterial->AddElement(Si, 1);
    windowMaterial->AddElement(O, 2);
    G4double quartzEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double quartzRIndex[2] = {1.59, 1.59};
    G4double quartzAbs[2] = {160.0 * cm, 160.0 * cm};
    setupMaterialProperties(windowMaterial, quartzEnergy, quartzRIndex, quartzAbs, 2);
    
    absorberMaterial = new G4Material("Absorber", 1.290 * mg/cm3, 2);
    absorberMaterial->AddElement(N, 70 * perCent);
    absorberMaterial->AddElement(O, 30 * perCent);
    G4double blackEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double blackRIndex[2] = {1.58, 1.58};
    G4double blackAbs[2] = {0.0 * mm, 0.0 * mm};
    setupMaterialProperties(absorberMaterial, blackEnergy, blackRIndex, blackAbs, 2);

    // Default scintillator (PVT) - unchanged
    scintMaterial = new G4Material("Scintillator", 1.023 * g/cm3, 2);
    scintMaterial->AddElement(C, 9);
    scintMaterial->AddElement(H, 10);
    const int nPVT = 12;
    G4double pvtEnergy[nPVT] = {2.08*eV, 2.38*eV, 2.58*eV, 2.7*eV, 2.76*eV,
        2.82*eV, 2.92*eV, 2.95*eV, 3.02*eV, 3.1*eV, 3.26*eV, 3.44*eV};
    G4double pvtScint[nPVT] = {0.00, 0.03, 0.17, 0.40, 0.55, 0.83,
        1.00, 0.84, 0.49, 0.20, 0.07, 0.04};
    G4double pvtRIndex[nPVT] = {1.58, 1.58, 1.58, 1.58, 1.58, 1.58,
        1.58, 1.58, 1.58, 1.58, 1.58, 1.58};
    G4double pvtAbs[nPVT] = {360.*cm, 360.*cm, 360.*cm, 360.*cm, 360.*cm,
        360.*cm, 360.*cm, 360.*cm, 360.*cm, 360.*cm, 360.*cm, 360.*cm};
    setupMaterialProperties(scintMaterial, pvtEnergy, pvtRIndex, pvtAbs, nPVT, pvtScint);
    scintMaterial->GetIonisation()->SetBirksConstant(0.126 * mm/MeV);
}

// New method to get SSLG4 scintillators
G4Material* MaterialBuilder::getScintillator(const G4String& scintCode) {
    G4bool isMPTOn = true; // Enable MaterialPropertiesTable
    G4Material* scintillator = nullptr;

    // Check if it's an organic scintillator (e.g., OPSC-100)
    if (scintCode.substr(0, 4) == "OPSC") {
        scintillator = OrganicScintillatorFactory::GetInstance()->Get(scintCode, isMPTOn);
    }
    // Check if it's an inorganic scintillator (e.g., ISC-1000)
    else if (scintCode.substr(0, 3) == "ISC") {
        scintillator = InorganicScintillatorFactory::GetInstance()->Get(scintCode, isMPTOn);
    }

    if (!scintillator) {
        G4cerr << "Scintillator " << scintCode << " not found in SSLG4 library!" << G4endl;
        return scintMaterial; // Fallback to default PVT
    }
    return scintillator;
}

// setupMaterialProperties remains unchanged
void MaterialBuilder::setupMaterialProperties(G4Material* mat, const G4double* energies,
    const G4double* rindex, const G4double* abslength, int nEntries, const G4double* scintillation) {
    std::vector<G4double> energiesCopy(energies, energies + nEntries);
    std::vector<G4double> rindexCopy(rindex, rindex + nEntries);
    std::vector<G4double> abslengthCopy(abslength, abslength + nEntries);
    
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", energiesCopy.data(), rindexCopy.data(), nEntries);
    mpt->AddProperty("ABSLENGTH", energiesCopy.data(), abslengthCopy.data(), nEntries);
    
    if (scintillation) {
        std::vector<G4double> scintillationCopy(scintillation, scintillation + nEntries);
        mpt->AddProperty("FASTCOMPONENT", energiesCopy.data(), scintillationCopy.data(), nEntries);
        mpt->AddConstProperty("SCINTILLATIONYIELD", 10000./MeV);
        mpt->AddConstProperty("RESOLUTIONSCALE", 1.0);
        mpt->AddConstProperty("FASTTIMECONSTANT", 3.2 * ns);
        mpt->AddConstProperty("YIELDRATIO", 0.);
    }
    
    mat->SetMaterialPropertiesTable(mpt);
}
#include "MaterialBuilder.hh"
#include "SimConfig.hh"
#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"
#include "G4Element.hh"
#include "G4Isotope.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "globals.hh"
#include <vector>

MaterialBuilder::MaterialBuilder() 
    : vacuum(nullptr), air(nullptr), scintMaterialPVT(nullptr), 
      scintMaterialGS20(nullptr), scintMaterialLYSO(nullptr), 
      scintMaterial(nullptr), sampleMaterial(nullptr), 
      windowMaterial(nullptr), absorberMaterial(nullptr), 
      currentScintType(ScintType::EJ200) {
    DefineMaterials();
    setScintillatorType(Sim::scintillatorMaterial); // Initialize with SimConfig default
}

void MaterialBuilder::DefineMaterials() {
    G4NistManager* nist = G4NistManager::Instance();
    G4cout << "MaterialBuilder: Initializing materials..." << G4endl;

    // Vacuum
    vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    if (!vacuum) {
        vacuum = new G4Material("Vacuum", 1., 1.01 * g/mole, CLHEP::universe_mean_density,
                                kStateGas, 2.73 * kelvin, 3.e-18 * pascal);
        G4cout << "MaterialBuilder: Vacuum material created" << G4endl;
    }

    // Air
    air = nist->FindOrBuildMaterial("G4_AIR");
    if (!air) {
        air = new G4Material("Air", 1.290 * mg/cm3, 2);
        air->AddElement(nist->FindOrBuildElement("N"), 70 * perCent);
        air->AddElement(nist->FindOrBuildElement("O"), 30 * perCent);
        G4cout << "MaterialBuilder: Air material created" << G4endl;
    }
    G4double airEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double airRIndex[2] = {1.0, 1.0};
    G4double airAbs[2] = {100.0 * m, 100.0 * m};
    setupMaterialProperties(air, airEnergy, airRIndex, airAbs, 2);

    // Graphite
    sampleMaterial = nist->FindOrBuildMaterial("G4_GRAPHITE");
    if (!sampleMaterial) {
        G4cerr << "ERROR: G4_GRAPHITE not found!" << G4endl;
    }

    // Quartz
    windowMaterial = nist->FindOrBuildMaterial("G4_SILICON_DIOXIDE");
    if (!windowMaterial) {
        windowMaterial = new G4Material("Quartz", 2.20 * g/cm3, 2);
        windowMaterial->AddElement(nist->FindOrBuildElement("Si"), 1);
        windowMaterial->AddElement(nist->FindOrBuildElement("O"), 2);
        G4cout << "MaterialBuilder: Quartz material created" << G4endl;
    }
    G4double quartzEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double quartzRIndex[2] = {1.59, 1.59};
    G4double quartzAbs[2] = {160.0 * cm, 160.0 * cm};
    setupMaterialProperties(windowMaterial, quartzEnergy, quartzRIndex, quartzAbs, 2);

    // Absorber
    absorberMaterial = nist->FindOrBuildMaterial("G4_C");
    if (!absorberMaterial) {
        absorberMaterial = new G4Material("Absorber", 1.290 * mg/cm3, 2);
        absorberMaterial->AddElement(nist->FindOrBuildElement("N"), 70 * perCent);
        absorberMaterial->AddElement(nist->FindOrBuildElement("O"), 30 * perCent);
        G4cout << "MaterialBuilder: Absorber material created" << G4endl;
    }
    G4double blackEnergy[2] = {1.0 * eV, 20.0 * eV};
    G4double blackRIndex[2] = {1.58, 1.58};
    G4double blackAbs[2] = {0.0 * mm, 0.0 * mm};
    setupMaterialProperties(absorberMaterial, blackEnergy, blackRIndex, blackAbs, 2);

    // Scintillator (PVT)
    scintMaterialPVT = new G4Material("ScintillatorPVT", 1.023 * g/cm3, 2);
    scintMaterialPVT->AddElement(nist->FindOrBuildElement("C"), 9);
    scintMaterialPVT->AddElement(nist->FindOrBuildElement("H"), 10);
    const int nPVT = 61;
    G4double pvtEnergy[nPVT] = {
        3.26*eV, 3.25*eV, 3.23*eV, 3.21*eV, 3.20*eV, 3.18*eV, 3.16*eV, 3.15*eV, 3.13*eV, 3.12*eV,
        3.10*eV, 3.08*eV, 3.07*eV, 3.05*eV, 3.04*eV, 3.02*eV, 3.01*eV, 2.99*eV, 2.98*eV, 2.96*eV,
        2.95*eV, 2.94*eV, 2.92*eV, 2.91*eV, 2.90*eV, 2.88*eV, 2.87*eV, 2.85*eV, 2.84*eV, 2.82*eV,
        2.81*eV, 2.80*eV, 2.79*eV, 2.77*eV, 2.76*eV, 2.75*eV, 2.74*eV, 2.73*eV, 2.72*eV, 2.70*eV,
        2.69*eV, 2.68*eV, 2.67*eV, 2.66*eV, 2.65*eV, 2.64*eV, 2.63*eV, 2.62*eV, 2.61*eV, 2.59*eV,
        2.58*eV, 2.57*eV, 2.56*eV, 2.55*eV, 2.54*eV, 2.52*eV, 2.51*eV, 2.50*eV, 2.49*eV, 2.48*eV,
        2.48*eV
    };
    G4double pvtScint[nPVT] = {
        0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.000, 0.002,
        0.006, 0.016, 0.044, 0.101, 0.173, 0.251, 0.348, 0.454, 0.596, 0.728,
        0.861, 0.949, 0.991, 0.999, 0.982, 0.956, 0.926, 0.890, 0.850, 0.804,
        0.755, 0.706, 0.658, 0.617, 0.582, 0.549, 0.519, 0.492, 0.468, 0.448,
        0.429, 0.410, 0.389, 0.364, 0.337, 0.306, 0.271, 0.238, 0.212, 0.191,
        0.171, 0.153, 0.137, 0.123, 0.109, 0.098, 0.087, 0.077, 0.068, 0.059,
        0.056
    };
    G4double pvtRIndex[nPVT];
    for (int i = 0; i < nPVT; i++) {
        pvtRIndex[i] = 1.58;
    }
    G4double pvtAbs[nPVT];
    for (int i = 0; i < nPVT; i++) {
        pvtAbs[i] = 380. * cm;
    }
    setupMaterialProperties(scintMaterialPVT, pvtEnergy, pvtRIndex, pvtAbs, nPVT, pvtScint);
    scintMaterialPVT->GetIonisation()->SetBirksConstant(0.126 * mm/MeV);

    // Scintillator (GS20)
    buildGS20();

    // Scintillator (LYSO)
    buildLYSO();

    // Set default scintillator
    scintMaterial = scintMaterialPVT;
    G4cout << "MaterialBuilder: Default scintillator set to ScintillatorPVT" << G4endl;
}

void MaterialBuilder::buildGS20() {
    G4NistManager* nist = G4NistManager::Instance();
    scintMaterialGS20 = new G4Material("ScintillatorGS20", 2.5 * g/cm3, 5);
    G4Element* enriched_li = new G4Element("Enriched_Lithium", "en_Li", 2);
    G4Isotope* li6 = new G4Isotope("6Li", 3, 6, 6.015 * g/mole);
    G4Isotope* li7 = new G4Isotope("7Li", 3, 7, 7.016 * g/mole);
    enriched_li->AddIsotope(li6, 95 * perCent);
    enriched_li->AddIsotope(li7, 5 * perCent);
    G4Material* enriched_li2o = new G4Material("Enriched_LITHIUM_OXIDE", 2.01 * g/cm3, 2);
    enriched_li2o->AddElement(enriched_li, 2);
    enriched_li2o->AddElement(nist->FindOrBuildElement("O"), 1);
    G4Material* ce2o3 = new G4Material("CERIUM_III_OXIDE", 6.2 * g/cm3, 2);
    ce2o3->AddElement(nist->FindOrBuildElement("Ce"), 2);
    ce2o3->AddElement(nist->FindOrBuildElement("O"), 3);
    scintMaterialGS20->AddMaterial(nist->FindOrBuildMaterial("G4_SILICON_DIOXIDE"), 57 * perCent);
    scintMaterialGS20->AddMaterial(nist->FindOrBuildMaterial("G4_ALUMINUM_OXIDE"), 18 * perCent);
    scintMaterialGS20->AddMaterial(nist->FindOrBuildMaterial("G4_MAGNESIUM_OXIDE"), 4 * perCent);
    scintMaterialGS20->AddMaterial(enriched_li2o, 17 * perCent);
    scintMaterialGS20->AddMaterial(ce2o3, 4 * perCent);
    const int nPoints = 117;
    G4double ScintEnergy[nPoints] = {
        1.771147508*eV, 1.785693715*eV, 1.806886491*eV, 1.835205916*eV, 1.859861711*eV, 1.887526235*eV,
        1.911215667*eV, 1.937982147*eV, 1.968038263*eV, 1.993817314*eV, 2.025656659*eV, 2.052965257*eV,
        2.086731237*eV, 2.11868072*eV, 2.14858044*eV, 2.176213997*eV, 2.214200693*eV, 2.243574408*eV,
        2.270347301*eV, 2.294304242*eV, 2.322309111*eV, 2.354650359*eV, 2.376721902*eV, 2.406788116*eV,
        2.433745705*eV, 2.469279938*eV, 2.493591512*eV, 2.522559239*eV, 2.556474115*eV, 2.586920261*eV,
        2.609157128*eV, 2.627217637*eV, 2.650155755*eV, 2.673476749*eV, 2.692441907*eV, 2.711710765*eV,
        2.726355401*eV, 2.746080968*eV, 2.761100218*eV, 2.77121984*eV, 2.796776839*eV, 2.817562006*eV,
        2.828136098*eV, 2.8547708*eV, 2.871018238*eV, 2.892938281*eV, 2.898532973*eV, 2.915258618*eV,
        2.926567493*eV, 2.937990045*eV, 2.955188453*eV, 2.966796809*eV, 2.978496723*eV, 2.990276024*eV,
        2.996253925*eV, 3.008174378*eV, 3.020203583*eV, 3.038367271*eV, 3.050653453*eV, 3.069200332*eV,
        3.081737652*eV, 3.100651336*eV, 3.139096176*eV, 3.191743043*eV, 3.225507311*eV, 3.239154774*eV,
        3.245982849*eV, 3.259772991*eV, 3.273696694*eV, 3.2806075*eV, 3.294726235*eV, 3.301758535*eV,
        3.308837152*eV, 3.323134953*eV, 3.330272736*eV, 3.337474277*eV, 3.344707031*eV, 3.351971202*eV,
        3.359216802*eV, 3.366443383*eV, 3.366544205*eV, 3.381295329*eV, 3.396090789*eV, 3.396227597*eV,
        3.411206068*eV, 3.418692972*eV, 3.426230217*eV, 3.441598022*eV, 3.449219077*eV, 3.464740871*eV,
        3.480313198*eV, 3.495989894*eV, 3.496098625*eV, 3.520093022*eV, 3.528102993*eV, 3.544381799*eV,
        3.560849122*eV, 3.569083637*eV, 3.60277172*eV, 3.654582691*eV, 3.735258252*eV, 3.829204446*eV,
        3.927975404*eV, 4.010757254*eV, 4.130449603*eV, 4.245659698*eV, 4.367453005*eV, 4.470036937*eV,
        4.577524753*eV, 4.763720755*eV, 4.886025126*eV, 5.06482313*eV, 5.293804937*eV, 5.464820789*eV,
        5.64730209*eV, 5.864795368*eV, 6.149039422*eV
    };
    G4double ScintFast[nPoints] = {
        0.004514673, 0.006772009, 0.006772009, 0.009029345, 0.006772009, 0.004514673, 
        0.002257336, 0.004514673, 0.002257336, 0.004514673, 0.006772009, 0.004514673, 
        0.004514673, 0.006772009, 0.006772009, 0.004514673, 0.006772009, 0.009029345, 
        0.011286682, 0.013544018, 0.015801354, 0.020316027, 0.0248307, 0.027088036, 
        0.033860045, 0.036117381, 0.047404063, 0.058690745, 0.065462754, 0.074492099, 
        0.090293454, 0.101580135, 0.11738149, 0.128668172, 0.139954853, 0.158013544, 
        0.173814898, 0.18510158, 0.200902935, 0.214446953, 0.23476298, 0.250564334, 
        0.270880361, 0.293453725, 0.311512415, 0.329571106, 0.34537246, 0.358916479, 
        0.376975169, 0.399548533, 0.415349887, 0.431151242, 0.446952596, 0.460496614, 
        0.476297968, 0.489841986, 0.505643341, 0.519187359, 0.53724605, 0.553047404, 
        0.571106095, 0.584650113, 0.598194131, 0.598194131, 0.591422122, 0.58013544, 
        0.568848758, 0.553047404, 0.539503386, 0.519187359, 0.507900677, 0.492099323, 
        0.478555305, 0.458239278, 0.440180587, 0.426636569, 0.413092551, 0.399548533, 
        0.379232506, 0.35214447, 0.365688488, 0.338600451, 0.300225734, 0.318284424, 
        0.286681716, 0.264108352, 0.243792325, 0.227990971, 0.205417607, 0.182844244, 
        0.148984199, 0.110609481, 0.124153499, 0.09255079, 0.074492099, 0.056433409, 
        0.042889391, 0.029345372, 0.018058691, 0.009029345, 0.006772009, 0.006772009, 
        0.004514673, 0.004514673, 0.004514673, 0.006772009, 0.006772009, 0.006772009, 
        0.004514673, 0.004514673, 0.004514673, 0.004514673, 0.006772009, 0.006772009, 
        0.009029345, 0.006772009, 0.006772009
    };
    G4double gs20RIndex[nPoints];
    for (int i = 0; i < nPoints; i++) {
        gs20RIndex[i] = 1.55;
    }
    G4double gs20Abs[nPoints];
    for (int i = 0; i < nPoints; i++) {
        gs20Abs[i] = 100 * cm;
    }
    setupMaterialProperties(scintMaterialGS20, ScintEnergy, gs20RIndex, gs20Abs, nPoints, ScintFast);
}

void MaterialBuilder::buildLYSO() {
    G4NistManager* nist = G4NistManager::Instance();
    scintMaterialLYSO = new G4Material("ScintillatorLYSO", 7.1 * g/cm3, 4);
    scintMaterialLYSO->AddElement(nist->FindOrBuildElement("Lu"), 71.45 * perCent);
    scintMaterialLYSO->AddElement(nist->FindOrBuildElement("Y"), 4.03 * perCent);
    scintMaterialLYSO->AddElement(nist->FindOrBuildElement("Si"), 6.37 * perCent);
    scintMaterialLYSO->AddElement(nist->FindOrBuildElement("O"), 18.15 * perCent);
    const int nLYSO = 77;
    G4double lysoEnergy[nLYSO] = {
        3.542*eV, 3.503*eV, 3.464*eV, 3.426*eV, 3.390*eV, 3.353*eV, 3.318*eV, 3.283*eV, 3.249*eV, 3.216*eV,
        3.183*eV, 3.151*eV, 3.120*eV, 3.089*eV, 3.059*eV, 3.030*eV, 3.001*eV, 2.972*eV, 2.945*eV, 2.917*eV,
        2.890*eV, 2.864*eV, 2.838*eV, 2.813*eV, 2.788*eV, 2.763*eV, 2.739*eV, 2.716*eV, 2.692*eV, 2.669*eV,
        2.647*eV, 2.625*eV, 2.603*eV, 2.582*eV, 2.561*eV, 2.540*eV, 2.519*eV, 2.499*eV, 2.480*eV, 2.460*eV,
        2.441*eV, 2.422*eV, 2.404*eV, 2.386*eV, 2.368*eV, 2.350*eV, 2.332*eV, 2.315*eV, 2.298*eV, 2.282*eV,
        2.265*eV, 2.249*eV, 2.233*eV, 2.217*eV, 2.202*eV, 2.186*eV, 2.171*eV, 2.156*eV, 2.141*eV, 2.127*eV,
        2.113*eV, 2.099*eV, 2.085*eV, 2.071*eV, 2.057*eV, 2.044*eV, 2.031*eV, 2.018*eV, 1.992*eV, 1.980*eV,
        1.967*eV, 1.955*eV, 1.943*eV, 1.931*eV, 1.919*eV, 1.907*eV, 1.907*eV
    };
    G4double lysoScint[nLYSO] = {
        0.000, 0.000, 0.000, 0.003, 0.006, 0.010, 0.025, 0.054, 0.089, 0.146,
        0.238, 0.295, 0.378, 0.492, 0.575, 0.679, 0.743, 0.819, 0.873, 0.921,
        0.959, 0.987, 0.997, 1.000, 0.994, 0.984, 0.968, 0.949, 0.921, 0.889,
        0.854, 0.819, 0.787, 0.749, 0.708, 0.676, 0.632, 0.590, 0.559, 0.530,
        0.505, 0.473, 0.444, 0.410, 0.378, 0.349, 0.321, 0.295, 0.270, 0.251,
        0.232, 0.213, 0.190, 0.168, 0.152, 0.140, 0.124, 0.111, 0.098, 0.089,
        0.076, 0.067, 0.063, 0.057, 0.048, 0.041, 0.032, 0.029, 0.025, 0.019,
        0.016, 0.013, 0.013, 0.010, 0.006, 0.006, 0.006
    };
    G4double lysoRIndex[nLYSO];
    for (int i = 0; i < nLYSO; i++) {
        lysoRIndex[i] = 1.81;
    }
    G4double lysoAbs[nLYSO];
    for (int i = 0; i < nLYSO; i++) {
        lysoAbs[i] = 41.3 * cm;
    }
    setupMaterialProperties(scintMaterialLYSO, lysoEnergy, lysoRIndex, lysoAbs, nLYSO, lysoScint);
    scintMaterialLYSO->GetIonisation()->SetBirksConstant(0.023 * mm/MeV);
}

void MaterialBuilder::setupMaterialProperties(G4Material* mat, const G4double* energies,
                                             const G4double* rindex, const G4double* abslength,
                                             int nEntries, const G4double* scintillation) {
    if (!mat) {
        G4cerr << "ERROR: Material is nullptr in setupMaterialProperties!" << G4endl;
        return;
    }
    std::vector<G4double> energiesCopy(energies, energies + nEntries);
    std::vector<G4double> rindexCopy(rindex, rindex + nEntries);
    std::vector<G4double> abslengthCopy(abslength, abslength + nEntries);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", energiesCopy.data(), rindexCopy.data(), nEntries);
    mpt->AddProperty("ABSLENGTH", energiesCopy.data(), abslengthCopy.data(), nEntries);
    if (scintillation) {
        std::vector<G4double> scintillationCopy(scintillation, scintillation + nEntries);
        mpt->AddProperty("FASTCOMPONENT", energiesCopy.data(), scintillationCopy.data(), nEntries);
        if (mat == scintMaterialPVT) {
            mpt->AddConstProperty("SCINTILLATIONYIELD", 10000./MeV);
            mpt->AddConstProperty("RESOLUTIONSCALE", 1.0);
            mpt->AddConstProperty("FASTTIMECONSTANT", 3.2 * ns);
            mpt->AddConstProperty("SCINTILLATIONRISETIME", 0.9 * ns);
            mpt->AddConstProperty("YIELDRATIO", 1.0);
        } else if (mat == scintMaterialGS20) {
            mpt->AddConstProperty("SCINTILLATIONYIELD", 1255.23 / MeV);
            mpt->AddConstProperty("RESOLUTIONSCALE", 1.0);
            mpt->AddConstProperty("FASTTIMECONSTANT", 57. * ns);
            mpt->AddConstProperty("SLOWTIMECONSTANT", 98. * ns);
            mpt->AddConstProperty("YIELDRATIO", 1.0);
        } else if (mat == scintMaterialLYSO) {
            mpt->AddConstProperty("SCINTILLATIONYIELD", 32000. / MeV);
            mpt->AddConstProperty("RESOLUTIONSCALE", 1.0);
            mpt->AddConstProperty("FASTTIMECONSTANT", 41. * ns);
            mpt->AddConstProperty("SCINTILLATIONRISETIME", 0.09 * ns);
            mpt->AddConstProperty("YIELDRATIO", 1.0);
        }
    }
    mat->SetMaterialPropertiesTable(mpt);
    G4cout << "MaterialBuilder: Material properties set for " << mat->GetName() << G4endl;
}

void MaterialBuilder::setScintillatorType(ScintType type) {
    currentScintType = type;
    if (type == ScintType::EJ200) {
        if (scintMaterialPVT) {
            scintMaterial = scintMaterialPVT;
            G4cout << "MaterialBuilder: Scintillator set to EJ200 (ScintillatorPVT)" << G4endl;
        } else {
            G4cerr << "ERROR: scintMaterialPVT is nullptr!" << G4endl;
        }
    } else if (type == ScintType::GS20) {
        if (scintMaterialGS20) {
            scintMaterial = scintMaterialGS20;
            G4cout << "MaterialBuilder: Scintillator set to GS20" << G4endl;
        } else {
            G4cerr << "ERROR: scintMaterialGS20 is nullptr!" << G4endl;
        }
    } else if (type == ScintType::LYSO) {
        if (scintMaterialLYSO) {
            scintMaterial = scintMaterialLYSO;
            G4cout << "MaterialBuilder: Scintillator set to LYSO" << G4endl;
        } else {
            G4cerr << "ERROR: scintMaterialLYSO is nullptr!" << G4endl;
        }
    }
}

void MaterialBuilder::setScintillatorType(const G4String& typeName) {
    G4cout << "MaterialBuilder: Attempting to set scintillator type to " << typeName << G4endl;
    if (typeName == "EJ200" || typeName == "ScintillatorPVT") {
        setScintillatorType(ScintType::EJ200);
    } else if (typeName == "GS20" || typeName == "ScintillatorGS20") {
        setScintillatorType(ScintType::GS20);
    } else if (typeName == "LYSO" || typeName == "ScintillatorLYSO") {
        setScintillatorType(ScintType::LYSO);
    } else {
        G4cerr << "ERROR: Unknown scintillator type: " << typeName << ". Available types: EJ200, GS20, LYSO, ScintillatorPVT, ScintillatorGS20, ScintillatorLYSO" << G4endl;
    }
}
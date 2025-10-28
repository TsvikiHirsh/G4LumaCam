#include "GeometryConstructor.hh"
#include "EventProcessor.hh"
#include "SimConfig.hh"
#include "G4Box.hh"
#include "G4UnionSolid.hh"
#include "G4PVPlacement.hh"
#include "G4VisAttributes.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4SystemOfUnits.hh"
#include "G4NistManager.hh"
#include "G4SDManager.hh"
#include "G4SubtractionSolid.hh"
#include "LumaCamMessenger.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4RunManager.hh"
#include "SimConfig.hh"

GeometryConstructor::GeometryConstructor(ParticleGenerator* gen) 
    : matBuilder(new MaterialBuilder()), eventProc(nullptr), sampleLog(nullptr), scintLog(nullptr), lumaCamMessenger(nullptr),
      blackSideLog(nullptr), blackBackLog(nullptr) {
    G4cout << "GeometryConstructor: Initializing..." << G4endl;
    matBuilder->DefineMaterials();
    eventProc = new EventProcessor("EventProcessor", gen);
    G4SDManager* sdManager = G4SDManager::GetSDMpointer();
    sdManager->AddNewDetector(eventProc);
    G4String filename = Sim::outputFileName; // Use SimConfig's outputFileName
    lumaCamMessenger = new LumaCamMessenger(&filename, nullptr, nullptr, Sim::batchSize);
}

GeometryConstructor::~GeometryConstructor() {
    G4cout << "GeometryConstructor: Cleaning up..." << G4endl;
    delete matBuilder;
    delete lumaCamMessenger;
    // delete eventProc; // Commented out to avoid double deletion
}

G4VPhysicalVolume* GeometryConstructor::Construct() {
    G4cout << "GeometryConstructor: Constructing geometry..." << G4endl;
    G4PhysicalVolumeStore* physVolStore = G4PhysicalVolumeStore::GetInstance();
    G4VPhysicalVolume* existingWorld = physVolStore->GetVolume("World", false);
    if (existingWorld) {
        G4cout << "GeometryConstructor: Reusing existing world volume" << G4endl;
        return existingWorld;
    }

    G4VPhysicalVolume* worldPhys = createWorld();
    G4LogicalVolume* worldLog = worldPhys->GetLogicalVolume();
    G4LogicalVolume* lShapeLog = buildLShape(worldLog);

    // Place sample in world volume
    G4NistManager* nist = G4NistManager::Instance();
    G4Material* sampleMat = nist->FindOrBuildMaterial(Sim::sampleMaterial);
    if (!sampleMat) {
        G4cerr << "ERROR: Sample material " << Sim::sampleMaterial << " not found! Defaulting to G4_GRAPHITE" << G4endl;
        sampleMat = nist->FindOrBuildMaterial("G4_GRAPHITE");
    }
    G4Box* sampleSolid = new G4Box("SampleSolid", Sim::SAMPLE_WIDTH/2, Sim::SCINT_SIZE/2, Sim::SAMPLE_THICKNESS/2);
    sampleLog = new G4LogicalVolume(sampleSolid, sampleMat, "SampleLog");
    if (!sampleLog) {
        G4cerr << "ERROR: sampleLog is nullptr!" << G4endl;
    } else {
        G4cout << "GeometryConstructor: Sample logical volume created" << G4endl;
    }
    G4VisAttributes* sampleVisAttributes = new G4VisAttributes(G4Colour(0.15, 0.2, 0.8, 0.5));
    sampleVisAttributes->SetForceSolid(true);
    sampleVisAttributes->SetVisibility(true);
    new G4PVPlacement(nullptr, G4ThreeVector(-Sim::SCINT_SIZE/2 + Sim::SAMPLE_WIDTH/2, 0, -Sim::SCINT_THICKNESS - Sim::COATING_THICKNESS - Sim::SAMPLE_THICKNESS/2), 
                      sampleLog, "SamplePhys", worldLog, false, 0, true);
    sampleLog->SetVisAttributes(sampleVisAttributes);

    // Set sampleLog in LumaCamMessenger
    if (lumaCamMessenger) {
        lumaCamMessenger->SetSampleLog(sampleLog);
        G4cout << "GeometryConstructor: Set sampleLog in LumaCamMessenger" << G4endl;
    }

    // Build other components (including scintillator and coating)
    addComponents(lShapeLog);

    // Set scintLog in LumaCamMessenger after addComponents
    if (lumaCamMessenger && scintLog) {
        lumaCamMessenger->SetScintLog(scintLog);
        G4cout << "GeometryConstructor: Set scintLog in LumaCamMessenger" << G4endl;
    }

    G4cout << "GeometryConstructor: LumaCamMessenger updated with sampleLog=" 
           << (sampleLog ? sampleLog->GetName() : "null") 
           << ", scintLog=" << (scintLog ? scintLog->GetName() : "null") << G4endl;

    return worldPhys;
}

void GeometryConstructor::UpdateScintillatorGeometry(G4double thickness) {
    G4cout << "GeometryConstructor: Updating scintillator geometry with thickness: " 
           << thickness/cm << " cm" << G4endl;
    
    G4PhysicalVolumeStore* physVolStore = G4PhysicalVolumeStore::GetInstance();

    // Update scintillator
    if (scintLog) {
        G4Box* scintSolid = dynamic_cast<G4Box*>(scintLog->GetSolid());
        if (scintSolid) {
            scintSolid->SetZHalfLength(thickness/2); // Update scintillator thickness
            G4cout << "GeometryConstructor: Scintillator solid updated" << G4endl;
        } else {
            G4cerr << "ERROR: scintLog has no valid G4Box solid!" << G4endl;
        }
    } else {
        G4cerr << "ERROR: scintLog is nullptr!" << G4endl;
    }

    G4VPhysicalVolume* scintPhys = physVolStore->GetVolume("ScintPhys", false);
    if (scintPhys) {
        scintPhys->SetTranslation(G4ThreeVector(0, 0, thickness/2));
        G4cout << "GeometryConstructor: Scintillator placement updated" << G4endl;
    } else {
        G4cerr << "ERROR: ScintPhys not found in volume store!" << G4endl;
    }

    // Update coating (black tape side and back boxes)
    if (blackSideLog) {
        G4Box* blackSideSolid = dynamic_cast<G4Box*>(blackSideLog->GetSolid());
        if (blackSideSolid) {
            blackSideSolid->SetZHalfLength(thickness/2); // Update side coating thickness
            G4cout << "GeometryConstructor: Black tape side solid updated" << G4endl;
        } else {
            G4cerr << "ERROR: blackSideLog has no valid G4Box solid!" << G4endl;
        }
    } else {
        G4cerr << "ERROR: blackSideLog is nullptr!" << G4endl;
    }

    // Update side coating placements
    G4VPhysicalVolume* blackSideTop = physVolStore->GetVolume("black_side_top", false);
    G4VPhysicalVolume* blackSideBottom = physVolStore->GetVolume("black_side_bottom", false);
    G4VPhysicalVolume* blackSideLeft = physVolStore->GetVolume("black_side_left", false);
    G4VPhysicalVolume* blackSideRight = physVolStore->GetVolume("black_side_right", false);
    if (blackSideTop && blackSideBottom && blackSideLeft && blackSideRight) {
        blackSideTop->SetTranslation(G4ThreeVector(0, Sim::SCINT_SIZE/2 + Sim::COATING_THICKNESS/2, thickness/2));
        blackSideBottom->SetTranslation(G4ThreeVector(0, -(Sim::SCINT_SIZE/2 + Sim::COATING_THICKNESS/2), thickness/2));
        blackSideLeft->SetTranslation(G4ThreeVector(-(Sim::SCINT_SIZE/2 + Sim::COATING_THICKNESS/2), 0, thickness/2));
        blackSideRight->SetTranslation(G4ThreeVector(Sim::SCINT_SIZE/2 + Sim::COATING_THICKNESS/2, 0, thickness/2));
        G4cout << "GeometryConstructor: Black tape side placements updated" << G4endl;
    } else {
        G4cerr << "ERROR: One or more black tape side physical volumes not found!" << G4endl;
    }

    if (blackBackLog) {
        G4Box* blackBackSolid = dynamic_cast<G4Box*>(blackBackLog->GetSolid());
        if (blackBackSolid) {
            // Back box dimensions remain the same (covers XY plane)
            G4cout << "GeometryConstructor: Black tape back solid verified" << G4endl;
        } else {
            G4cerr << "ERROR: blackBackLog has no valid G4Box solid!" << G4endl;
        }
    } else {
        G4cerr << "ERROR: blackBackLog is nullptr!" << G4endl;
    }

    G4VPhysicalVolume* blackBack = physVolStore->GetVolume("black_back", false);
    if (blackBack) {
        blackBack->SetTranslation(G4ThreeVector(0, 0, -Sim::COATING_THICKNESS/2));
        G4cout << "GeometryConstructor: Black tape back placement updated" << G4endl;
    } else {
        G4cerr << "ERROR: black_back not found in volume store!" << G4endl;
    }

    // Update sample position to maintain alignment
    G4VPhysicalVolume* samplePhys = physVolStore->GetVolume("SamplePhys", false);
    if (samplePhys) {
        samplePhys->SetTranslation(G4ThreeVector(0, 0, -thickness - Sim::COATING_THICKNESS - Sim::SAMPLE_THICKNESS/2));
        G4cout << "GeometryConstructor: Sample placement updated" << G4endl;
    } else {
        G4cerr << "ERROR: SamplePhys not found in volume store!" << G4endl;
    }

    // Update monitor position
    G4VPhysicalVolume* monitorPhys = physVolStore->GetVolume("MonitorPhys", false);
    if (monitorPhys) {
        monitorPhys->SetTranslation(G4ThreeVector(0, 0, thickness + 0.5*um));
        G4cout << "GeometryConstructor: Monitor placement updated" << G4endl;
    } else {
        G4cerr << "ERROR: MonitorPhys not found in volume store!" << G4endl;
    }

    G4RunManager::GetRunManager()->GeometryHasBeenModified();
}

void GeometryConstructor::UpdateSampleGeometry(G4double thickness, G4Material* material, G4double width) {
    G4cout << "GeometryConstructor: Updating sample geometry with thickness: " 
           << thickness/cm << " cm, width: " << width/cm << " cm, material: " 
           << (material ? material->GetName() : "null") << G4endl;
    
    G4PhysicalVolumeStore* physVolStore = G4PhysicalVolumeStore::GetInstance();
    
    if (sampleLog) {
        G4Box* sampleSolid = dynamic_cast<G4Box*>(sampleLog->GetSolid());
        if (sampleSolid) {
            sampleSolid->SetXHalfLength(width/2); // Update sample width
            sampleSolid->SetZHalfLength(thickness/2); // Update sample thickness
            sampleLog->SetMaterial(material);
            G4cout << "GeometryConstructor: Sample solid and material updated" << G4endl;
        } else {
            G4cerr << "ERROR: sampleLog has no valid G4Box solid!" << G4endl;
        }
    } else {
        G4cerr << "ERROR: sampleLog is nullptr!" << G4endl;
    }

    G4VPhysicalVolume* samplePhys = physVolStore->GetVolume("SamplePhys", false);
    if (samplePhys) {
        samplePhys->SetTranslation(G4ThreeVector(-Sim::SCINT_SIZE/2 + width/2, 0, -Sim::SCINT_THICKNESS - Sim::COATING_THICKNESS - thickness/2));
        G4cout << "GeometryConstructor: Sample placement updated" << G4endl;
    } else {
        G4cerr << "ERROR: SamplePhys not found in volume store!" << G4endl;
    }

    G4RunManager::GetRunManager()->GeometryHasBeenModified();
}

G4VPhysicalVolume* GeometryConstructor::createWorld() {
    G4cout << "GeometryConstructor: Creating world volume..." << G4endl;
    G4double worldZSize = std::max(Sim::WORLD_SIZE, Sim::SCINT_THICKNESS + Sim::SAMPLE_THICKNESS + Sim::COATING_THICKNESS + 50*cm);
    G4Box* worldSolid = new G4Box("WorldSolid", Sim::WORLD_SIZE/2, Sim::WORLD_SIZE/2, worldZSize/2);
    G4LogicalVolume* worldLog = new G4LogicalVolume(worldSolid, matBuilder->getVacuum(), "WorldLog");
    G4VPhysicalVolume* worldPhys = new G4PVPlacement(nullptr, G4ThreeVector(), worldLog, "World", nullptr, false, 0, true);
    
    G4VisAttributes* visAttr = new G4VisAttributes(G4Colour(1.0, 1.0, 1.0));
    visAttr->SetVisibility(false);
    visAttr->SetForceWireframe(true);
    worldLog->SetVisAttributes(visAttr);
    return worldPhys;
}

G4LogicalVolume* GeometryConstructor::buildLShape(G4LogicalVolume* worldLog) {
    G4cout << "GeometryConstructor: Building L-shape volume..." << G4endl;
    G4double minZSize = std::max(30*cm, Sim::SCINT_THICKNESS*2 + 5*cm);
    G4Box* arm1 = new G4Box("Arm1", 10*cm, 10*cm, minZSize);
    G4Box* arm2 = new G4Box("Arm2", 15*cm, 10*cm, 10*cm);
    G4UnionSolid* lShapeSolid = new G4UnionSolid("LShapeSolid", arm1, arm2, nullptr, G4ThreeVector(25*cm, 0, 20*cm));
    G4Box* cutBox = new G4Box("CutBox", 50*cm, 50*cm, 100*cm);
    G4SubtractionSolid* trimmedLShape = new G4SubtractionSolid("TrimmedLShape", lShapeSolid, cutBox, 
                                                               nullptr, G4ThreeVector(0, 0, -100.5*cm));
    G4LogicalVolume* lShapeLog = new G4LogicalVolume(trimmedLShape, matBuilder->getAir(), "LShapeLog");
    new G4PVPlacement(nullptr, G4ThreeVector(), lShapeLog, "LShapePhys", worldLog, false, 0, true);

    G4OpticalSurface* blackSurf = new G4OpticalSurface("DarkSurface");
    blackSurf->SetType(dielectric_metal);
    blackSurf->SetFinish(ground);
    blackSurf->SetModel(unified);
    G4MaterialPropertiesTable* blackSurfProp = new G4MaterialPropertiesTable();
    G4double PhotonEnergyPVT[12] = {2.08*eV, 2.38*eV, 2.58*eV, 2.7*eV, 2.76*eV, 2.82*eV,
                                    2.92*eV, 2.95*eV, 3.02*eV, 3.1*eV, 3.26*eV, 3.44*eV};
    G4double reflectivity[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    G4double efficiency[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    blackSurfProp->AddProperty("REFLECTIVITY", PhotonEnergyPVT, reflectivity, 12);
    blackSurfProp->AddProperty("EFFICIENCY", PhotonEnergyPVT, efficiency, 12);
    blackSurf->SetMaterialPropertiesTable(blackSurfProp);
    new G4LogicalSkinSurface("DarkLShape", lShapeLog, blackSurf);

    G4VisAttributes* visAttr = new G4VisAttributes(G4Colour(0.6, 0.3, 0.1));
    visAttr->SetVisibility(true);
    visAttr->SetForceSolid(false);
    lShapeLog->SetVisAttributes(visAttr);
    return lShapeLog;
}

void GeometryConstructor::addComponents(G4LogicalVolume* lShapeLog) {
    G4cout << "GeometryConstructor: Adding components..." << G4endl;
    // Scintillator
    G4Box* scintSolid = new G4Box("ScintSolid", Sim::SCINT_SIZE/2, Sim::SCINT_SIZE/2, Sim::SCINT_THICKNESS/2);
    G4VisAttributes* scintVisAttributes = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.5));
    scintVisAttributes->SetForceSolid(true);
    scintVisAttributes->SetVisibility(true);
    G4Material* scintMaterial = matBuilder->getScintillator();
    if (!scintMaterial) {
        G4cerr << "ERROR: Scintillator material is nullptr, defaulting to PVT!" << G4endl;
        scintMaterial = matBuilder->getPVT();
    }
    scintLog = new G4LogicalVolume(scintSolid, scintMaterial, "ScintLog");
    if (!scintLog) {
        G4cerr << "ERROR: scintLog is nullptr!" << G4endl;
    } else {
        G4cout << "GeometryConstructor: Scintillator logical volume created with material " 
               << scintMaterial->GetName() << G4endl;
    }
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, Sim::SCINT_THICKNESS/2), scintLog, "ScintPhys", lShapeLog, false, 0, true);
    scintLog->SetVisAttributes(scintVisAttributes);
    scintLog->SetSensitiveDetector(eventProc);

    G4OpticalSurface* scintSurf = new G4OpticalSurface("ScintSurface");
    scintSurf->SetType(dielectric_dielectric);
    scintSurf->SetFinish(polished);
    scintSurf->SetModel(unified);
    G4MaterialPropertiesTable* surfProp = new G4MaterialPropertiesTable();
    G4double PhotonEnergyPVT[12] = {2.08*eV, 2.38*eV, 2.58*eV, 2.7*eV, 2.76*eV, 2.82*eV,
                                    2.92*eV, 2.95*eV, 3.02*eV, 3.1*eV, 3.26*eV, 3.44*eV};
    G4double reflectivity[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    G4double transmittance[12] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    surfProp->AddProperty("REFLECTIVITY", PhotonEnergyPVT, reflectivity, 12);
    surfProp->AddProperty("TRANSMITTANCE", PhotonEnergyPVT, transmittance, 12);
    scintSurf->SetMaterialPropertiesTable(surfProp);
    new G4LogicalSkinSurface("ScintSkinSurface", scintLog, scintSurf);

    // Black tape side boxes
    G4Box* blackSideSolid = new G4Box("black_side_box", Sim::SCINT_SIZE/2, Sim::COATING_THICKNESS/2, Sim::SCINT_THICKNESS/2);
    blackSideLog = new G4LogicalVolume(blackSideSolid, matBuilder->getVacuum(), "black_side_log");
    blackSideLog->SetVisAttributes(new G4VisAttributes(G4Colour(0.1, 0.1, 0.1)));

    G4ThreeVector placement_top(0, Sim::SCINT_SIZE/2 + Sim::COATING_THICKNESS/2, Sim::SCINT_THICKNESS/2);
    new G4PVPlacement(nullptr, placement_top, blackSideLog, "black_side_top", lShapeLog, false, 0, true);

    G4ThreeVector placement_bottom(0, -(Sim::SCINT_SIZE/2 + Sim::COATING_THICKNESS/2), Sim::SCINT_THICKNESS/2);
    new G4PVPlacement(nullptr, placement_bottom, blackSideLog, "black_side_bottom", lShapeLog, false, 1, true);

    G4RotationMatrix* sideRotation = new G4RotationMatrix();
    sideRotation->rotateZ(90.*deg);

    G4ThreeVector placement_left(-(Sim::SCINT_SIZE/2 + Sim::COATING_THICKNESS/2), 0, Sim::SCINT_THICKNESS/2);
    new G4PVPlacement(sideRotation, placement_left, blackSideLog, "black_side_left", lShapeLog, false, 2, true);

    G4ThreeVector placement_right(Sim::SCINT_SIZE/2 + Sim::COATING_THICKNESS/2, 0, Sim::SCINT_THICKNESS/2);
    new G4PVPlacement(sideRotation, placement_right, blackSideLog, "black_side_right", lShapeLog, false, 3, true);

    // Black tape back box
    G4Box* blackBackSolid = new G4Box("black_back_box", Sim::SCINT_SIZE/2, Sim::SCINT_SIZE/2, Sim::COATING_THICKNESS/2);
    blackBackLog = new G4LogicalVolume(blackBackSolid, matBuilder->getVacuum(), "black_back_log");
    blackBackLog->SetVisAttributes(new G4VisAttributes(G4Colour(0.1, 0.1, 0.1)));
    G4ThreeVector placement_back(0, 0, -Sim::COATING_THICKNESS/2);
    new G4PVPlacement(nullptr, placement_back, blackBackLog, "black_back", lShapeLog, false, 4, true);

    G4OpticalSurface* blackTapeSurf = new G4OpticalSurface("BlackTapeSurface");
    blackTapeSurf->SetType(dielectric_metal);
    blackTapeSurf->SetFinish(ground);
    blackTapeSurf->SetModel(unified);
    G4MaterialPropertiesTable* blackTapeProp = new G4MaterialPropertiesTable();
    G4double blackReflectivity[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    G4double blackEfficiency[12] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    blackTapeProp->AddProperty("REFLECTIVITY", PhotonEnergyPVT, blackReflectivity, 12);
    blackTapeProp->AddProperty("EFFICIENCY", PhotonEnergyPVT, blackEfficiency, 12);
    blackTapeSurf->SetMaterialPropertiesTable(blackTapeProp);

    new G4LogicalSkinSurface("black_side_top_surf", blackSideLog, blackTapeSurf);
    new G4LogicalSkinSurface("black_side_bottom_surf", blackSideLog, blackTapeSurf);
    new G4LogicalSkinSurface("black_side_left_surf", blackSideLog, blackTapeSurf);
    new G4LogicalSkinSurface("black_side_right_surf", blackSideLog, blackTapeSurf);
    new G4LogicalSkinSurface("black_back_surf", blackBackLog, blackTapeSurf);

    // Mirror
    G4Box* mirrorSolid = new G4Box("MirrorSolid", 95*mm, 65*mm, 0.5*um);
    G4LogicalVolume* mirrorLog = new G4LogicalVolume(mirrorSolid, matBuilder->getQuartz(), "MirrorLog");
    G4VisAttributes* mirrorVisAttributes = new G4VisAttributes(G4Colour(0.5, 0.5, 0.5, 0.5));
    mirrorVisAttributes->SetForceSolid(true);
    mirrorVisAttributes->SetVisibility(true);
    G4RotationMatrix* rot = new G4RotationMatrix();
    rot->rotateY(45*deg);
    new G4PVPlacement(rot, G4ThreeVector(0, 0, 20*cm), mirrorLog, "MirrorPhys", lShapeLog, false, 0, true);
    G4OpticalSurface* mirrorSurf = new G4OpticalSurface("ReflectiveSurface");
    mirrorSurf->SetType(dielectric_metal);
    mirrorSurf->SetFinish(polished);
    mirrorSurf->SetModel(unified);
    mirrorSurf->SetMaterialPropertiesTable(matBuilder->getQuartz()->GetMaterialPropertiesTable());
    new G4LogicalSkinSurface("MirrorSkin", mirrorLog, mirrorSurf);
    mirrorLog->SetVisAttributes(mirrorVisAttributes);

    // Sensor
    G4Box* sensorSolid = new G4Box("SensorSolid", 10*mm, 10*mm, 0.5*um);
    G4LogicalVolume* sensorLog = new G4LogicalVolume(sensorSolid, matBuilder->getAir(), "SensorLog");
    G4VisAttributes* sensorVisAttributes = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 0.5));
    sensorVisAttributes->SetForceSolid(true);
    sensorVisAttributes->SetVisibility(true);
    rot = new G4RotationMatrix();
    rot->rotateY(90*deg);
    new G4PVPlacement(rot, G4ThreeVector(30*cm, 0, 20*cm), sensorLog, "SensorPhys", lShapeLog, false, 0, true);
    sensorLog->SetVisAttributes(sensorVisAttributes);
    sensorLog->SetSensitiveDetector(eventProc);

    // Monitor
    G4Box* monitorSolid = new G4Box("MonitorSolid", Sim::SCINT_SIZE/2, Sim::SCINT_SIZE/2, 0.5*um);
    G4LogicalVolume* monitorLog = new G4LogicalVolume(monitorSolid, matBuilder->getAir(), "MonitorLog");
    G4VisAttributes* monitorVisAttributes = new G4VisAttributes(G4Colour(1.0, 0.0, 0.0, 0.5));
    monitorVisAttributes->SetForceSolid(true);
    monitorVisAttributes->SetVisibility(true);
    new G4PVPlacement(nullptr, G4ThreeVector(0, 0, Sim::SCINT_THICKNESS + 0.5*um), monitorLog, "MonitorPhys", lShapeLog, false, 0, true);
    monitorLog->SetVisAttributes(monitorVisAttributes);
    monitorLog->SetSensitiveDetector(eventProc);

    G4OpticalSurface* monitorSurf = new G4OpticalSurface("MonitorSurface");
    monitorSurf->SetType(dielectric_dielectric);
    monitorSurf->SetFinish(polished);
    monitorSurf->SetModel(unified);
    G4MaterialPropertiesTable* monitorSurfProp = new G4MaterialPropertiesTable();
    monitorSurfProp->AddProperty("TRANSMITTANCE", PhotonEnergyPVT, transmittance, 12);
    monitorSurfProp->AddProperty("REFLECTIVITY", PhotonEnergyPVT, reflectivity, 12);
    monitorSurf->SetMaterialPropertiesTable(monitorSurfProp);
    new G4LogicalSkinSurface("MonitorSkinSurface", monitorLog, monitorSurf);
}
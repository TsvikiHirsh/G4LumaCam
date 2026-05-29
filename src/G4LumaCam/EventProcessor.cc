#include "EventProcessor.hh"
#include "ParticleGenerator.hh"
#include "SimConfig.hh"
#include "G4Step.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include <filesystem>
#include <cstdlib>

EventProcessor::EventProcessor(const G4String& name, ParticleGenerator* gen) 
    : G4VSensitiveDetector(name), neutronCount(-1), batchCount(0), eventCount(0), 
      particleGen(gen), neutronRecorded(false), currentEventTriggerTime(-1.0) {
    resetData();
}

EventProcessor::~EventProcessor() {
    if (dataFile.is_open()) dataFile.close();
}

void EventProcessor::Initialize(G4HCofThisEvent*) {
    resetData();
}

void EventProcessor::resetData() {
    photons.clear();
    tracks.clear();
    neutronPos[0] = neutronPos[1] = neutronPos[2] = 0.;
    neutronEnergy = 0.;
    protonEnergy = 0.;
    neutronRecorded = false;
    currentEventTriggerTime = -1.0;
}

G4bool EventProcessor::ProcessHits(G4Step* step, G4TouchableHistory*) {
    G4Track* track = step->GetTrack();
    G4String volName = track->GetVolume()->GetName();
    G4String particleName = track->GetDefinition()->GetParticleName();
    G4StepPoint* preStep = step->GetPreStepPoint();
    G4StepPoint* postStep = step->GetPostStepPoint();
    G4ThreeVector prePos = preStep->GetPosition();
    G4ThreeVector postPos = postStep->GetPosition();
    G4ThreeVector preDir = preStep->GetMomentumDirection();
    G4int tid = track->GetTrackID();
    G4int parentID = track->GetParentID();

    // Set trigger time, neutron energy, and neutron position for every event
    if (!neutronRecorded) {
        const G4Event* event = G4RunManager::GetRunManager()->GetCurrentEvent();
        if (event && event->GetNumberOfPrimaryVertex() > 0) {
            currentEventTriggerTime = event->GetPrimaryVertex(0)->GetT0() / ns;
            neutronEnergy = particleGen ? particleGen->getParticleEnergy() : track->GetKineticEnergy() / MeV;
            G4ThreeVector primaryPos = event->GetPrimaryVertex(0)->GetPosition();
            neutronPos[0] = primaryPos.x();
            neutronPos[1] = primaryPos.y();
            neutronPos[2] = primaryPos.z();
            neutronCount++;
            neutronRecorded = true;
            if (currentEventTriggerTime < 0) {
                G4cerr << "WARNING: Invalid trigger time " << currentEventTriggerTime 
                       << " for event " << event->GetEventID() << G4endl;
            }
            if (neutronEnergy <= 0) {
                G4cerr << "WARNING: Invalid neutron energy " << neutronEnergy 
                       << " MeV for event " << event->GetEventID() << G4endl;
            }
            if (neutronPos[0] == 0 && neutronPos[1] == 0 && neutronPos[2] == 0) {
                G4cerr << "WARNING: Neutron position set to (0, 0, 0) from primary vertex for event " 
                       << event->GetEventID() << G4endl;
            }
        } else {
            G4cerr << "WARNING: No primary vertex for event " << event->GetEventID() << G4endl;
            currentEventTriggerTime = -1.0;
            neutronEnergy = 0.0;
            neutronPos[0] = neutronPos[1] = neutronPos[2] = 0.;
        }
    }

    // Record neutron position in ScintPhys or SamplePhys.
    // Nuclear interaction (non-Transportation): record the interaction point — this is
    // the true conversion position and takes priority over the entry fallback below.
    // First-step entry (Transportation, step==1): record the scintillator entry face as
    // a fallback so that events where the neutron only scatters in surrounding air (and
    // nz would otherwise stay frozen at the source vertex) still get a meaningful nz.
    if ((volName == "ScintPhys" || volName == "SamplePhys") && parentID == 0 && particleName == "neutron") {
        G4String processName = postStep->GetProcessDefinedStep() ?
                               postStep->GetProcessDefinedStep()->GetProcessName() : "None";
        if (processName != "Transportation") {
            // True nuclear interaction: overwrite with actual conversion point.
            neutronPos[0] = postPos.x();
            neutronPos[1] = postPos.y();
            neutronPos[2] = postPos.z();
        } else if (track->GetCurrentStepNumber() == 1) {
            // First Transportation step = neutron just entered this volume.
            // Only set if not already overwritten by a real interaction (nz != source z).
            if (neutronPos[2] < -1000. * mm) {
                neutronPos[0] = prePos.x();
                neutronPos[1] = prePos.y();
                neutronPos[2] = prePos.z();
            }
        }
    }

    // Track charged particles in scintillator
    if (volName == "ScintPhys" && particleName != "opticalphoton") {
        if (tracks.find(tid) == tracks.end()) {
            G4double energy = track->GetKineticEnergy() / MeV;
            if (parentID != 0 && energy <= 0) {
                energy = neutronEnergy;
            }
            // x0/y0/z0 = birth position (frozen); x/y/z updated to last scintillation step.
            tracks[tid] = {particleName, prePos.x(), prePos.y(), prePos.z(), energy, false,
                           prePos.x(), prePos.y(), prePos.z(), 0., 0., 0.};
        }

        G4String processName = postStep->GetProcessDefinedStep() ?
                               postStep->GetProcessDefinedStep()->GetProcessName() : "None";
        if (processName == "Scintillation" || processName == "Cerenkov") {
            tracks[tid].x = prePos.x();  // last scintillation step ≈ Bragg peak
            tracks[tid].y = prePos.y();
            tracks[tid].z = prePos.z();
            tracks[tid].isLightProducer = true;
            G4double currentEnergy = track->GetKineticEnergy() / MeV;
            if (currentEnergy <= 0 && tracks[tid].energy <= 0) {
                tracks[tid].energy = neutronEnergy;
            }
        }
    }

    // NEW: Capture optical photon generation position and direction
    if (particleName == "opticalphoton" && track->GetCurrentStepNumber() == 1) {
        // First step of optical photon - record where it was created
        if (tracks.find(tid) == tracks.end()) {
            tracks[tid] = {"opticalphoton", 0., 0., 0., 0., false, 
                          prePos.x(), prePos.y(), prePos.z(), 
                          preDir.x(), preDir.y(), preDir.z()};
        } else {
            // Update generation info
            tracks[tid].x0 = prePos.x();
            tracks[tid].y0 = prePos.y();
            tracks[tid].z0 = prePos.z();
            tracks[tid].dx0 = preDir.x();
            tracks[tid].dy0 = preDir.y();
            tracks[tid].dz0 = preDir.z();
        }
    }

    // Process photons that reach the monitor
    if (volName == "MonitorPhys" && particleName == "opticalphoton") {
        // Accept only photons aimed at the entrance pupil.
        // Python traces from exit position (x,y,z=0) with exit direction (dx,dy,dz).
        // Project that same ray to z = dist_from_obj = 461.535mm for the EPD check.
        static const G4double LENS_Z_MM = 461.535;       // dist_from_obj (mm)
        static const G4double EPD_R2    = 30.53 * 30.53; // (EFL/f# /2)^2, f/0.95 58mm

        G4double x_exit = prePos.x() / mm;
        G4double y_exit = prePos.y() / mm;
        G4double dx     = preDir.x();
        G4double dy     = preDir.y();
        G4double dz     = preDir.z();

        bool inLens = false;
        if (dz > 1e-6) {
            // Python traces from (x_exit, y_exit, z=0), so project from z=0 to lens
            G4double t      = LENS_Z_MM / dz;
            G4double x_lens = x_exit + dx * t;
            G4double y_lens = y_exit + dy * t;
            inLens = (x_lens*x_lens + y_lens*y_lens < EPD_R2);
        }

        if (inLens) {
            if (tracks.find(parentID) == tracks.end()) {
                tracks[parentID] = {"unknown", neutronPos[0], neutronPos[1], neutronPos[2], neutronEnergy, true, 0., 0., 0., 0., 0., 0.};
            }
            
            if (tracks[parentID].energy <= 0) {
                tracks[parentID].energy = neutronEnergy;
            }
            
            PhotonRecord rec;
            rec.id = track->GetTrackID();
            rec.parentId = parentID;
            rec.neutronId = neutronCount;

            // Birth position+direction: Python traces from here.
            // x0,y0,z0 is the true object point — no lateral drift error vs exit pos.
            // Birth dir == exit dir (black coating means no reflections, straight travel).
            if (tracks.find(tid) != tracks.end()) {
                rec.x0  = tracks[tid].x0 / mm;
                rec.y0  = tracks[tid].y0 / mm;
                rec.z0  = tracks[tid].z0 / mm;
                rec.dx0 = tracks[tid].dx0;
                rec.dy0 = tracks[tid].dy0;
                rec.dz0 = tracks[tid].dz0;
            } else {
                rec.x0 = rec.y0 = rec.z0 = 0.;
                rec.dx0 = rec.dy0 = 0.;  rec.dz0 = 1.;
            }

            rec.timeOfArrival = track->GetGlobalTime() / ns;
            rec.wavelength = 1240. / (track->GetTotalEnergy() / eV);
            rec.parentType = tracks[parentID].type;
            rec.px = tracks[parentID].x0 / mm;   // parent birth position (= neutron interaction vertex)
            rec.py = tracks[parentID].y0 / mm;
            rec.pz = tracks[parentID].z0 / mm;
            rec.parentEnergy = tracks[parentID].energy;
            rec.nx = neutronPos[0] / mm;
            rec.ny = neutronPos[1] / mm;
            rec.nz = neutronPos[2] / mm;
            rec.neutronEnergy = neutronEnergy;
            rec.pulseId = particleGen ? particleGen->getCurrentPulseIndex() : -1;
            rec.pulseTime = currentEventTriggerTime;
            photons.push_back(rec);
        }
    }
    return true;
}

void EventProcessor::EndOfEvent(G4HCofThisEvent*) {
    if (eventCount == 0 && batchCount == 0) {
        // G4cout << "EventProcessor: Starting new run with batchSize=" << Sim::batchSize << G4endl;
        openOutputFile();
    }
    
    if (!photons.empty()) writeData();
    
    if (Sim::batchSize > 0) {
        eventCount++;
        // G4cout << "EventProcessor: eventCount=" << eventCount << ", batchSize=" << Sim::batchSize << G4endl;
        if (eventCount >= Sim::batchSize) {
            batchCount++;
            eventCount = 0;
            // G4cout << "EventProcessor: Starting new batch " << batchCount << G4endl;
            openOutputFile();
        }
    }
    resetData();
}

void EventProcessor::openOutputFile() {
    if (dataFile.is_open()) dataFile.close();

    std::filesystem::path currentPath = std::filesystem::current_path();
    std::filesystem::path simPhotonsDir = currentPath / "SimPhotons";
    
    try {
        std::filesystem::create_directories(simPhotonsDir);
        // G4cout << "Created/verified directory: " << simPhotonsDir << G4endl;
    } catch (const std::filesystem::filesystem_error& e) {
        G4cerr << "ERROR: Failed to create directory " << simPhotonsDir << ": " << e.what() << G4endl;
        G4Exception("EventProcessor::openOutputFile()", "IO001", 
                    FatalException, "Cannot create SimPhotons directory");
    }

    G4String fileName = Sim::outputFileName;
    size_t csvPos = fileName.find(".csv");
    if (csvPos != G4String::npos) {
        fileName = fileName.substr(0, csvPos);
    }

    if (Sim::batchSize > 0) {
        fileName += "_" + std::to_string(batchCount) + ".csv";
    } else {
        fileName += ".csv";
    }
    
    std::filesystem::path fullPath = simPhotonsDir / std::string(fileName);
    
    // G4cout << "Opening output file: " << fullPath << G4endl;
    
    dataFile.open(fullPath);
    
    if (!dataFile.is_open()) {
        G4cerr << "ERROR: Failed to open file: " << fullPath << G4endl;
        G4Exception("EventProcessor::openOutputFile()", "IO002", 
                    FatalException, "Cannot open output file");
    }

    dataFile << std::fixed;
    
    // Updated header with generation position (x0,y0,z0) and direction (dx0,dy0,dz0)
    dataFile << "id,parent_id,neutron_id,pulse_id,pulse_time_ns,"
             << "x,y,z,dx,dy,dz,"   // photon birth pos+dir (Python traces from here)
             << "toa,wavelength,"
             << "parentName,px,py,pz,parentEnergy,"  // px/py/pz = parent birth pos
             << "nx,ny,nz,neutronEnergy\n";
}

void EventProcessor::writeData() {
    for (const auto& p : photons) {
        // Integer columns
        dataFile << p.id << "," 
                 << p.parentId << "," 
                 << p.neutronId << ","
                 << p.pulseId << ",";
        
        // HIGH PRECISION: pulse_time_ns
        dataFile << std::setprecision(15) << p.pulseTime << ",";
        
        // Photon birth position (x,y,z) and direction — Python traces from birth pos
        dataFile << std::setprecision(4) << p.x0 << "," << p.y0 << "," << p.z0 << ",";
        dataFile << std::setprecision(6) << p.dx0 << "," << p.dy0 << "," << p.dz0 << ",";

        // HIGH PRECISION: timeOfArrival
        dataFile << std::setprecision(15) << p.timeOfArrival << ",";

        // LOW PRECISION: wavelength (nm)
        dataFile << std::setprecision(2) << p.wavelength << "," << p.parentType << ",";

        // MEDIUM PRECISION: parent birth position (= neutron interaction vertex for direct events)
        dataFile << std::setprecision(4)
                 << p.px << "," << p.py << "," << p.pz << ",";

        dataFile << std::setprecision(4) << p.parentEnergy << ",";

        // MEDIUM PRECISION: neutron first interaction point and energy
        dataFile << std::setprecision(4)
                 << p.nx << "," << p.ny << "," << p.nz << ",";

        dataFile << std::setprecision(4) << p.neutronEnergy << "\n";
    }
    dataFile.flush();
}
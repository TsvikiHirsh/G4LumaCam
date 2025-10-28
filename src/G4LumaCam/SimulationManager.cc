#include "SimulationManager.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "ParticleGenerator.hh"
#include "G4UnitsTable.hh"
#include "SimConfig.hh"

SimulationManager::SimulationManager() 
    : processor(new EventProcessor("Tracker")), eventCounter(0), totalNeutrons(0) {}

void SimulationManager::BeginOfRunAction(const G4Run* run) {
    eventCounter = 0;
    
    G4cout << "\n################################################" << G4endl;
    G4cout << "### Run " << run->GetRunID() << " Starting ###" << G4endl;
    G4cout << "################################################" << G4endl;
    
    G4int eventsToProcess = run->GetNumberOfEventToBeProcessed();
    G4cout << "Events to process (from G4RunManager): " << eventsToProcess << G4endl;
    G4cout << "Total neutrons (from SimulationManager): " << totalNeutrons << G4endl;
    
    // Use the number of events from the run if totalNeutrons wasn't explicitly set
    G4int neutronsForPulseStructure = (totalNeutrons > 0) ? totalNeutrons : eventsToProcess;
    
    G4cout << "Total neutrons for pulse structure: " << neutronsForPulseStructure << G4endl;
    
    // Set total neutrons in ParticleGenerator
    G4RunManager* runManager = G4RunManager::GetRunManager();
    ParticleGenerator* generator = dynamic_cast<ParticleGenerator*>(
        const_cast<G4VUserPrimaryGeneratorAction*>(runManager->GetUserPrimaryGeneratorAction()));
    
    if (generator) {
        // Check if pulsed beam is configured
        if (Sim::FLUX > 0 && Sim::FREQ > 0) {
            G4cout << "\n=== Pulsed Beam Configuration ===" << G4endl;
            G4cout << "Flux: " << Sim::FLUX << " n/cm²/s" << G4endl;
            G4cout << "Frequency: " << Sim::FREQ << " Hz" << G4endl;
            G4cout << "Batch size: " << Sim::batchSize << G4endl;
            G4cout << "Setting up pulse structure..." << G4endl;
            
            generator->SetTotalNeutrons(neutronsForPulseStructure);
            
            G4cout << "Pulse structure setup complete!" << G4endl;
            G4cout << "Total pulses created: " << Sim::pulseTimes.size() << G4endl;
            G4cout << "==================================" << G4endl;
        } else {
            G4cout << "\nRunning in continuous beam mode (FLUX=" << Sim::FLUX 
                   << ", FREQ=" << Sim::FREQ << ")" << G4endl;
        }
    } else {
        G4cerr << "ERROR: Could not find ParticleGenerator!" << G4endl;
    }
    
    G4cout << "################################################\n" << G4endl;
}

void SimulationManager::EndOfRunAction(const G4Run* run) {
    G4cout << "\n################################################" << G4endl;
    G4cout << "### Run " << run->GetRunID() << " Ended ###" << G4endl;
    G4cout << "Total events processed: " << eventCounter << G4endl;
    G4cout << "################################################\n" << G4endl;
    
    // Clear pulse structure for next run
    Sim::pulseTimes.clear();
    Sim::neutronsPerPulse.clear();
}

void SimulationManager::SetTotalNeutrons(G4int nNeutrons) {
    this->totalNeutrons = nNeutrons;
    G4cout << "SimulationManager: Total neutrons set to " << nNeutrons << G4endl;
}

SimulationManager::EventHandler::EventHandler(SimulationManager* mgr) : manager(mgr) {}

void SimulationManager::EventHandler::BeginOfEventAction(const G4Event*) {}

void SimulationManager::EventHandler::EndOfEventAction(const G4Event*) {
    manager->eventCounter++;
    
    // Print progress every 100 events
    if (manager->eventCounter % 100 == 0) {
        G4cout << "Processed " << manager->eventCounter << " events..." << G4endl;
    }
}
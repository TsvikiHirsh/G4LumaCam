#ifndef SIMULATION_MANAGER_HH
#define SIMULATION_MANAGER_HH

#include "G4UserRunAction.hh"
#include "G4UserEventAction.hh"
#include "EventProcessor.hh"

class SimulationManager : public G4UserRunAction {
public:
    SimulationManager();
    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*) override;
    void SetTotalNeutrons(G4int totalNeutrons);

    class EventHandler : public G4UserEventAction {
    public:
        EventHandler(SimulationManager* mgr);
        void BeginOfEventAction(const G4Event*) override;
        void EndOfEventAction(const G4Event*) override;
    private:
        SimulationManager* manager;
    };

private:
    EventProcessor* processor;
    G4int eventCounter;
    G4int totalNeutrons;
};

#endif
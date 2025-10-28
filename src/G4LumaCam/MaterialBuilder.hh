#ifndef MATERIAL_BUILDER_HH
#define MATERIAL_BUILDER_HH

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4ios.hh"

class MaterialBuilder {
public:
    MaterialBuilder();
    ~MaterialBuilder() {}

    enum class ScintType { EJ200, GS20, LYSO };

    void DefineMaterials();
    G4Material* getVacuum() const { return vacuum; }
    G4Material* getAir() const { return air; }
    G4Material* getPVT() const { return scintMaterialPVT; }
    G4Material* getGS20() const { return scintMaterialGS20; }
    G4Material* getLYSO() const { return scintMaterialLYSO; }
    G4Material* getGraphite() const { return sampleMaterial; }
    G4Material* getQuartz() const { return windowMaterial; }
    G4Material* getBlackMat() const { return absorberMaterial; }
    G4Material* getScintillator() const { return scintMaterial; }

    void setScintillatorType(ScintType type);
    void setScintillatorType(const G4String& typeName);

private:
    void setupMaterialProperties(G4Material* mat, const G4double* energies,
                                const G4double* rindex, const G4double* abslength,
                                int nEntries, const G4double* scintillation = nullptr);
    void buildGS20();
    void buildLYSO();

    G4Material* vacuum;
    G4Material* air;
    G4Material* scintMaterialPVT;
    G4Material* scintMaterialGS20;
    G4Material* scintMaterialLYSO;
    G4Material* scintMaterial;
    G4Material* sampleMaterial;
    G4Material* windowMaterial;
    G4Material* absorberMaterial;

    ScintType currentScintType;
};

#endif
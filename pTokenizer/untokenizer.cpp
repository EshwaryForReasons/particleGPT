
#include "main.h"
#include "utils.h"
#include "dictionary.h"

#include "G4RunManager.hh"
#include "G4PhysListFactory.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4ParticleTable.hh"
#include "G4IonTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4GenericIon.hh"

#include "TLorentzVector.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <ranges>
#include <algorithm>
#include <thread>
#include <iomanip>
#include <format>

G4ParticleTable* particleTable = nullptr;
G4RunManager* runManager = nullptr;

//Need a dummy detector to initialize the G4RunManager properly
class DummyDetector : public G4VUserDetectorConstruction
{
public:
    DummyDetector() = default;
    virtual ~DummyDetector() = default;

    virtual G4VPhysicalVolume* Construct() override
    {
        // Just build a tiny dummy box world
        G4NistManager* nist = G4NistManager::Instance();
        G4Material* air = nist->FindOrBuildMaterial("G4_AIR");

        G4Box* solidWorld = new G4Box("World", 1 * CLHEP::m, 1 * CLHEP::m, 1 * CLHEP::m);
        G4LogicalVolume* logicWorld = new G4LogicalVolume(solidWorld, air, "World");
        G4VPhysicalVolume* physWorld = new G4PVPlacement(
            0,                     // no rotation
            G4ThreeVector(),       // at (0,0,0)
            logicWorld,            // logical volume
            "World",               // name
            0,                     // mother volume
            false,                 // no boolean operation
            0,                     // copy number
            true);                 // check overlaps

        return physWorld;
    }
};

void initialize_geant4()
{
    G4PhysListFactory factory;
    G4VModularPhysicsList* physicsList = factory.GetReferencePhysList("FTFP_BERT_ATL");
    if (!physicsList)
        throw std::runtime_error("Failed to get reference physics list.");
    
    runManager = new G4RunManager();
    if (!runManager)
        throw std::runtime_error("Failed to create G4RunManager instance.");
    runManager->SetUserInitialization(new DummyDetector());
    runManager->SetUserInitialization(physicsList);
    runManager->Initialize();

    particleTable = G4ParticleTable::GetParticleTable();
    if (!particleTable)
        throw std::runtime_error("Failed to get G4ParticleTable instance.");
    particleTable->SetReadiness();
    //Add ions to the particle table
    G4IonTable* ions = particleTable->GetIonTable();
    ions->CreateAllIon();
    ions->CreateAllIsomer();
}

/**
 * Geant4 lazy loads ions and, resultantly, some I need are not included by default.
 * This parses the PDGID and loads those ions as needed.
 * 
 * ionTable->GetIon() creates the ion if it does not exist, so we can use it to create ions on demand.
 */
G4ParticleDefinition* FindParticleByPDGID(int pdgCode)
{
    // Not a nuclear PDG code, use normal lookup
    if (pdgCode < 1000000000)
        return G4ParticleTable::GetParticleTable()->FindParticle(pdgCode);

    // Extract Z, A, and isomer level
    int Z = (pdgCode / 10000) % 1000;
    int A = (pdgCode / 10) % 1000;
    int I = pdgCode % 10;
    double excitationEnergy = I * CLHEP::MeV;

    G4IonTable* ionTable = G4ParticleTable::GetParticleTable()->GetIonTable();
    return ionTable->GetIon(Z, A, excitationEnergy);
}

void Tokenizer::untokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path)
{
    std::printf("----------------------------------------\n");
    const auto dictionary = DataManager::load_dictionary(dictionary_path);
    const auto tokenized_data = DataManager::load_tokenized_data(input_data_path);
    initialize_geant4();

    std::printf("pTokenizer: untokenizer: Began untokenizing data.\n");

    //Profiling shows no reason to multithread this one (more threads was actually slower?)
    std::vector<std::vector<double>> raw_data;
    for (auto& event : tokenized_data)
    {
        const auto untokenized_event = untokenize_event(event, dictionary);
        raw_data.push_back(untokenized_event);
    }

    DataManager::output_raw_data(output_data_path, raw_data);
    std::printf("pTokenizer: untokenizer: Finished untokenizing data.\n");
    std::printf("----------------------------------------\n");
}

const std::vector<double> Tokenizer::untokenize_event(const std::vector<int>& event, const Dictionary& dictionary)
{
    std::vector<double> untokenized_event;
    for (int i = 0; i < event.size(); i += dictionary.get_num_tokens_per_particle())
    {
        const auto determine_bin_idx = [&](const std::string& type_str, int type_offset) -> std::size_t {
            const auto it = std::ranges::find(dictionary.tokenization_schema, type_str);
            if (it == dictionary.tokenization_schema.end())
                return (std::size_t)-1;
            const std::size_t type_pos = std::distance(dictionary.tokenization_schema.begin(), it);
            return event[i + type_pos] - type_offset;
        };

        int pdgid_idx         = determine_bin_idx("pdgid", dictionary.offsets.pdgid_offset);
        int energy_bin_idx    = determine_bin_idx("energy", dictionary.offsets.energy_offset);
        int pt_bin_idx        = determine_bin_idx("pt", dictionary.offsets.pt_offset);
        int eta_bin_idx       = determine_bin_idx("eta", dictionary.offsets.eta_offset);
        int theta_bin_idx     = determine_bin_idx("theta", dictionary.offsets.theta_offset);
        int phi_bin_idx       = determine_bin_idx("phi", dictionary.offsets.phi_offset);

        // We can reasonably assume pdgid exists in the tokenization since it is needed to have a proper particle.
        int pdgid = 0;
        for (auto& [pdg_id, pdg_idx] : dictionary.pdgid_to_index)
        {
            if (pdg_idx == pdgid_idx)
            {
                pdgid = pdg_id;
                break;
            }
        }

        double energy = 0.0f;
        double pt = 0.0f;
        double eta = 0.0f;
        double theta = 0.0f;
        double phi = 0.0f;

        if (energy_bin_idx != (std::size_t)-1)
            energy = pMath::get_bin_median(dictionary.e_bins, energy_bin_idx);
        if (pt_bin_idx != (std::size_t)-1)
            pt = pMath::get_bin_median(dictionary.pt_bins, pt_bin_idx);
        if (eta_bin_idx != (std::size_t)-1)
            eta = pMath::get_bin_median(dictionary.eta_bins, eta_bin_idx);
        if (theta_bin_idx != (std::size_t)-1)
            theta = pMath::get_bin_median(dictionary.theta_bins, theta_bin_idx);
        if (phi_bin_idx != (std::size_t)-1)
            phi = pMath::get_bin_median(dictionary.phi_bins, phi_bin_idx);

        auto* particle = FindParticleByPDGID(pdgid);

        TLorentzVector vec;
        
        // Either pt or energy should exist.
        if (pt_bin_idx != (std::size_t)-1 && eta_bin_idx != (std::size_t)-1 && phi_bin_idx != (std::size_t)-1)
        {
            vec.SetPtEtaPhiM(pt, eta, phi, particle->GetPDGMass());
        }
        else if (energy_bin_idx != (std::size_t)-1 && theta_bin_idx != (std::size_t)-1 && phi_bin_idx != (std::size_t)-1)
        {
            vec.SetE(energy);
            vec.SetTheta(theta);
            vec.SetPhi(phi);
        }
        else
            throw std::runtime_error("pTokenizer: untokenizer: Cannot calculate linear momentum.");

        untokenized_event.push_back(pdgid);
        untokenized_event.push_back(vec.Energy());
        untokenized_event.push_back(vec.Px());
        untokenized_event.push_back(vec.Py());
        untokenized_event.push_back(vec.Pz());
    }
    return untokenized_event;
}

#include "dictionary.h"
#include "utils.h"

#include "json.hpp"

#include <fstream>

Dictionary::Dictionary(const std::string& _dictionary_path)
    : dictionary_path(_dictionary_path)
{
    //Load dictionary

    std::ifstream dictionary_file(dictionary_path);
    nlohmann::json dictionary_json = nlohmann::json::parse(dictionary_file);

    //Load special tokens

    special_tokens.padding = dictionary_json["special_tokens"]["padding"];
    special_tokens.event_start = dictionary_json["special_tokens"]["event_start"];
    special_tokens.event_end = dictionary_json["special_tokens"]["event_end"];
    special_tokens.particle_start = dictionary_json["special_tokens"]["particle_start"];
    special_tokens.particle_end = dictionary_json["special_tokens"]["particle_end"];

    //Load particles

    nlohmann::json particles_index_json = dictionary_json["particles_index"];
    nlohmann::json particles_id_json = dictionary_json["particles_id"];
    for (auto& [particle_id, particle_name] : particles_id_json.items())
    {
        for (auto& [particle_name_inner, particle_index] : particles_index_json.items())
        {
            if (particle_name == particle_name_inner)
            {
                pdgid_index_pair pair = { std::stoi(particle_id), particle_index };
                pdgid_to_index.push_back(pair);
                break;
            }
        }
    }

    //Generate bins

    double e_min = dictionary_json["e_bin_data"]["min"];
    double e_max = dictionary_json["e_bin_data"]["max"];
    double e_step = dictionary_json["e_bin_data"]["step_size"];
    e_bins = pMath::arange(e_min, e_max, e_step);

    double eta_min = dictionary_json["eta_bin_data"]["min"];
    double eta_max = dictionary_json["eta_bin_data"]["max"];
    double eta_step = dictionary_json["eta_bin_data"]["step_size"];
    eta_bins = pMath::arange(eta_min, eta_max, eta_step);

    double theta_min = -2 * M_PI;
    double theta_max = 2 * M_PI;
    double theta_step = dictionary_json["theta_bin_data"]["step_size"];
    theta_bins = pMath::arange(theta_min, theta_max, theta_step);

    double phi_min = -2 * M_PI;
    double phi_max = 2 * M_PI;
    double phi_step = dictionary_json["phi_bin_data"]["step_size"];
    phi_bins = pMath::arange(phi_min, phi_max, phi_step);

    //Calculate offsets

    std::size_t num_special_tokens = dictionary_json["special_tokens"].size();
    std::size_t num_particles = dictionary_json["particles_index"].size();
    std::size_t num_materials = dictionary_json["materials_named"].size();

    offsets.special_tokens_offset = 0;
    offsets.pdgid_offset = offsets.special_tokens_offset + num_special_tokens;
    offsets.materials_offset = offsets.pdgid_offset + num_particles;
    offsets.energy_offset = offsets.materials_offset + num_materials;
    offsets.eta_offset = offsets.energy_offset + e_bins.size();
    offsets.theta_offset = offsets.eta_offset + eta_bins.size();
    offsets.phi_offset = offsets.theta_offset + theta_bins.size();
}

#include "main.h"
#include "utils.h"

#include "json.hpp"

#include <pybind11/pybind11.h>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <format>

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

    // double theta_min = -2 * M_PI;
    // double theta_max = 2 * M_PI;
    // double theta_step = dictionary_json["theta_bin_data"]["step_size"];
    // theta_bins = pMath::arange(theta_min, theta_max, theta_step);

    // double phi_min = - 2 * M_PI;
    // double phi_max =  2 * M_PI;
    // double phi_step = dictionary_json["phi_bin_data"]["step_size"];

    double theta_min = dictionary_json["theta_bin_data"]["min"];
    double theta_max = dictionary_json["theta_bin_data"]["max"];
    double theta_step = dictionary_json["theta_bin_data"]["step_size"];
    theta_bins = pMath::arange(theta_min, theta_max, theta_step);

    double phi_min = dictionary_json["phi_bin_data"]["min"];
    double phi_max = dictionary_json["phi_bin_data"]["max"];
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

namespace DataManager
{
const Dictionary load_dictionary(std::string dictionary_path)
{
    if (!std::filesystem::exists(dictionary_path))
    {
        std::cout << "pTokenizer: Dictionary not found at " << dictionary_path << std::endl;
        throw std::runtime_error("pTokenizer: Dictionary not found.");
    }

    std::cout << "pTokenizer: Found dictionary at " << dictionary_path << std::endl;
    Dictionary dictionary = Dictionary(dictionary_path);
    std::cout << "pTokenizer: Dictionary successfully loaded." << std::endl;
    return dictionary;
}

//We assume tokenized data has the structure
//"event_start particle_start pdgid energy eta theta phi particle_end particle_start pdgid energy eta theta phi particle_end ... event_end"
//Generally this will be the generated samples from the model
const std::vector<std::vector<int>> load_tokenized_data(std::string input_data_path)
{
    std::printf("pTokenizer: Began loading tokenized data.\n");

    if (!std::filesystem::exists(input_data_path))
    {
        std::cout << "pTokenizer: Tokenized data not found at " << input_data_path << std::endl;
        throw std::runtime_error("pTokenizer: Tokenized data not found.");
    }

    std::vector<std::vector<int>> loaded_tokenized_data;
    
    std::ifstream input_data_file(input_data_path);
    std::string event;
    int event_counter = 0;
    while (std::getline(input_data_file, event))
    {
        loaded_tokenized_data.push_back(std::vector<int>());
        std::stringstream particle_stream(event);
        std::string token;
        while (std::getline(particle_stream, token, ' '))
        {
            int value = std::stoi(token);
            loaded_tokenized_data[event_counter].push_back(value);
        }

        event_counter++;
    }

    std::printf("pTokenizer: Finished loading tokenized data.\n");
    return loaded_tokenized_data;
}

void output_tokenized_data(std::string output_file_path, const std::vector<std::vector<int>>& tokenized_data)
{
    std::printf("pTokenizer: Began outputting tokenized data.\n");
    std::ofstream output_file(output_file_path);
    for (int i = 0; i < tokenized_data.size(); ++i)
    {
        auto event = tokenized_data[i];
        for (int j = 0; j < event.size(); ++j)
        {
            output_file << event[j];
            if (j != event.size() - 1)
                output_file << " ";
        }

        if (i != tokenized_data.size() - 1)
            output_file << "\n";
    }
    std::printf("pTokenizer: Finished outputting tokenized data.\n");
}

void output_raw_data(std::string output_file_path, const std::vector<std::vector<double>>& raw_data)
{
    std::printf("pTokenizer: Began outputting raw data.\n");
    std::ofstream output_file(output_file_path);
    for (auto event : raw_data)
    {
        for (int i = 0; i < event.size(); i += 5)
        {
            output_file << std::format("{:d} {:.5f} {:.5f} {:.5f} {:.5f};", (int)event[i], event[i + 1], event[i + 2], event[i + 3], event[i + 4]);
        }
        output_file << "\n";
    }
    std::printf("pTokenizer: Finished outputting raw data.\n");
}
}

PYBIND11_MODULE(pTokenizerModule, m)
{
    m.doc() = "Backend for tokenization and untokenization of data.";
    m.def("tokenize_data", &StandardScheme::tokenize_data);
    m.def("tokenize_data_scheme_no_eta", &SchemeNoEta::tokenize_data);
    m.def("tokenize_data_scheme_no_particle_boundaries", &SchemeNoParticleBoundaries::tokenize_data);
    m.def("tokenize_data_scheme_paddingv2", &SchemePaddingV2::tokenize_data);
    m.def("untokenize_data", &StandardScheme::untokenize_data);
    m.def("untokenize_data_scheme_no_eta", &SchemeNoEta::untokenize_data);
    m.def("untokenize_data_scheme_no_particle_boundaries", &SchemeNoParticleBoundaries::untokenize_data);
    m.def("untokenize_data_scheme_paddingv2", &SchemePaddingV2::untokenize_data);
}
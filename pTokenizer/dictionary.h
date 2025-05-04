
#pragma once

#include "utils.h"

#include "json.hpp"

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

struct FSpecialTokens {
    int padding = -1;
    int event_start = -1;
    int event_end = -1;
    int particle_start = -1;
    int particle_end = -1;
};

struct FOffsets {
    int special_tokens_offset = 0;
    int pdgid_offset = 0;
    int materials_offset = 0;
    int energy_offset = 0;
    int eta_offset = 0;
    int theta_offset = 0;
    int phi_offset = 0;
    int pt_offset = 0;
};

struct pdgid_index_pair
{
    int pdgid = 0;
    int index = 0;
};

class Dictionary
{
public:

    Dictionary() = default;
    Dictionary(const std::string& _dictionary_path)
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

        nlohmann::json pdgids_json = dictionary_json["pdgids"];
        for (auto& [idx_str, pdgid] : pdgids_json.items())
        {
            const int idx = pMath::fast_stoi(idx_str);
            pdgid_index_pair pair = { pdgid, idx };
            pdgid_to_index.push_back(pair);
        }

        //Generate bins

        auto create_bins = [=](const std::string& type_str)
        {
            const std::string token_bin_key_name = type_str + "_bin_data";
            if (!dictionary_json.contains(token_bin_key_name))
                return BinData(std::vector<double>(), "none");
            
            //See if we should use arange or linspace. step_size implies arange and n_bins implies linspace.
            const bool b_use_linspace = dictionary_json[token_bin_key_name].contains("n_bins");
            auto bin_generation_func = b_use_linspace ? pMath::linspace : pMath::arange;
            const std::string spacing_key = b_use_linspace ? "n_bins" : "step_size";

            //Determine the transform function
            std::string transform_type = "linear";
            if (dictionary_json[token_bin_key_name].contains("transform"))
                transform_type = dictionary_json[token_bin_key_name]["transform"];
            auto ret_same = [](double x) { return x; };
            auto ret_log = [](double x) { return std::log(std::max(1.0, x)); };
            auto transform_func = transform_type == "log" ? ret_log : ret_same;

            std::vector<double> bins = bin_generation_func(
                transform_func(dictionary_json[token_bin_key_name]["min"]),
                transform_func(dictionary_json[token_bin_key_name]["max"]),
                dictionary_json[token_bin_key_name][spacing_key]);
            return BinData{ bins, transform_type };
        };

        e_bins     = create_bins("e");
        eta_bins   = create_bins("eta");
        theta_bins = create_bins("theta");
        phi_bins   = create_bins("phi");
        pt_bins    = create_bins("pt");

        //Calculate offsets

        std::size_t num_special_tokens = dictionary_json["special_tokens"].size();
        std::size_t num_particles = dictionary_json["particles_index"].size();
        std::size_t num_materials = dictionary_json["materials_named"].size();

        offsets.special_tokens_offset = 0;
        offsets.pdgid_offset = offsets.special_tokens_offset + num_special_tokens;
        offsets.materials_offset = offsets.pdgid_offset + num_particles;
        offsets.energy_offset = offsets.materials_offset + num_materials;
        offsets.eta_offset = offsets.energy_offset + e_bins.bins.size();
        offsets.theta_offset = offsets.eta_offset + eta_bins.bins.size();
        offsets.phi_offset = offsets.theta_offset + theta_bins.bins.size();
        offsets.pt_offset = offsets.phi_offset + phi_bins.bins.size();
    }

    std::string dictionary_path;

    FSpecialTokens special_tokens;
    FOffsets offsets;

    BinData e_bins;
    BinData eta_bins;
    BinData theta_bins;
    BinData phi_bins;
    BinData pt_bins;
    std::vector<pdgid_index_pair> pdgid_to_index;
};
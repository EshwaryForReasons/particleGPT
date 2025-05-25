
#include "main.h"
#include "utils.h"
#include "dictionary.h"

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

void Tokenizer::untokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path)
{
    std::printf("----------------------------------------\n");
    const auto dictionary = DataManager::load_dictionary(dictionary_path);
    const auto tokenized_data = DataManager::load_tokenized_data(input_data_path);
    std::printf("pTokenizer: untokenizer: Began untokenizing data.\n");
    std::vector<std::vector<double>> raw_data;
    
    //Profiling shows no reason to multithread this one (more threads was actually slower?)
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

        // We can reasonably assume pdgid exists since it is need to have a proper particle.
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

        double px = 0.0f;
        double py = 0.0f;
        double pz = 0.0f;

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

        // Either pt or energy should exist.
        if (pt_bin_idx != (std::size_t)-1 && eta_bin_idx != (std::size_t)-1 && phi_bin_idx != (std::size_t)-1)
        {
            px = pt * std::cos(phi);
            py = pt * std::sin(phi);
            pz = pt * std::sinh(eta);
        }
        else if (energy_bin_idx != (std::size_t)-1 && theta_bin_idx != (std::size_t)-1 && phi_bin_idx != (std::size_t)-1)
        {
            px = energy * std::sin(theta) * std::cos(phi);
            py = energy * std::sin(theta) * std::sin(phi);
            pz = energy * std::cos(theta);
        }
        else
            throw std::runtime_error("pTokenizer: untokenizer: Cannot calculate linear momentum.");

        if (energy_bin_idx == (std::size_t)-1)
        {
            //@TODO: find a way to get the mass from the pdgid. Particle in python can do this, but not sure how to do it in C++.
            double mass = 0.0f;
            energy = std::sqrt(px * px + py * py + pz * pz + mass * mass);
        }

        untokenized_event.push_back(pdgid);
        untokenized_event.push_back(energy);
        untokenized_event.push_back(px);
        untokenized_event.push_back(py);
        untokenized_event.push_back(pz);
    }
    return untokenized_event;
}
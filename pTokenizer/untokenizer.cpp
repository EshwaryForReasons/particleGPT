
#include "main.h"
#include "dictionary.h"
#include "utils.h"

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

void Untokenizer::untokenize_particles(const std::size_t start_idx, const std::size_t end_idx)
{
    std::size_t current_idx = 0;
    for (auto& event : data_manager->tokenized_data | std::views::drop(start_idx) | std::views::take(end_idx - start_idx + 1))
    {
        std::vector<double> untokenized_event;
        for (int i = 0; i < event.size(); i += 5)
        {
            int pdgid_idx = event[i] - dictionary->offsets.pdgid_offset;
            int energy_idx = event[i + 1] - dictionary->offsets.energy_offset;
            int eta_idx = event[i + 2] - dictionary->offsets.eta_offset;
            int theta_idx = event[i + 3] - dictionary->offsets.theta_offset;
            int phi_idx = event[i + 4] - dictionary->offsets.phi_offset;

            int pdgid = 0;
            for (auto& [pdg_id, pdg_idx] : dictionary->pdgid_to_index)
            {
                if (pdg_idx == pdgid_idx)
                {
                    pdgid = pdg_id;
                    break;
                }
            }
            double energy = pMath::get_bin_median(dictionary->e_bins, energy_idx);
            double eta = pMath::get_bin_median(dictionary->eta_bins, eta_idx);
            double theta = pMath::get_bin_median(dictionary->theta_bins, theta_idx);
            double phi = pMath::get_bin_median(dictionary->phi_bins, phi_idx);

            double px = energy * std::sin(theta) * std::cos(phi);
            double py = energy * std::sin(theta) * std::sin(phi);
            double pz = energy * std::cos(theta);

            untokenized_event.push_back(pdgid);
            untokenized_event.push_back(energy);
            untokenized_event.push_back(px);
            untokenized_event.push_back(py);
            untokenized_event.push_back(pz);
        }

        data_manager->raw_data[start_idx + current_idx] = untokenized_event;
        ++current_idx;
    }
}

void Untokenizer::untokenize_data(const std::string& output_data_path)
{
    std::printf("----------------------------------------\n");

    if (data_manager->tokenized_data.size() == 0)
    {
        throw std::runtime_error("pTokenizer: untokenizer: Tokenized data not loaded.");
    }

    //Unokenize data

    std::printf("pTokenizer: untokenizer: Began untokenizing data.\n");
    data_manager->raw_data.resize(data_manager->tokenized_data.size());
    //Profiling shows no reason to multithread this one (more threads was actually slower?)
    untokenize_particles(0, data_manager->tokenized_data.size() - 1);
    std::printf("pTokenizer: untokenizer: Finished untokenizing data.\n");

    //Output untokenized data

    std::printf("pTokenizer: untokenizer: Began outputting untokenized data.\n");

    std::ofstream output_file(output_data_path);
    for (auto event : data_manager->raw_data)
    {
        for (int i = 0; i < event.size(); i += 5)
        {
            output_file << std::format("{:d} {:.5f} {:.5f} {:.5f} {:.5f};", (int)event[i], event[i + 1], event[i + 2], event[i + 3], event[i + 4]);
        }
        output_file << "\n";
    }

    std::printf("pTokenizer: untokenizer: Finished outputting untokenized data.\n");
    std::printf("----------------------------------------\n");
}

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

void Tokenizer::tokenize_particles(const std::size_t start_idx, const std::size_t end_idx)
{
    std::size_t current_idx = 0;
    for (auto& event : data_manager->raw_data | std::views::drop(start_idx) | std::views::take(end_idx - start_idx + 1))
    {
        std::vector<int> tokenized_event = { dictionary->special_tokens.event_start };
        bool b_use_event = true;
        // As opposed to padding
        bool b_tokenizing_secondaries = true;

        for (int particle_idx = 0; particle_idx < event.size() / 5; ++particle_idx)
        {
            double pdgid = event[particle_idx * 5];
            if (pdgid == -1)
            {
                if (b_tokenizing_secondaries)
                {
                    tokenized_event.push_back(dictionary->special_tokens.event_end);
                    b_tokenizing_secondaries = false;
                }
                tokenized_event.insert(tokenized_event.end(), { dictionary->special_tokens.particle_start, 0, 0, 0, 0, 0, dictionary->special_tokens.particle_end });
                continue;
            }

            double energy = event[particle_idx * 5 + 1];
            double px = event[particle_idx * 5 + 2];
            double py = event[particle_idx * 5 + 3];
            double pz = event[particle_idx * 5 + 4];

            double r = std::sqrt(px * px + py * py + pz * pz);
            double theta = std::acos(pz / r);
            double phi = std::atan2(py, px);
            double eta = -std::log(std::tan(theta / 2));

            if (std::abs(eta) > 4)
            {
                b_use_event = false;
                break;
            }

            int particle_index = 0;
            for (auto& [i_pdgid, i_index] : dictionary->pdgid_to_index)
            {
                if (i_pdgid == pdgid)
                {
                    particle_index = i_index;
                    break;
                }
            }

            tokenized_event.push_back(dictionary->special_tokens.particle_start);
            tokenized_event.push_back(particle_index + dictionary->offsets.pdgid_offset);
            tokenized_event.push_back(pMath::digitize(energy, dictionary->e_bins) + dictionary->offsets.energy_offset);
            tokenized_event.push_back(pMath::digitize(eta, dictionary->eta_bins) + dictionary->offsets.eta_offset);
            tokenized_event.push_back(pMath::digitize(theta, dictionary->theta_bins) + dictionary->offsets.theta_offset);
            tokenized_event.push_back(pMath::digitize(phi, dictionary->phi_bins) + dictionary->offsets.phi_offset);
            tokenized_event.push_back(dictionary->special_tokens.particle_end);
        }

        //In the case that this is the longest event this will have not padding which means the event end token will have never been added
        if (b_tokenizing_secondaries)
        {
            tokenized_event.push_back(dictionary->special_tokens.event_end);
        }

        if (b_use_event)
            data_manager->tokenized_data[start_idx + current_idx] = tokenized_event;

        ++current_idx;
    }
}

void Tokenizer::tokenize_data(const std::string& output_data_path)
{
    std::printf("----------------------------------------\n");

    if (data_manager->raw_data.size() == 0)
    {
        throw std::runtime_error("pTokenizer: tokenizer: Raw data not loaded.");
    }

    //Add padding

    std::printf("pTokenizer: tokenizer: Began padding data.\n");
    std::size_t num_tokens_in_largest_event = std::ranges::max(data_manager->raw_data, {}, &std::vector<double>::size).size();
    for (auto& v : data_manager->raw_data)
        v.resize(num_tokens_in_largest_event, -1);
    std::printf("pTokenizer: tokenizer: Finished padding data.\n");

    //Tokenize data

    std::printf("pTokenizer: tokenizer: Began tokenizing data.\n");

    data_manager->tokenized_data.resize(data_manager->raw_data.size());

    std::size_t num_threads = std::thread::hardware_concurrency();
    std::size_t events_per_thread = data_manager->raw_data.size() / num_threads;

    std::printf("Number of threads: %zu\n", num_threads);
    std::printf("Events per thread: %zu\n", events_per_thread);

    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < num_threads; ++i)
    {
        std::size_t start_idx = i * events_per_thread;
        std::size_t end_idx = ((i + 1) * events_per_thread) - 1;
        if (i == num_threads - 1)
        {
            end_idx = data_manager->raw_data.size() - 1;
        }

        threads.push_back(std::thread(&Tokenizer::tokenize_particles, start_idx, end_idx));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    std::printf("pTokenizer: tokenizer: Finished tokenizing data.\n");

    //Remove empty events (those that we ignore and thus do not bother tokenizing)

    auto is_vec_empty = [](const std::vector<int>& v) {return v.empty();};
    data_manager->tokenized_data.erase(std::remove_if(data_manager->tokenized_data.begin(), data_manager->tokenized_data.end(), is_vec_empty), data_manager->tokenized_data.end());

    //Output tokenized data

    std::printf("pTokenizer: tokenizer: Began outputting data.\n");
    std::ofstream output_file(output_data_path);
    for (int i = 0; i < data_manager->tokenized_data.size(); ++i)
    {
        auto event = data_manager->tokenized_data[i];
        for (int j = 0; j < event.size(); ++j)
        {
            output_file << event[j];
            if (j != event.size() - 1)
                output_file << " ";
        }

        if (i != data_manager->tokenized_data.size() - 1)
            output_file << "\n";
    }

    std::printf("pTokenizer: tokenizer: Finished outputting data.\n");
    std::printf("----------------------------------------\n");
}
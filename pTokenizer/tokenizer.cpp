
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
    for (auto& event : input_data | std::views::drop(start_idx) | std::views::take(end_idx - start_idx + 1))
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


        if (b_use_event)
            tokenized_data[start_idx + current_idx] = tokenized_event;

        ++current_idx;
    }
}

void Tokenizer::tokenize_data(const std::string& input_data_path, const std::string& output_data_path)
{
    //input_data_path is expected to be raw data from MCGenerators

    //Load data

    std::ifstream input_data_file(input_data_path);
    std::string event;
    int event_counter = 0;
    while (std::getline(input_data_file, event))
    {
        input_data.push_back(std::vector<double>());
        std::stringstream particle_stream(event);
        std::string particle;
        while (std::getline(particle_stream, particle, ';'))
        {
            std::stringstream token_stream(particle);
            std::string token;
            while (std::getline(token_stream, token, ' '))
            {
                double value = std::stod(token);
                input_data[event_counter].push_back(value);
            }
        }

        event_counter++;
    }

    //Add padding

    std::size_t num_tokens_in_largest_event = std::ranges::max(input_data, {}, &std::vector<double>::size).size();
    for (auto& v : input_data)
        v.resize(num_tokens_in_largest_event, -1);

    //Tokenize data

    tokenized_data.resize(input_data.size());

    std::size_t num_threads = std::thread::hardware_concurrency();
    std::size_t events_per_thread = input_data.size() / num_threads;

    std::printf("Number of threads: %zu\n", num_threads);
    std::printf("Events per thread: %zu\n", events_per_thread);

    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < num_threads; ++i)
    {
        std::size_t start_idx = i * events_per_thread;
        std::size_t end_idx = ((i + 1) * events_per_thread) - 1;
        if (i == num_threads - 1)
        {
            end_idx = input_data.size() - 1;
        }

        threads.push_back(std::thread(&Tokenizer::tokenize_particles, this, start_idx, end_idx));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    //Output tokenized data

    std::ofstream output_file(output_data_path);
    for (auto event : tokenized_data | std::views::filter([](const auto& v) { return v.size() > 0; }))
    {
        for (int i = 0; i < event.size(); ++i)
        {
            output_file << event[i];
            if (i != event.size() - 1)
                output_file << " ";
        }
        output_file << "\n";
    }
}
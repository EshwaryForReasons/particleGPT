
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
#include <filesystem>
#include <format>

Dictionary dictionary;

const std::size_t get_free_memory_size()
{
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    size_t free_kb = 0;

    while (std::getline(meminfo, line))
    {
        if (line.find("MemAvailable:") == 0)
        {
            std::sscanf(line.c_str(), "MemAvailable: %zu kB", &free_kb);
            break;
        }
    }

    return free_kb * 1024;
}

const auto analyze_dataset(const std::string& filename)
{
    struct AnalysisResult
    {
        std::size_t num_events;
        std::size_t num_particles_max;
    };

    const char delimiter = ';';

    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Could not open file: " + filename);

    std::string line;
    size_t num_rows = 0;
    size_t max_cols = 0;

    while (std::getline(file, line)) 
    {
        ++num_rows;
        size_t col_count = 1;
        for (char ch : line)
            if (ch == delimiter)
                ++col_count;
        max_cols = std::max(max_cols, col_count);
    }

    return AnalysisResult{.num_events = num_rows, .num_particles_max = max_cols};
}

void Tokenizer::tokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path)
{
    std::printf("----------------------------------------\n");

    dictionary = DataManager::load_dictionary(dictionary_path);

    //This code probably wont work if num_events < num_threads.
    const std::size_t four_gigs = (std::size_t)4 * 1024 * 1024 * 1024;
    const std::size_t ninety_six_gigs = (std::size_t)96 * 1024 * 1024 * 1024;
    const std::size_t available_memory = std::min(get_free_memory_size() - four_gigs, ninety_six_gigs);

    //Step one is to determine the number of events in the data
    const auto [total_num_events, num_particles_max] = analyze_dataset(input_data_path);

    //We want to use all possible threads and allocate events such that we do not exceed the available memory
    const std::size_t num_threads = std::thread::hardware_concurrency();
    //Rough calculation, but it should handle all cases
    const std::size_t memory_usage_per_event = (num_particles_max * 7) * (sizeof(double) + sizeof(int));
    //Assume we use num_threads
    const std::size_t max_events_per_thread = available_memory / (memory_usage_per_event * num_threads);

    const std::size_t required_threads = std::ceil(static_cast<double>(total_num_events) / static_cast<double>(max_events_per_thread));
    const std::size_t events_per_thread = std::min<std::size_t>(
        max_events_per_thread,
        std::ceil(static_cast<double>(total_num_events) / static_cast<double>(required_threads))
    );

    //Tokenize data

    std::printf("pTokenizer: tokenizer: Began tokenizing data.\n");
    std::printf("Input data file path: %s\n", input_data_path.c_str());
    std::printf("Available memory: %zu GB\n", available_memory / 1024 / 1024 / 1024);
    std::printf("Number of hardware threads: %zu\n", num_threads);
    std::printf("Number of required threads: %zu\n", required_threads);
    std::printf("Events per thread: %zu\n", events_per_thread);

    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < required_threads; ++i)
    {
        std::size_t start_idx = i * events_per_thread;
        std::size_t end_idx = ((i + 1) * events_per_thread) - 1;
        if (i == required_threads - 1)
        {
            end_idx = total_num_events - 1;
        }
        threads.push_back(std::thread(&tokenize_events_in_range, input_data_path, output_data_path, num_particles_max, start_idx, end_idx, i));

        const std::size_t running_threads = threads.size();
        //TODO: ideally we would handle this on a thread by thread basis.
        //The wait can't be too bad but still should implement some sort of thread by thread thing.
        if (running_threads >= num_threads)
        {
            for (auto& thread : threads)
                thread.join();
            threads.clear();
        }
    }

    //Wait for remaining threads
    for (auto& thread : threads)
        thread.join();

    std::printf("pTokenizer: tokenizer: Finished tokenizing data.\n");

    std::printf("----------------------------------------\n");
}

void Tokenizer::tokenize_events_in_range(const std::string& input_data_path, const std::string& output_data_path, const std::size_t num_particles_max, const std::size_t start_idx, const std::size_t end_idx, const std::size_t idx)
{
    std::ifstream input_data_file(input_data_path);

    //Tokenize data
    
    std::vector<std::vector<int>> tokenized_events;
    std::string event;
    std::vector<double> input_events;
    for (std::size_t i = 0; i <= end_idx; ++i)
    {
        std::getline(input_data_file, event);
        
        //Skip to start_idx
        if (i < start_idx)
            continue;
        
        std::replace(event.begin(), event.end(), ';', ' ');
        const std::vector<std::string_view> event_split_str = Utils::split(event, ' ');
        std::vector<double> event_split;
        for (auto event_split_single : event_split_str)
            event_split.push_back(std::stod(event_split_single.data()));
        
        const std::vector<int> tokenized_event = tokenize_event(event_split);
        if (!tokenized_event.empty())
            tokenized_events.push_back(tokenized_event);
    }

    //Pad data

    const auto padding_sequence = dictionary.get_padding_sequence();
    const std::size_t max_sequence_length = num_particles_max * dictionary.get_num_tokens_per_particle() + 2;
    for (auto& tokenized_event : tokenized_events)
    {
        if (tokenized_event.size() < max_sequence_length)
        {
            const std::size_t num_particles = (tokenized_event.size() - 2) / dictionary.get_num_tokens_per_particle();
            const std::size_t required_delta = num_particles_max - num_particles;
            for (std::size_t i = 0; i < required_delta; ++i)
            {
                tokenized_event.insert(tokenized_event.end(), padding_sequence.begin(), padding_sequence.end());
            }
        }
        else if (tokenized_event.size() > max_sequence_length)
        {
            throw std::runtime_error("pTokenizer: Tokenized event is larger than max sequence length.");
        }
    }

    //Output data

    std::filesystem::path output_data_filename = std::filesystem::path(output_data_path) / std::format("tokenized_batch_{}.csv", idx);
    std::ofstream output_data_file(output_data_filename.c_str());
    for (const auto& tokenized_event : tokenized_events)
    {
        for (std::size_t i = 0; i < tokenized_event.size(); ++i)
        {
            output_data_file << tokenized_event[i];
            if (i != tokenized_event.size() - 1)
                output_data_file << " ";
        }
        output_data_file << "\n";
    }
}

const std::vector<int> Tokenizer::tokenize_event(const std::vector<double>& event)
{
    std::vector<int> tokenized_event = { dictionary.special_tokens.event_start };
    for (int particle_idx = 0; particle_idx < event.size() / 5; ++particle_idx)
    {
        double pdgid    = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 0];
        double energy   = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 1];
        double px       = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 2];
        double py       = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 3];
        double pz       = event[particle_idx * NUM_FEATURES_PER_PARTICLE_RAW + 4];

        double r        = std::sqrt(px * px + py * py + pz * pz);
        double pt       = std::sqrt(px * px + py * py);
        double theta    = std::acos(pz / r);
        double phi      = std::atan2(py, px);
        double eta      = -std::log(std::tan(theta / 2));

        //Eta constrain
        if (std::abs(eta) > 4)
            return {};

        int particle_index = 0;
        for (auto& [i_pdgid, i_index] : dictionary.pdgid_to_index)
        {
            if (i_pdgid == pdgid)
            {
                particle_index = i_index;
                break;
            }
        }

        for (int i = 0; i < dictionary.get_num_tokens_per_particle(); ++i)
        {
            if (dictionary.tokenization_schema[i] == "pdgid")
                tokenized_event.push_back(particle_index + dictionary.offsets.pdgid_offset);
            else if (dictionary.tokenization_schema[i] == "pt")
                tokenized_event.push_back(pMath::digitize(pt, dictionary.pt_bins) + dictionary.offsets.pt_offset);
            else if (dictionary.tokenization_schema[i] == "energy")
                tokenized_event.push_back(pMath::digitize(energy, dictionary.e_bins) + dictionary.offsets.energy_offset);
            else if (dictionary.tokenization_schema[i] == "eta")
                tokenized_event.push_back(pMath::digitize(eta, dictionary.eta_bins) + dictionary.offsets.eta_offset);
            else if (dictionary.tokenization_schema[i] == "theta")
                tokenized_event.push_back(pMath::digitize(theta, dictionary.theta_bins) + dictionary.offsets.theta_offset);
            else if (dictionary.tokenization_schema[i] == "phi")
                tokenized_event.push_back(pMath::digitize(phi, dictionary.phi_bins) + dictionary.offsets.phi_offset);
            else if (dictionary.tokenization_schema[i] == "particle_start")
                tokenized_event.push_back(dictionary.special_tokens.particle_start);
            else if (dictionary.tokenization_schema[i] == "particle_end")
                tokenized_event.push_back(dictionary.special_tokens.particle_end);
            else
                throw std::runtime_error("pTokenizer: Unknown tokenization schema: " + dictionary.tokenization_schema[i]);
        }
    }

    tokenized_event.push_back(dictionary.special_tokens.event_end);
    return tokenized_event;
}
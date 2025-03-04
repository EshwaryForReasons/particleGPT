
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <ranges>
#include <algorithm>
#include <thread>

#include "json.hpp"

std::string dataset_path = "";

std::vector<std::vector<double>> input_data;
std::vector<std::vector<int>> tokenized_data;

struct FSpecialTokens {
    int padding = -4;
    int event_start = -4;
    int event_end = -4;
    int particle_start = -4;
    int particle_end = -4;
} special_tokens;

struct FOffsets {
    int special_tokens_offset = 0;
    int pdgid_offset = 0;
    int materials_offset = 0;
    int energy_offset = 0;
    int eta_offset = 0;
    int theta_offset = 0;
    int phi_offset = 0;
} offsets;

struct pdgid_index_pair
{
    int pdgid = 0;
    int index = 0;
};

std::vector<double> e_bins;
std::vector<double> eta_bins;
std::vector<double> theta_bins;
std::vector<double> phi_bins;
std::vector<pdgid_index_pair> pdgid_to_index;

//Replica of numpy.arange
std::vector<double> arange(double start, double stop, double step)
{
    std::vector<double> result;
    for (double i = start; i < stop; i += step)
    {
        result.push_back(i);
    }
    return result;
}

//Replica of numpy.digitize
int digitize(double value, const std::vector<double>& bins)
{
    for (int i = 1; i <= bins.size(); ++i)
    {
        if (value >= bins[i - 1] && value < bins[i])
        {
            return i;
        }
    }
    
    return -4;
}

void init_dictionary()
{
    //Load in dictionary
    std::ifstream dictionary_file(dataset_path + "/dictionary.json");
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
                pdgid_index_pair pair = {std::stoi(particle_id), particle_index};
                pdgid_to_index.push_back(pair);
                break;
            }
        }
    }

    //Generate bins

    double e_min = dictionary_json["e_bin_data"]["min"];
    double e_max = dictionary_json["e_bin_data"]["max"];
    double e_step = dictionary_json["e_bin_data"]["step_size"];
    e_bins = arange(e_min, e_max, e_step);

    double eta_min = dictionary_json["eta_bin_data"]["min"];
    double eta_max = dictionary_json["eta_bin_data"]["max"];
    double eta_step = dictionary_json["eta_bin_data"]["step_size"];
    eta_bins = arange(eta_min, eta_max, eta_step);

    double theta_min = -2 * M_PI;
    double theta_max = 2 * M_PI;
    double theta_step = dictionary_json["theta_bin_data"]["step_size"];
    theta_bins = arange(theta_min, theta_max, theta_step);

    double phi_min = -2 * M_PI;
    double phi_max = 2 * M_PI;
    double phi_step = dictionary_json["phi_bin_data"]["step_size"];
    phi_bins = arange(phi_min, phi_max, phi_step);

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

void tokenize_data(std::size_t start_idx, std::size_t end_idx)
{
    std::size_t current_idx = 0;
    for (auto& event : input_data | std::views::drop(start_idx) | std::views::take(end_idx - start_idx + 1))
    {
        std::vector<int> tokenized_event = {special_tokens.event_start};
        bool b_use_event = true;

        for (int particle_idx = 0; particle_idx < event.size() / 5; ++particle_idx)
        {
            double pdgid = event[particle_idx * 5];
            if (pdgid == -1)
            {
                tokenized_event.insert(tokenized_event.end(), {special_tokens.particle_start, 0, 0, 0, 0, 0, special_tokens.particle_end});
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
            for (auto& [i_pdgid, i_index] : pdgid_to_index)
            {
                if (i_pdgid == pdgid)
                {
                    particle_index = i_index;
                    break;
                }
            }
            
            tokenized_event.push_back(special_tokens.particle_start);
            tokenized_event.push_back(particle_index + offsets.pdgid_offset);
            tokenized_event.push_back(digitize(energy, e_bins) + offsets.energy_offset);
            tokenized_event.push_back(digitize(eta, eta_bins) + offsets.eta_offset);
            tokenized_event.push_back(digitize(theta, theta_bins) + offsets.theta_offset);
            tokenized_event.push_back(digitize(phi, phi_bins) + offsets.phi_offset);
            tokenized_event.push_back(special_tokens.particle_end);
        }

        tokenized_event.push_back(special_tokens.event_end);
 
        if (b_use_event)
            tokenized_data[start_idx + current_idx] = tokenized_event;

        ++current_idx;
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Dataset not provided to tokenizer. Exiting..." << std::endl;
        return 0;
    }

    dataset_path = argv[1];

    init_dictionary();

    //Load data

    std::ifstream input_data_file(dataset_path + "/data.csv");
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

        threads.push_back(std::thread(tokenize_data, start_idx, end_idx));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    //Output tokenized data
    
    std::ofstream output_file(dataset_path + "/outputs/temp_tokenized.csv");
    for (auto event : tokenized_data | std::views::filter([](const auto& v) { return v.size() > 0; }))
    {
        for (int i = 0; i < event.size(); ++i)
        {
            output_file << event[i];
            if (i != event.size() - 1)
                output_file << " ";
        }
        // for (auto token : event)
        // {
        //     output_file << token << " ";
        // }
        output_file << "\n";
    }

    return 0;
}
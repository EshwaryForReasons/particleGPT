
#include "main.h"
#include "dictionary.h"

#include <pybind11/pybind11.h>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <format>

Dictionary* dictionary = nullptr;
DataManager* data_manager = nullptr;

void load_dictionary(std::string dictionary_path)
{
    if (!std::filesystem::exists(dictionary_path))
    {
        std::cout << "pTokenizer: Dictionary not found at " << dictionary_path << std::endl;
        throw std::runtime_error("pTokenizer: Dictionary not found.");
    }

    std::cout << "pTokenizer: Found dictionary at " << dictionary_path << std::endl;
    dictionary = new Dictionary(dictionary_path);

    if (!dictionary)
    {
        std::cout << "pTokenizer: Failed to load dictionary." << std::endl;
        throw std::runtime_error("pTokenizer: Failed to load dictionary.");
    }

    std::cout << "pTokenizer: Dictionary successfully loaded." << std::endl;
}

//We assume raw data has the structure "pdgid energy px py pz; pdgid energy px py pz; ..."
//Basically what MCGenerators generates
void load_raw_data(std::string input_data_path)
{
    if (!std::filesystem::exists(input_data_path))
    {
        std::cout << "pTokenizer: Raw data not found at " << input_data_path << std::endl;
        throw std::runtime_error("pTokenizer: Raw data not found.");
    }
    
    std::printf("pTokenizer: Began loading raw data.\n");
    std::ifstream input_data_file(input_data_path);
    std::string event;
    int event_counter = 0;
    while (std::getline(input_data_file, event))
    {
        data_manager->raw_data.push_back(std::vector<double>());
        std::stringstream particle_stream(event);
        std::string particle;
        while (std::getline(particle_stream, particle, ';'))
        {
            std::stringstream token_stream(particle);
            std::string token;
            while (std::getline(token_stream, token, ' '))
            {
                double value = std::stod(token);
                data_manager->raw_data[event_counter].push_back(value);
            }
        }

        event_counter++;
    }
    std::printf("pTokenizer: Finished loading raw data.\n");
}

//We assume tokenized data has the structure
//"event_start particle_start pdgid energy eta theta phi particle_end particle_start pdgid energy eta theta phi particle_end ... event_end"
//Generally this will be the generated samples from the model
void load_tokenized_data(std::string input_data_path)
{
    if (!std::filesystem::exists(input_data_path))
    {
        std::cout << "pTokenizer: Tokenized data not found at " << input_data_path << std::endl;
        throw std::runtime_error("pTokenizer: Tokenized data not found.");
    }
    
    std::printf("pTokenizer: untokenizer: Began loading data.\n");
    std::ifstream input_data_file(input_data_path);
    std::string event;
    int event_counter = 0;
    while (std::getline(input_data_file, event))
    {
        data_manager->tokenized_data.push_back(std::vector<int>());
        std::stringstream particle_stream(event);
        std::string token;
        while (std::getline(particle_stream, token, ' '))
        {
            int value = std::stoi(token);
            data_manager->tokenized_data[event_counter].push_back(value);
        }

        event_counter++;
    }
    std::printf("pTokenizer: untokenizer: Finished loading data.\n");
}

//These will output the tokenized events into .bin files for training, validation, and testing
//The number of events will be done from the start of the file in the order: train, then val, then test
void output_split_bins(std::size_t num_train_events, std::size_t num_val_events, std::size_t num_test_events,
                        std::string train_bin_path, std::string val_bin_path, std::string test_bin_path)
{
    std::printf("----------------------------------------\n");
    std::printf("pTokenizer: output_split_bins: Began.\n");
    if (data_manager->tokenized_data.empty())
    {
        throw std::runtime_error("pTokenizer: No tokenized data to split into bins.");
    }

    //Convert data to std::uint16_t and flatten
    std::vector<std::uint16_t> tokenized_data_flat;
    auto conv_to_uint16 = [](int value) {return static_cast<std::uint16_t>(value);};
    for (const auto& inner_vec : data_manager->tokenized_data)
    {
        std::transform(inner_vec.begin(), inner_vec.end(), std::back_inserter(tokenized_data_flat), conv_to_uint16);
    }

    std::ofstream train_bin_file(train_bin_path, std::ios::binary);
    std::ofstream val_bin_file(val_bin_path, std::ios::binary);
    std::ofstream test_bin_file(test_bin_path, std::ios::binary);
    const std::size_t num_tokens_per_event = data_manager->tokenized_data[0].size();
    for (std::size_t i = 0; i < num_train_events * num_tokens_per_event; ++i)
    {
        train_bin_file.write(reinterpret_cast<const char*>(&tokenized_data_flat[i]), sizeof(std::uint16_t));
    }
    for (std::size_t i = num_train_events * num_tokens_per_event; i < (num_train_events + num_val_events) * num_tokens_per_event; ++i)
    {
        val_bin_file.write(reinterpret_cast<const char*>(&tokenized_data_flat[i]), sizeof(std::uint16_t));
    }
    for (std::size_t i = (num_train_events + num_val_events) * num_tokens_per_event; i < (num_train_events + num_val_events + num_test_events) * num_tokens_per_event; ++i)
    {
        test_bin_file.write(reinterpret_cast<const char*>(&tokenized_data_flat[i]), sizeof(std::uint16_t));
    }

    data_manager->num_train_tokens = num_train_events * num_tokens_per_event;
    data_manager->num_val_tokens = num_val_events * num_tokens_per_event;
    data_manager->num_test_tokens = num_test_events * num_tokens_per_event;
    std::printf("pTokenizer: output_split_bins: Finished.\n");
    std::printf("----------------------------------------\n");
}

//Leading particle information for the raw data (that generated by MCGenerators)
//Will be of the form "num_particles pdgid energy px py pz eta theta phi"
//The particle information will only be for the leading particle (the one with the highest energy)
void output_real_leading_particle_information(std::string output_data_path, std::size_t num_test_particles)
{
    if (data_manager->raw_data.empty())
    {
        throw std::runtime_error("pTokenizer: No raw data loaded.");
    }

    auto event_is_invalid = [](const std::vector<double>& event) {
        bool b_valid_so_far = true;
        for (int particle_idx = 0; particle_idx < event.size() / 5; ++particle_idx)
        {
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
                b_valid_so_far = false;
                break;
            }
        }
        return !b_valid_so_far;
    };

    std::vector<std::vector<double>> validated_raw_data = data_manager->raw_data;
    validated_raw_data.erase(std::remove_if(validated_raw_data.begin(), validated_raw_data.end(), event_is_invalid), validated_raw_data.end());

    std::vector<std::vector<double>> test_events(validated_raw_data.end() - num_test_particles, validated_raw_data.end());
    //Inner vector will be of form: num_particles pdgid energy px py pz eta theta phi
    std::vector<std::vector<double>> leading_particles;
    leading_particles.reserve(test_events.size());
    for (const auto& event : test_events)
    {
        double max_energy = 0;
        int max_energy_idx = 0;
        //Start at 1 because 0 is primary particle which will always have the highest energy
        for (int particle_idx = 1; particle_idx < event.size() / 5; ++particle_idx)
        {
            double energy = event[particle_idx * 5 + 1];
            if (energy > max_energy)
            {
                max_energy = energy;
                max_energy_idx = particle_idx;
            }
        }

        /**
         * The input data may be padded with -1 tokens, we need to eliminate those to calculate num_secondaries accurately
         * 
         * @TODO: handling this here is a morally terrible solution but there are only two other options:
         * 1. Change the tokenizer to pad a copy of the input data or
         * 2. Change the tokenizer to tokenize the input data in place
         * 
         * 1 is not ideal because of memory usage and 2 because I don't feel like changing the tokenizer
         * right now. If needed we will make that change.
         */
        const std::size_t num_secondaries = std::ranges::count_if(event, [](double val) { return val != -1; }) / 5 - 1;
        double pdgid = event[max_energy_idx * 5];
        double energy = event[max_energy_idx * 5 + 1];
        double px = event[max_energy_idx * 5 + 2];
        double py = event[max_energy_idx * 5 + 3];
        double pz = event[max_energy_idx * 5 + 4];

        double r = std::sqrt(px * px + py * py + pz * pz);
        double theta = std::acos(pz / r);
        double phi = std::atan2(py, px);
        double eta = -std::log(std::tan(theta / 2));

        leading_particles.emplace_back(std::vector<double>{static_cast<double>(num_secondaries), pdgid, energy, px, py, pz, eta, theta, phi});
    }

    std::ofstream output_data_file(output_data_path);
    for (const auto& leading_particle : leading_particles)
    {
        std::size_t num_secondaries = leading_particle[0];
        int pdgid = leading_particle[1];
        double energy = leading_particle[2];
        double px = leading_particle[3];
        double py = leading_particle[4];
        double pz = leading_particle[5];
        double eta = leading_particle[6];
        double theta = leading_particle[7];
        double phi = leading_particle[8];
        output_data_file << std::format("{:d} {:d} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n", num_secondaries, pdgid, energy, px, py, pz, eta, theta, phi);
    }
}

//Leading particle information for the generated data (that generated by MCGenerators)
//Will be of the form "num_particles pdgid energy px py pz eta theta phi"
//The particle information will only be for the leading particle (the one with the highest energy)
void output_generated_leading_particle_information(std::string output_data_path)
{
    //We use raw_data as this function will only be ran for untokenized data
    if (data_manager->raw_data.empty())
    {
        throw std::runtime_error("pTokenizer: No tokenized data loaded.");
    }

    //IMPORTANT: We do not filter or validate anything here! Doing so before calling this function is a choice left to the user.

    //Inner vector will be of form: num_particles pdgid energy px py pz eta theta phi
    std::vector<std::vector<double>> leading_particles;
    leading_particles.reserve(data_manager->raw_data.size());
    for (const auto& event : data_manager->raw_data)
    {
        double max_energy = 0;
        int max_energy_idx = 0;
        //Start at 1 because 0 is primary particle which will always have the highest energy
        for (int particle_idx = 1; particle_idx < event.size() / 5; ++particle_idx)
        {
            double energy = event[particle_idx * 5 + 1];
            if (energy > max_energy)
            {
                max_energy = energy;
                max_energy_idx = particle_idx;
            }
        }

        /**
         * The input data may be padded with -1 tokens, we need to eliminate those to calculate num_secondaries accurately
         * 
         * @TODO: handling this here is a morally terrible solution but there are only two other options:
         * 1. Change the tokenizer to pad a copy of the input data or
         * 2. Change the tokenizer to tokenize the input data in place
         * 
         * 1 is not ideal because of memory usage and 2 because I don't feel like changing the tokenizer
         * right now. If needed we will make that change.
         */
        const std::size_t num_secondaries = std::ranges::count_if(event, [](double val) { return val != -1; }) / 5 - 1;
        double pdgid = event[max_energy_idx * 5];
        double energy = event[max_energy_idx * 5 + 1];
        double px = event[max_energy_idx * 5 + 2];
        double py = event[max_energy_idx * 5 + 3];
        double pz = event[max_energy_idx * 5 + 4];

        double r = std::sqrt(px * px + py * py + pz * pz);
        double theta = std::acos(pz / r);
        double phi = std::atan2(py, px);
        double eta = -std::log(std::tan(theta / 2));

        leading_particles.emplace_back(std::vector<double>{static_cast<double>(num_secondaries), pdgid, energy, px, py, pz, eta, theta, phi});
    }

    std::ofstream output_data_file(output_data_path);
    for (const auto& leading_particle : leading_particles)
    {
        std::size_t num_secondaries = leading_particle[0];
        int pdgid = leading_particle[1];
        double energy = leading_particle[2];
        double px = leading_particle[3];
        double py = leading_particle[4];
        double pz = leading_particle[5];
        double eta = leading_particle[6];
        double theta = leading_particle[7];
        double phi = leading_particle[8];
        output_data_file << std::format("{:d} {:d} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f} {:.5f}\n", num_secondaries, pdgid, energy, px, py, pz, eta, theta, phi);
    }
}

PYBIND11_MODULE(pTokenizerModule, m)
{
    if (!data_manager)
    {
        data_manager = new DataManager();
    }

    m.doc() = "Backend for tokenization and untokenization of data.";
    m.def("load_dictionary", &load_dictionary);
    m.def("load_raw_data", &load_raw_data);
    m.def("load_tokenized_data", &load_tokenized_data);

    //TODO: Maybe add checking to these to avoid potential crashes
    m.def("get_num_tokenized_events", []() {return data_manager->tokenized_data.size();});
    m.def("get_num_train_tokens", []() {return data_manager->num_train_tokens;});
    m.def("get_num_val_tokens", []() {return data_manager->num_val_tokens;});
    m.def("get_num_test_tokens", []() {return data_manager->num_test_tokens;});
    m.def("get_max_sequence_length", []() {return data_manager->tokenized_data[0].size();});

    m.def("tokenize_data", &Tokenizer::tokenize_data);
    m.def("untokenize_data", &Untokenizer::untokenize_data);
    m.def("filter_data", &Filter::filter_data);
    m.def("output_split_bins", &output_split_bins);
    m.def("output_real_leading_particle_information", &output_real_leading_particle_information);
    m.def("output_generated_leading_particle_information", &output_generated_leading_particle_information);
}
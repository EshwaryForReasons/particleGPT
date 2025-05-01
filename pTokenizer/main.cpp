
#include "main.h"
#include "utils.h"
#include "dictionary.h"

#include "json.hpp"

#include <pybind11/pybind11.h>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <format>

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

void tokenize_data_router(std::string dictionary_path, std::string scheme, std::string input_data_path, std::string output_data_path)
{
    if (scheme == "standard")
        SchemeStandard::tokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "no_eta")
        SchemeNoEta::tokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "no_particle_boundaries")
        SchemeNoParticleBoundaries::tokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "paddingv2")
        SchemePaddingV2::tokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "neo_no_particle_boundaries")
        SchemeNeoNoParticleBoundaries::tokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "neov2")
        SchemeNeoV2::tokenize_data(dictionary_path, input_data_path, output_data_path);
    else
        throw std::runtime_error("pTokenizer: Invalid scheme specified.");
}

void untokenize_data_router(std::string dictionary_path, std::string scheme, std::string input_data_path, std::string output_data_path)
{
    if (scheme == "standard")
        SchemeStandard::untokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "no_eta")
        SchemeNoEta::untokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "no_particle_boundaries")
        SchemeNoParticleBoundaries::untokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "paddingv2")
        SchemePaddingV2::untokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "neo_no_particle_boundaries")
        SchemeNeoNoParticleBoundaries::untokenize_data(dictionary_path, input_data_path, output_data_path);
    else if (scheme == "neov2")
        SchemeNeoV2::untokenize_data(dictionary_path, input_data_path, output_data_path);
    else
        throw std::runtime_error("pTokenizer: Invalid scheme specified.");
}

PYBIND11_MODULE(pTokenizerModule, m)
{
    m.doc() = "Backend for tokenization and untokenization of data.";
    m.def("tokenize_data", &tokenize_data_router);
    m.def("untokenize_data", &untokenize_data_router);
}
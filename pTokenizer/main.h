
#pragma once

#include <vector>
#include <string>

class Dictionary;

class Tokenizer
{
public:

    static const std::size_t NUM_FEATURES_PER_PARTICLE_RAW = 5;

    static void tokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path);
    static void tokenize_events_in_range(const std::string& input_data_path, const std::string& output_data_path, const std::size_t num_particles_max, const std::size_t start_idx, const std::size_t end_idx, const std::size_t idx);
    static const std::vector<int> tokenize_event(const std::vector<double>& event);

    static void untokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path);
    static const std::vector<double> untokenize_event(const std::vector<int>& event, const Dictionary& dictionary);
};

namespace DataManager
{
    const Dictionary load_dictionary(std::string dictionary_path);
    const std::vector<std::vector<int>> load_tokenized_data(std::string input_data_path);
    void output_tokenized_data(std::string output_file_path, const std::vector<std::vector<int>>& tokenized_data);
    void output_raw_data(std::string output_file_path, const std::vector<std::vector<double>>& raw_data);
};

#pragma once

#include <vector>
#include <string>

struct FSpecialTokens {
    int padding = -4;
    int event_start = -4;
    int event_end = -4;
    int particle_start = -4;
    int particle_end = -4;
};

struct FOffsets {
    int special_tokens_offset = 0;
    int pdgid_offset = 0;
    int materials_offset = 0;
    int energy_offset = 0;
    int eta_offset = 0;
    int theta_offset = 0;
    int phi_offset = 0;
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
    Dictionary(const std::string& dictionary_path);

    std::string dictionary_path;

    FSpecialTokens special_tokens;
    FOffsets offsets;

    std::vector<double> e_bins;
    std::vector<double> eta_bins;
    std::vector<double> theta_bins;
    std::vector<double> phi_bins;
    std::vector<pdgid_index_pair> pdgid_to_index;

    //Includes tokenization in bin ranges, min, max, etc in a human readable format
    void write_humanized_dictionary(const std::string& output_path);
    //Updates the dictionary.json particle list with the particles present in the input data
    void update_particle_list(const std::string& input_data_path);
};

namespace DataManager
{
    void tokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path);
    void untokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path);
    void filter_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path);
    const Dictionary load_dictionary(std::string dictionary_path);
    const std::vector<std::vector<int>> load_tokenized_data(std::string input_data_path);
    void output_tokenized_data(std::string output_file_path, const std::vector<std::vector<int>>& tokenized_data);
    void output_raw_data(std::string output_file_path, const std::vector<std::vector<double>>& raw_data);
};
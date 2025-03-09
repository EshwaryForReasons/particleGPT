
#pragma once

#include <vector>
#include <string>

class Tokenizer
{
public:

    std::vector<std::vector<double>> input_data;
    std::vector<std::vector<int>> tokenized_data;

    void tokenize_particles(const std::size_t start_idx, const std::size_t end_idx);
    void tokenize_data(const std::string& input_data_path, const std::string& output_data_path);
};

class Untokenizer
{
public:

    std::vector<std::vector<int>> input_data;
    std::vector<std::vector<double>> untokenized_data;

    void untokenize_particles(const std::size_t start_idx, const std::size_t end_idx);
    void untokenize_data(const std::string& input_data_path, const std::string& output_data_path);
};

class Filter
{
public:

    std::vector<std::vector<int>> input_data;
    std::vector<std::vector<int>> filtered_data;

    void filter_data(const std::string& input_data_path, const std::string& output_data_path) {}
};

extern class Dictionary* dictionary;
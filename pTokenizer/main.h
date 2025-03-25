
#pragma once

#include <vector>
#include <string>

//We do not want to constantly load the data so this exists to keep data persistently loaded
class DataManager
{
public:

    std::vector<std::vector<double>> raw_data;
    std::vector<std::vector<int>> tokenized_data;

    std::size_t num_train_tokens = 0;
    std::size_t num_val_tokens = 0;
    std::size_t num_test_tokens = 0;
};

namespace Tokenizer
{
    void tokenize_particles(const std::size_t start_idx, const std::size_t end_idx);
    void tokenize_data(const std::string& output_data_path);
};

namespace Untokenizer
{
    void untokenize_particles(const std::size_t start_idx, const std::size_t end_idx);
    void untokenize_data(const std::string& output_data_path);
};

namespace Filter
{
    void filter_particles(const std::size_t start_idx, const std::size_t end_idx);
    void filter_data(const std::string& output_data_path);
};

extern class Dictionary* dictionary;
extern class DataManager* data_manager;
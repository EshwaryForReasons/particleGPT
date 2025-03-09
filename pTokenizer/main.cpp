
#include "main.h"
#include "dictionary.h"

#include <pybind11/pybind11.h>

#include <iostream>

Dictionary* dictionary = nullptr;

void tokenize_stub(std::string dictionary_path, std::string input_data_path, std::string output_data_path)
{
    dictionary = new Dictionary(dictionary_path);
    Tokenizer tokenizer;
    tokenizer.tokenize_data(input_data_path, output_data_path);
}

void untokenize_stub(std::string dictionary_path, std::string input_data_path, std::string output_data_path)
{
    dictionary = new Dictionary(dictionary_path);
    Untokenizer untokenizer;
    untokenizer.untokenize_data(input_data_path, output_data_path);
}

PYBIND11_MODULE(pTokenizerModule, m)
{
    m.doc() = "Backend for tokenization and untokenization of data.";
    m.def("tokenize_data", &tokenize_stub);
    m.def("untokenize_data", &untokenize_stub);
}

// int main(int argc, char* argv[])
// {
//     //We expect arguments in the order: task, dictionary_path, input_data_path, output_data_path
//     if (argc < 4)
//     {
//         std::cout << "The correct arguments were not provided. Exiting..." << std::endl;
//         return 0;
//     }

//     const std::string task = argv[1];
//     const std::string dictionary_path = argv[2];
//     const std::string input_data_path = argv[3];
//     const std::string output_data_path = argv[4];

//     dictionary = new Dictionary(dictionary_path);

//     if (task == "tokenize")
//     {
//         Tokenizer tokenizer;
//         tokenizer.tokenize_data(input_data_path, output_data_path);
//     }
//     else if (task == "untokenize")
//     {
//         Untokenizer untokenizer;
//         untokenizer.untokenize_data(input_data_path, output_data_path);
//     }
//     else if (task == "filter")
//     {
//         Filter filter;
//         filter.filter_data(input_data_path, output_data_path);
//     }
//     else
//     {
//         std::cout << "Invalid task provided. Exiting..." << std::endl;
//         return 0;
//     }

//     //tokenize_data(dataset_path + "/data.csv", dataset_path + "/outputs/temp_tokenized.csv");

//     return 0;
// }

#pragma once

#include <vector>
#include <string>

class Dictionary;

template<typename Derived>
class SchemeBase
{
public:

    static void tokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path);
    static void tokenize_events_in_range(const std::string& input_data_path, const std::string& output_data_path, const std::size_t num_particles_max, const std::size_t start_idx, const std::size_t end_idx, const std::size_t idx);
    static void untokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path);
};

/**
 * Default schemes tokenize into pdgid, energy, eta, theta, phi.
 * NoEta is an exception. 
 */

class SchemeStandard : public SchemeBase<SchemeStandard>
{
public:

    static const std::size_t NUM_TOKENS_PER_PARTICLE = 7;
    static const std::vector<int> get_padding_sequence();
    static const std::vector<int> tokenize_event(const std::vector<double>& event);
    static const std::vector<double> untokenize_event(const std::vector<int>& event, const Dictionary& dictionary);
};

//Scheme no_eta does not include eta.
class SchemeNoEta : public SchemeBase<SchemeNoEta>
{
public:

    static const std::size_t NUM_TOKENS_PER_PARTICLE = 6;
    static const std::vector<int> get_padding_sequence();
    static const std::vector<int> tokenize_event(const std::vector<double>& event);
    static const std::vector<double> untokenize_event(const std::vector<int>& event, const Dictionary& dictionary);
};

//Scheme no_particle_boundaries does not the particle_start and particle_end tokens.
class SchemeNoParticleBoundaries : public SchemeBase<SchemeNoParticleBoundaries>
{
public:

    static const std::size_t NUM_TOKENS_PER_PARTICLE = 5;
    static const std::vector<int> get_padding_sequence();
    static const std::vector<int> tokenize_event(const std::vector<double>& event);
    static const std::vector<double> untokenize_event(const std::vector<int>& event, const Dictionary& dictionary);
};

//Scheme paddingv2 does not the particle_start and particle_end tokens for the padding, but does have it for the particles.
class SchemePaddingV2 : public SchemeBase<SchemePaddingV2>
{
public:

    static const std::size_t NUM_TOKENS_PER_PARTICLE = 7;
    static const std::vector<int> get_padding_sequence();
    static const std::vector<int> tokenize_event(const std::vector<double>& event);
    static const std::vector<double> untokenize_event(const std::vector<int>& event, const Dictionary& dictionary);
};

/**
 * Neo schemes tokenize into pdgid, energy, pt, eta, phi.
 */

//Scheme no_particle_boundaries does not the particle_start and particle_end tokens.
class SchemeNeoNoParticleBoundaries : public SchemeBase<SchemeNeoNoParticleBoundaries>
{
public:

    static const std::size_t NUM_TOKENS_PER_PARTICLE = 5;
    static const std::vector<int> get_padding_sequence();
    static const std::vector<int> tokenize_event(const std::vector<double>& event);
    static const std::vector<double> untokenize_event(const std::vector<int>& event, const Dictionary& dictionary);
};

/**
 * NeoV2 schemes tokenize into pdgid, pt, eta, phi.
 * Contain no particle boundaries.
 */

//Scheme no_particle_boundaries does not the particle_start and particle_end tokens.
class SchemeNeoV2 : public SchemeBase<SchemeNeoV2>
{
public:

    static const std::size_t NUM_TOKENS_PER_PARTICLE = 4;
    static const std::vector<int> get_padding_sequence();
    static const std::vector<int> tokenize_event(const std::vector<double>& event);
    static const std::vector<double> untokenize_event(const std::vector<int>& event, const Dictionary& dictionary);
};

namespace DataManager
{
    const Dictionary load_dictionary(std::string dictionary_path);
    const std::vector<std::vector<int>> load_tokenized_data(std::string input_data_path);
    void output_tokenized_data(std::string output_file_path, const std::vector<std::vector<int>>& tokenized_data);
    void output_raw_data(std::string output_file_path, const std::vector<std::vector<double>>& raw_data);
};
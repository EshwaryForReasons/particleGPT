
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
#include <iomanip>
#include <format>

template class SchemeBase<SchemeStandard>;
template class SchemeBase<SchemeNoEta>;
template class SchemeBase<SchemeNoParticleBoundaries>;
template class SchemeBase<SchemePaddingV2>;
template class SchemeBase<SchemeNeoNoParticleBoundaries>;
template class SchemeBase<SchemeNeoV2>;

template<typename Derived>
void SchemeBase<Derived>::untokenize_data(std::string dictionary_path, std::string input_data_path, std::string output_data_path)
{
    std::printf("----------------------------------------\n");
    const auto dictionary = DataManager::load_dictionary(dictionary_path);
    const auto tokenized_data = DataManager::load_tokenized_data(input_data_path);
    std::printf("pTokenizer: untokenizer: Began untokenizing data.\n");
    std::vector<std::vector<double>> raw_data;
    
    //Profiling shows no reason to multithread this one (more threads was actually slower?)
    for (auto& event : tokenized_data)
    {
        const auto untokenized_event = Derived::untokenize_event(event, dictionary);
        raw_data.push_back(untokenized_event);
    }

    DataManager::output_raw_data(output_data_path, raw_data);
    std::printf("pTokenizer: untokenizer: Finished untokenizing data.\n");
    std::printf("----------------------------------------\n");
}

const std::vector<double> SchemeStandard::untokenize_event(const std::vector<int>& event, const Dictionary& dictionary)
{
    std::vector<double> untokenized_event;
    for (int i = 0; i < event.size(); i += NUM_TOKENS_PER_PARTICLE)
    {
        int pdgid_idx = event[i] - dictionary.offsets.pdgid_offset;
        int energy_idx = event[i + 1] - dictionary.offsets.energy_offset;
        int eta_idx = event[i + 2] - dictionary.offsets.eta_offset;
        int theta_idx = event[i + 3] - dictionary.offsets.theta_offset;
        int phi_idx = event[i + 4] - dictionary.offsets.phi_offset;

        int pdgid = 0;
        for (auto& [pdg_id, pdg_idx] : dictionary.pdgid_to_index)
        {
            if (pdg_idx == pdgid_idx)
            {
                pdgid = pdg_id;
                break;
            }
        }
        double energy = pMath::get_bin_median(dictionary.e_bins, energy_idx);
        double eta = pMath::get_bin_median(dictionary.eta_bins, eta_idx);
        double theta = pMath::get_bin_median(dictionary.theta_bins, theta_idx);
        double phi = pMath::get_bin_median(dictionary.phi_bins, phi_idx);

        double px = energy * std::sin(theta) * std::cos(phi);
        double py = energy * std::sin(theta) * std::sin(phi);
        double pz = energy * std::cos(theta);

        untokenized_event.push_back(pdgid);
        untokenized_event.push_back(energy);
        untokenized_event.push_back(px);
        untokenized_event.push_back(py);
        untokenized_event.push_back(pz);
    }
    return untokenized_event;
}

const std::vector<double> SchemeNoEta::untokenize_event(const std::vector<int>& event, const Dictionary& dictionary)
{
    std::vector<double> untokenized_event;
    for (int i = 0; i < event.size(); i += NUM_TOKENS_PER_PARTICLE)
    {
        int pdgid_idx = event[i] - dictionary.offsets.pdgid_offset;
        int energy_idx = event[i + 1] - dictionary.offsets.energy_offset;
        int theta_idx = event[i + 2] - dictionary.offsets.theta_offset;
        int phi_idx = event[i + 3] - dictionary.offsets.phi_offset;

        int pdgid = 0;
        for (auto& [pdg_id, pdg_idx] : dictionary.pdgid_to_index)
        {
            if (pdg_idx == pdgid_idx)
            {
                pdgid = pdg_id;
                break;
            }
        }
        double energy = pMath::get_bin_median(dictionary.e_bins, energy_idx);
        double theta = pMath::get_bin_median(dictionary.theta_bins, theta_idx);
        double phi = pMath::get_bin_median(dictionary.phi_bins, phi_idx);

        double px = energy * std::sin(theta) * std::cos(phi);
        double py = energy * std::sin(theta) * std::sin(phi);
        double pz = energy * std::cos(theta);

        untokenized_event.push_back(pdgid);
        untokenized_event.push_back(energy);
        untokenized_event.push_back(px);
        untokenized_event.push_back(py);
        untokenized_event.push_back(pz);
    }
    return untokenized_event;
}

const std::vector<double> SchemeNoParticleBoundaries::untokenize_event(const std::vector<int>& event, const Dictionary& dictionary)
{
    std::vector<double> untokenized_event;
    for (int i = 0; i < event.size(); i += NUM_TOKENS_PER_PARTICLE)
    {
        int pdgid_idx = event[i] - dictionary.offsets.pdgid_offset;
        int energy_idx = event[i + 1] - dictionary.offsets.energy_offset;
        int eta_idx = event[i + 2] - dictionary.offsets.eta_offset;
        int theta_idx = event[i + 3] - dictionary.offsets.theta_offset;
        int phi_idx = event[i + 4] - dictionary.offsets.phi_offset;

        int pdgid = 0;
        for (auto& [pdg_id, pdg_idx] : dictionary.pdgid_to_index)
        {
            if (pdg_idx == pdgid_idx)
            {
                pdgid = pdg_id;
                break;
            }
        }
        double energy = pMath::get_bin_median(dictionary.e_bins, energy_idx);
        double eta = pMath::get_bin_median(dictionary.eta_bins, eta_idx);
        double theta = pMath::get_bin_median(dictionary.theta_bins, theta_idx);
        double phi = pMath::get_bin_median(dictionary.phi_bins, phi_idx);

        double px = energy * std::sin(theta) * std::cos(phi);
        double py = energy * std::sin(theta) * std::sin(phi);
        double pz = energy * std::cos(theta);

        untokenized_event.push_back(pdgid);
        untokenized_event.push_back(energy);
        untokenized_event.push_back(px);
        untokenized_event.push_back(py);
        untokenized_event.push_back(pz);
    }
    return untokenized_event;
}

const std::vector<double> SchemePaddingV2::untokenize_event(const std::vector<int>& event, const Dictionary& dictionary)
{
    std::vector<double> untokenized_event;
    for (int i = 0; i < event.size(); i += NUM_TOKENS_PER_PARTICLE)
    {
        int pdgid_idx = event[i] - dictionary.offsets.pdgid_offset;
        int energy_idx = event[i + 1] - dictionary.offsets.energy_offset;
        int eta_idx = event[i + 2] - dictionary.offsets.eta_offset;
        int theta_idx = event[i + 3] - dictionary.offsets.theta_offset;
        int phi_idx = event[i + 4] - dictionary.offsets.phi_offset;

        int pdgid = 0;
        for (auto& [pdg_id, pdg_idx] : dictionary.pdgid_to_index)
        {
            if (pdg_idx == pdgid_idx)
            {
                pdgid = pdg_id;
                break;
            }
        }
        double energy = pMath::get_bin_median(dictionary.e_bins, energy_idx);
        double eta = pMath::get_bin_median(dictionary.eta_bins, eta_idx);
        double theta = pMath::get_bin_median(dictionary.theta_bins, theta_idx);
        double phi = pMath::get_bin_median(dictionary.phi_bins, phi_idx);

        double px = energy * std::sin(theta) * std::cos(phi);
        double py = energy * std::sin(theta) * std::sin(phi);
        double pz = energy * std::cos(theta);

        untokenized_event.push_back(pdgid);
        untokenized_event.push_back(energy);
        untokenized_event.push_back(px);
        untokenized_event.push_back(py);
        untokenized_event.push_back(pz);
    }
    return untokenized_event;
}

const std::vector<double> SchemeNeoNoParticleBoundaries::untokenize_event(const std::vector<int>& event, const Dictionary& dictionary)
{
    std::vector<double> untokenized_event;
    for (int i = 0; i < event.size(); i += NUM_TOKENS_PER_PARTICLE)
    {
        int pdgid_idx = event[i] - dictionary.offsets.pdgid_offset;
        int energy_idx = event[i + 1] - dictionary.offsets.energy_offset;
        int pt_idx = event[i + 2] - dictionary.offsets.pt_offset;
        int eta_idx = event[i + 3] - dictionary.offsets.eta_offset;
        int phi_idx = event[i + 4] - dictionary.offsets.phi_offset;

        int pdgid = 0;
        for (auto& [pdg_id, pdg_idx] : dictionary.pdgid_to_index)
        {
            if (pdg_idx == pdgid_idx)
            {
                pdgid = pdg_id;
                break;
            }
        }
        double energy = pMath::get_bin_median(dictionary.e_bins, energy_idx);
        double pt = pMath::get_bin_median(dictionary.pt_bins, pt_idx);
        double eta = pMath::get_bin_median(dictionary.eta_bins, eta_idx);
        double phi = pMath::get_bin_median(dictionary.phi_bins, phi_idx);

        double px = pt * std::cos(phi);
        double py = pt * std::sin(phi);
        double pz = pt * std::sinh(phi);

        untokenized_event.push_back(pdgid);
        untokenized_event.push_back(energy);
        untokenized_event.push_back(px);
        untokenized_event.push_back(py);
        untokenized_event.push_back(pz);
    }
    return untokenized_event;
}

const std::vector<double> SchemeNeoV2::untokenize_event(const std::vector<int>& event, const Dictionary& dictionary)
{
    std::vector<double> untokenized_event;
    for (int i = 0; i < event.size(); i += NUM_TOKENS_PER_PARTICLE)
    {
        int pdgid_idx = event[i] - dictionary.offsets.pdgid_offset;
        int pt_idx = event[i + 1] - dictionary.offsets.pt_offset;
        int eta_idx = event[i + 1] - dictionary.offsets.eta_offset;
        int phi_idx = event[i + 3] - dictionary.offsets.phi_offset;

        int pdgid = 0;
        for (auto& [pdg_id, pdg_idx] : dictionary.pdgid_to_index)
        {
            if (pdg_idx == pdgid_idx)
            {
                pdgid = pdg_id;
                break;
            }
        }
        double eta = pMath::get_bin_median(dictionary.eta_bins, eta_idx);
        double pt = pMath::get_bin_median(dictionary.pt_bins, pt_idx);
        double phi = pMath::get_bin_median(dictionary.phi_bins, phi_idx);

        double px = pt * std::cos(phi);
        double py = pt * std::sin(phi);
        double pz = pt * std::sinh(phi);
        //@TODO: find a way to get the mass from the pdgid. Particle in python can do this, but not sure how to do it in C++.
        double mass = 0.0f;
        double energy = std::sqrt(px * px + py * py + pz * pz + mass * mass);

        untokenized_event.push_back(pdgid);
        untokenized_event.push_back(energy);
        untokenized_event.push_back(px);
        untokenized_event.push_back(py);
        untokenized_event.push_back(pz);
    }
    return untokenized_event;
}
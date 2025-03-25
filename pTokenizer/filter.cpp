
#include "main.h"
#include "dictionary.h"
#include "utils.h"

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

void Filter::filter_data(const std::string& output_data_path)
{
    //data_manager->tokenized_data_path is expected to be the generated samples (which are tokenized)

    std::printf("----------------------------------------\n");

    if (data_manager->tokenized_data.size() == 0)
    {
        throw std::runtime_error("pTokenizer: untokenizer: Tokenized data not loaded.");
    }

    //Does the event contain event_start and event_end tokens at the start and end respectively
    auto valid_borders = [](const std::vector<int>& event)
    {
        return event[0] == dictionary->special_tokens.event_start && event.back() == dictionary->special_tokens.event_end;
    };

    //Does this event have secondaries and five tokens per particle. Assumes special tokens have been removed already
    auto well_formed_events = [](const std::vector<int>& event)
    {
        bool b_event_has_secondaries = event.size() > 5;
        bool b_event_has_five_tokens_per_particle = event.size() % 5 == 0;
        return b_event_has_secondaries && b_event_has_five_tokens_per_particle;
    };

    //Removes all special tokens from events
    auto remove_special_tokens = [](std::vector<int>& event)
    {
        auto is_special_token = [](int val) { return val == 0 || val == 1 || val == 2 || val == 3 || val == 4; };
        event.erase(std::remove_if(event.begin(), event.end(), is_special_token), event.end());
        return event;
    };

    //Does this event have valid token ranges
    auto valid_token_ranges = [](const std::vector<int>& event)
    {
        bool b_token_ranges_valid = true;
        for (int i = 0; i < event.size(); i += 5)
        {
            int pdgid_offset_min = dictionary->offsets.pdgid_offset;
            int pdgid_offset_max = dictionary->offsets.pdgid_offset + (int)dictionary->pdgid_to_index.size() - 1;
            int energy_offset_min = dictionary->offsets.energy_offset;
            int energy_offset_max = dictionary->offsets.energy_offset + (int)dictionary->e_bins.size() - 1;
            int eta_offset_min = dictionary->offsets.eta_offset;
            int eta_offset_max = dictionary->offsets.eta_offset + (int)dictionary->eta_bins.size() - 1;
            int theta_offset_min = dictionary->offsets.theta_offset;
            int theta_offset_max = dictionary->offsets.theta_offset + (int)dictionary->theta_bins.size() - 1;
            int phi_offset_min = dictionary->offsets.phi_offset;
            int phi_offset_max = dictionary->offsets.phi_offset + (int)dictionary->phi_bins.size() - 1;

            const bool b_invalid_range_found = event[i] < pdgid_offset_min || event[i] > pdgid_offset_max ||
                event[i + 1] < energy_offset_min || event[i + 1] > energy_offset_max ||
                event[i + 2] < eta_offset_min || event[i + 2] > eta_offset_max ||
                event[i + 3] < theta_offset_min || event[i + 3] > theta_offset_max ||
                event[i + 4] < phi_offset_min || event[i + 4] > phi_offset_max;
            
            if (b_invalid_range_found)
            {
                b_token_ranges_valid = false;
                break;
            }
        }

        return b_token_ranges_valid;
    };

    std::printf("pTokenizer: filter: Began filtering data.\n");
    auto filtered_data_view = data_manager->tokenized_data 
        | std::views::filter(valid_borders)
        | std::views::transform(remove_special_tokens)
        | std::views::filter(well_formed_events)
        | std::views::filter(valid_token_ranges);
    std::vector<std::vector<int>> filtered_data(filtered_data_view.begin(), filtered_data_view.end());
    std::printf("pTokenizer: filter: Finished filtering data.\n");

    //Update data_manager with filtered data
    
    data_manager->tokenized_data = filtered_data;

    //Output untokenized data

    std::printf("pTokenizer: filter: Began outputting filtered data.\n");
    std::ofstream output_file(output_data_path);
    for (auto& event : filtered_data)
    {
        for (int i = 0; i < event.size(); i += 5)
        {
            output_file << std::format("{:d} {:d} {:d} {:d} {:d}{}", (int)event[i], event[i + 1], event[i + 2], event[i + 3], event[i + 4], i + 5 == event.size() ? "" : " ");
        }
        output_file << "\n";
    }
    std::printf("pTokenizer: filter: Finished outputting filtered data.\n");
    std::printf("----------------------------------------\n");
}
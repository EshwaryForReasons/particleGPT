
#pragma once

#include <vector>
#include <stdexcept>
#include <cmath>
#include <iostream>

struct BinData
{
    std::vector<double> bins;
    std::string transform_type = "linear";
};

namespace pMath
{
    inline int fast_stoi(std::string_view str)
    {
        if (str.empty())
            throw std::invalid_argument("Input string is empty");

        int result = 0;
        size_t i = 0;
        bool negative = false;

        // Check for optional sign
        if (str[i] == '-')
        {
            negative = true;
            ++i;
        }
        else if (str[i] == '+')
        {
            ++i;
        }

        // Convert the digits
        for (; i < str.size(); ++i)
        {
            char ch = str[i];
            if (ch < '0' || ch > '9')
                throw std::invalid_argument("Invalid character in input string");

            result = result * 10 + (ch - '0');
        }

        return negative ? -result : result;
    }

    inline double round_to_dp(double value, int dp)
    {
        const double multiplier = std::pow(10.0, dp);
        return std::round(value * multiplier) / multiplier;
    }
    
    inline std::vector<double> arange(double start, double stop, double step)
    {
        if (step <= 0)
            return {};
        
        std::vector<double> result;
        for (double i = start; i < stop; i += step)
        {
            result.push_back(i);
        }
        return result;
    }

    inline std::vector<double> linspace(double start, double stop, double n_bins)
    {
        if (n_bins <= 0)
            return {};
        
        double step = (stop - start) / n_bins;
        int decimal_places = (int)std::ceil(std::log10(1 / step));

        std::vector<double> result;
        for (double i = start; i < stop; i += step)
        {
            result.push_back(round_to_dp(i, decimal_places));
        }
        return result;
    }

    //Replica of numpy.digitize
    inline int digitize(double value, const std::vector<double>& bins)
    {
        //Fit it in the middle
        for (int i = 1; i <= bins.size(); ++i)
        {
            if (value >= bins[i - 1] && value < bins[i])
            {
                return i;
            }
        }

        //Figure out if it is on the lower or upper edge
        if (value < bins[0])
            return 0;
        //We return the last bin if it is greater than the last bin.
        //Numpy returns one more than this (bins.size() instead of bins.size() - 1)
        else if (value >= bins[bins.size() - 1])
            return bins.size() - 1;

        throw std::runtime_error("Value is not in any bin");
    }

    inline int digitize(double value, const BinData& bin_data)
    {
        if (bin_data.transform_type == "log")
            value = std::log(value);

        //Fit it in the middle
        for (int i = 1; i <= bin_data.bins.size(); ++i)
        {
            if (value >= bin_data.bins[i - 1] && value < bin_data.bins[i])
            {
                return i;
            }
        }

        //Figure out if it is on the lower or upper edge
        if (value < bin_data.bins[0])
            return 0;
        //We return the last bin if it is greater than the last bin.
        //Numpy returns one more than this (bins.size() instead of bins.size() - 1)
        else if (value >= bin_data.bins[bin_data.bins.size() - 1])
            return bin_data.bins.size() - 1;

        throw std::runtime_error("Value is not in any bin");
    }

    inline double get_bin_median(const std::vector<double>& bins, int bin_idx)
    {
        return (bins[bin_idx - 1] + bins[bin_idx]) / 2;
    }

    inline double get_bin_median(const BinData& bin_data, int bin_idx)
    {
        return (bin_data.bins[bin_idx - 1] + bin_data.bins[bin_idx]) / 2;
    }
}

namespace Utils
{
    inline std::vector<std::string_view> split(std::string_view in_str, char delimiter = ' ')
    {
        std::vector<std::string_view> out_vec;
        while (in_str.size() > 0)
        {
            const std::size_t delim_pos = in_str.find_first_of(delimiter);
            std::string_view substr = in_str.substr(0, delim_pos);
            in_str.remove_prefix(delim_pos + 1);
            out_vec.push_back(substr);

            if (delim_pos == std::string_view::npos)
                break;
        }
        return out_vec;
    }
}
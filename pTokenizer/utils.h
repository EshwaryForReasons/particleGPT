
#pragma once

#include <vector>
#include <stdexcept>

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
    
    //Replica of numpy.arange
    inline std::vector<double> arange(double start, double stop, double step)
    {
        std::vector<double> result;
        for (double i = start; i < stop; i += step)
        {
            result.push_back(i);
        }
        return result;
    }

    //Replica of numpy.digitize
    inline int digitize(double value, const std::vector<double>& bins)
    {
        for (int i = 1; i <= bins.size(); ++i)
        {
            if (value >= bins[i - 1] && value < bins[i])
            {
                return i;
            }
        }

        return -4;
    }

    inline double get_bin_median(const std::vector<double>& bins, int bin_idx)
    {
        return (bins[bin_idx - 1] + bins[bin_idx]) / 2;
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
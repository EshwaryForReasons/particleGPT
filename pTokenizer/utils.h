
#pragma once

#include <vector>

namespace pMath
{
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
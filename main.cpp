#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

#include "BitDepth.hpp"
#include "EMA.hpp"
#include "Hysteresis.hpp"

float mean   = 0;
float stddev = 1;

auto dist = std::bind(std::normal_distribution<double>{mean, stddev},
                      std::mt19937(std::random_device{}()));

using std::cout;
using std::endl;

#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;

int main() {
    std::vector<uint16_t> input(1024);
    std::generate(input.begin(), input.end(), [i = 0u]() mutable {
        float result =
            (i < 512) ? 1000.0 / 2 * (1 - cos(i * M_PI / 512)) : 1000.0;
        result += dist();
        if (result < 0)
            result = 0;
        if (result >= 1024)
            result = 1023;
        i++;
        return result;
    });
    std::vector<uint16_t> filtered(input.size());
    std::transform(input.begin(), input.end(), filtered.begin(),
                   [filter = EMA<5, uint16_t>()](uint16_t input) mutable {
                       return filter.filter(input);
                   });
    std::vector<uint16_t> hiRes(input.size());
    std::transform(input.begin(), input.end(), hiRes.begin(),
                   [](uint16_t input) {
                       return increaseBitDepth<16, 10, uint16_t>(input);
                   });
    std::vector<uint16_t> hiResFiltered(input.size());
    std::transform(hiRes.begin(), hiRes.end(), hiResFiltered.begin(),
                   [filter = EMA<5, uint32_t>()](uint16_t input) mutable {
                       return filter.filter(input);
                   });
    std::vector<uint16_t> hysteresis(input.size());
    std::transform(
        filtered.begin(), filtered.end(), hysteresis.begin(),
        [hysteresis =
             Hysteresis<3, uint16_t, uint8_t>()](uint16_t input) mutable {
            hysteresis.update(input);
            return increaseBitDepth<10, 7, uint16_t>(hysteresis.getValue());
        });
    std::vector<uint16_t> hiResHysteresis(input.size());
    std::transform(hiResFiltered.begin(), hiResFiltered.end(),
                   hiResHysteresis.begin(),
                   [hysteresis = Hysteresis<6, uint16_t, uint16_t>()](
                       uint16_t input) mutable {
                       hysteresis.update(input);
                       return hysteresis.getValue();
                   });
    // for (size_t i = 0; i < input.size(); i++)
    //     std::cout << input[i] << '\t' << filtered[i] << '\n';

    py::scoped_interpreter guard{};
    using namespace py::literals;

    py::exec(R"(
        import matplotlib.pyplot as plt
        import numpy as np

        plt.plot(input, label='input')
        plt.plot(filtered, label='filtered')
        plt.plot(np.array(hiRes) / 64.0, label='hiRes')
        plt.plot(np.array(hiResFiltered) / 64.0, label='hiResFiltered')
        plt.plot(hysteresis, label='hysteresis')
        plt.plot(hiResHysteresis, label='hiResHysteresis')

        plt.legend()
        plt.show()        
    )",
             py::globals(),
             py::dict{
                 "input"_a           = input,
                 "filtered"_a        = filtered,
                 "hiRes"_a           = hiRes,
                 "hiResFiltered"_a   = hiResFiltered,
                 "hysteresis"_a      = hysteresis,
                 "hiResHysteresis"_a = hiResHysteresis,
             });
}
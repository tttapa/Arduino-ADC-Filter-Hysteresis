#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

#include "BitDepth.hpp"
#include "EMA.hpp"
#include "Hysteresis.hpp"

using std::bind;
using std::cout;
using std::endl;
using std::mt19937;
using std::normal_distribution;
using std::transform;
using std::vector;

#include <pybind11/embed.h>
#include <pybind11/stl.h>
namespace py = pybind11;

const uint16_t MAX_VAL = 1023;

int main() {
    // Create a raised cosine step as the input from 0 to 1023 (10 bits)
    vector<float> clean(2048 - 512 - 128);
    std::generate(clean.begin(), clean.end(), [i = 0u]() mutable {
        float result =
            (i < 512) ? MAX_VAL / 2.0 * (1 - cos(i * M_PI / 512)) : MAX_VAL;
        i++;
        return result;
    });

    // Add some Gaussian noise
    float mean = 0;
    float stddev = 1;
    auto dist = bind(normal_distribution<float>{mean, stddev}, mt19937(10));
    vector<uint16_t> noisy(clean.size());
    transform(clean.begin(), clean.end(), noisy.begin(), [&](float clean) {
        float result = clean + dist();
        if (result < 0)
            result = 0;
        if (result >= 1024)
            result = 1023;
        return result;
    });

    // Filter the 10-bit noisy input signal using an EMA single-pole filter
    // with the pole in z = 0.96875 = 1 - 1/32 = 1 - 1/(2^5).
    vector<uint16_t> filtered(clean.size());
    transform(noisy.begin(), noisy.end(), filtered.begin(),
              [filter = EMA<5, uint16_t>()](uint16_t noisy) mutable {
                  return filter.filter(noisy);
              });

    // Scale up the 10-bit noisy input to 16 bits.
    // This will reduce the rounding errors when filtering (using integer math)
    // and the higher resolution allows us to use larger hysteresis thresholds
    // as well, without losing any precision.
    vector<uint16_t> hiRes(clean.size());
    transform(noisy.begin(), noisy.end(), hiRes.begin(), [](uint16_t noisy) {
        return increaseBitDepth<16, 10, uint16_t>(noisy);
        // return noisy << (16 - 10);
    });

    // Filter the 16-bit noisy signal using the same EMA filter as before.
    vector<uint16_t> hiResFiltered(clean.size());
    transform(hiRes.begin(), hiRes.end(), hiResFiltered.begin(),
              [filter = EMA<5, uint32_t>()](uint16_t noisy) mutable {
                  return filter.filter(noisy);
              });

    // Apply hysteresis with a threshold of 3 bits to the 10-bit filtered signal.
    // This means 3 bits of precision are lost, resulting in 7 bits of effective
    // resolution.
    vector<uint16_t> hysteresis(clean.size());
    transform(filtered.begin(), filtered.end(), hysteresis.begin(),
              [hysteresis =
                   Hysteresis<3, uint16_t, uint8_t>()](uint16_t noisy) mutable {
                  hysteresis.update(noisy);
                  return increaseBitDepth<10, 7, uint16_t>(
                      hysteresis.getValue());
                  // return hysteresis.getValue() << (10 - 7);
              });

    // Apply hysteresis with a threshold of 6 bits to the 16-bit filtered signal.
    // This means 6 bits of precision are lost, resulting in 10 bits of
    // effective resolution.
    vector<uint16_t> hiResHysteresis(clean.size());
    transform(hiResFiltered.begin(), hiResFiltered.end(),
              hiResHysteresis.begin(),
              [hysteresis = Hysteresis<6, uint16_t, uint16_t>()](
                  uint16_t noisy) mutable {
                  hysteresis.update(noisy);
                  return hysteresis.getValue();
              });

    // Start a Python interpreter
    py::scoped_interpreter guard{};
    using namespace py::literals;

    // Store all signals in a Python dict.
    auto data = py::dict{
        "clean"_a = clean,
        "noisy"_a = noisy,
        "filtered"_a = filtered,
        "hiRes"_a = hiRes,
        "hiResFiltered"_a = hiResFiltered,
        "hysteresis"_a = hysteresis,
        "hiResHysteresis"_a = hiResHysteresis,
    };

    // Plot the signals using Matplotlib
    py::exec(R"(
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 15})

    def plot(data, ax):
        import numpy as np
        t = np.arange(0, len(data['clean']))
        ax.plot(t, data['clean'], label='Clean Input')
        ax.plot(t, data['noisy'], label='Noisy Input')
        # ax.plot(t, data['filtered'], label='filtered')
        # ax.plot(t, np.floor(np.array(data['hiRes']) / 64.0), label='hiRes')
        # ax.plot(t, np.floor(np.array(data['hiResFiltered']) / 64.0), label='hiResFiltered')
        ax.plot(t, np.array(data['hiResFiltered']) / 64.0, label='High Bit-Depth Filtered')
        ax.step(t, data['hysteresis'], label='Filtered + Hysteresis', where='post')
        ax.step(t, data['hiResHysteresis'], label='High Bit-Depth Filtered + Hysteresis', where='post')

    fig, ax = plt.subplots(figsize=[16,10])
    plot(data, ax)
    plt.legend(loc='upper left', fontsize=13)

    plt.title('ADC Filtering using an Exponential Moving Average Filter and Hysteresis')
    plt.xlabel('Sample')
    plt.ylabel('ADC value')

    # inset axes....
    axins1 = ax.inset_axes([0.5, 0.05, 0.45, 0.4])
    plot(data, axins1)
    axins2 = ax.inset_axes([0.5, 0.5, 0.45, 0.4])
    plot(data, axins2)
    
    # sub region of the original plot
    x1, x2, y1, y2 = 0, 120, 0, 80
    axins1.set_xlim(x1, x2)
    axins1.set_ylim(y1, y2)
    axins1.set_xticklabels('')
    # axins1.set_yticklabels('')
    axins1.set_yticks(np.arange(8*(y1//8)-0, y2+1, 8.0))
    axins1.tick_params(axis='both', which='major', labelsize=10)

    x1, x2, y1, y2 = 460, 770, 982, 1030
    axins2.set_xlim(x1, x2)
    axins2.set_ylim(y1, y2)
    axins2.set_xticklabels('')
    # axins2.set_yticklabels('')
    axins2.set_yticks(np.arange(8*(y1//8)-1, y2+1, 8.0))
    axins2.tick_params(axis='both', which='major', labelsize=10)

    ax.indicate_inset_zoom(axins1)
    ax.indicate_inset_zoom(axins2)

    plt.tight_layout()
    plt.show()
    )",
             py::globals(), py::dict{"data"_a = data});
}
/* ✔ */

#pragma once

#include <stdint.h>

/**
 * @brief   A class for single-pole infinite impulse response filters
 *          or exponential moving average filters.
 *
 * Optimized implementation of the difference equation with a slight
 * optimization by using a factor of two as the pole location (this means
 * that no division or floating point operations are required).
 *
 * Difference equation: @f$ y[n] = \alpha·x[n]+(1-\alpha)·y[n-1] @f$
 * where @f$ \alpha = \left(\frac{1}{2}\right)^{K} @f$, @f$ x @f$ is the
 * input sequence, and @f$ y @f$ is the output sequence.
 *
 * [An in-depth explanation of the EMA filter]
 * (https://tttapa.github.io/Pages/Mathematics/Systems-and-Control-Theory/Digital-filters/Exponential%20Moving%20Average/)
 *
 * @tparam  K
 *          The amount of bits to shift by. This determines the location
 *          of the pole in the EMA transfer function, and therefore the
 *          cut-off frequency.  
 *          The higher this number, the more filtering takes place.  
 *          The pole location is @f$ 1 - 2^{-K} @f$.
 * @tparam  uint_t
 *          The (signed) integer type to use for the input, intermediate values
 *          and the output.
 *          Should be at least @f$ M+K @f$ bits wide, where @f$ M @f$
 *          is the maximum number of bits of the input.
 *          In case of the Arduino's built-in ADC,
 *          @f$ M = 10 = \log_2(1024) @f$.
 * 
 * @ingroup    Utilities
 */
template <uint8_t K, class uint_t>
class EMA {
  public:
    /**
     * @brief   Filter the input: Given @f$ x[n] @f$, calculate @f$ y[n] @f$.
     *
     * @param   input
     *          The new raw input value.
     * @return  The new filtered output value.
     */
    uint_t filter(uint_t input) {
        filtered += input;
        uint_t output = (filtered + fixedPointAHalf) >> K;
        filtered -= output;
        return output;
    }

    /**
     * @brief   Filter the input: Given @f$ x[n] @f$, calculate @f$ y[n] @f$.
     *
     * @param   value
     *          The new raw input value.
     * @return  The new filtered output value.
     */
    uint_t operator()(uint_t value) { return filter(value); }

    static_assert(
        uint_t(0) < uint_t(-1), // Check that `uint_t` is an unsigned type
        "Error: the uint_t type should be an unsigned integer, otherwise, "
        "the division using bit shifts is invalid.");

  private:
    uint_t filtered = 0;
    constexpr static uint_t fixedPointAHalf = 1 << (K - 1);
};

/* ✔ */

#pragma once

#include <stdint.h>

/**
 * @brief   A class for applying hysteresis to a given input.
 *
 * This reduces the noise by decreasing the resolution, and it prevents flipping
 * back and forth between two values.
 *
 * <b>An example for `BITS` = 7 and an input from 0 to 1023</b>
 * ```
 *    7                                                     ┌───◄───┬───
 * o  6                                             ┌───◄───┼───►───┘
 * u  5                                     ┌───◄───┼───►───┘
 * t  4                             ┌───◄───┼───►───┘
 * p  3                     ┌───◄───┼───►───┘
 * u  2             ┌───◄───┼───►───┘
 * t  1     ┌───◄───┼───►───┘
 *    0 ────┴───►───┘
 *      0      128     256     384     512     640     768     896    1023
 *                                  i n p u t
 * ```
 *
 * @tparam  BITS
 *          The number of bits to decrease in resolution.
 *          Increasing this number will result in a decrease in fluctuations.
 */
template <uint8_t BITS, class T_in = uint16_t, class T_out = uint8_t>
class Hysteresis {
  public:
    /**
     * @brief   Update the hysteresis output with a new input value.
     *
     * @param   input
     *          The input to calculate the output level from.
     * @return  true
     *          The output level has changed.
     * @return  false
     *          The output level is still the same.
     */
    bool update(T_in inputLevel) {
        T_in previousLevelFull = ((T_in) previousLevel << BITS) | offset;
        T_in lowerbound = previousLevel > 0 ? previousLevelFull - margin : 0;
        T_in upperbound = previousLevelFull + margin;
        if (inputLevel < lowerbound || inputLevel > upperbound) {
            previousLevel = inputLevel >> BITS;
            return true;
        }
        return false;
    }

    /**
     * @brief   Get the current output level.
     * 
     * @return  The output level.
     */
    T_out getValue() const { return previousLevel; }

  private:
    T_out previousLevel           = 0;
    constexpr static T_out margin = (1 << BITS) - 1;
    constexpr static T_out offset = 1 << (BITS - 1);
};
// Copyright (c) 2025 Mikko Kuha (University of Jyväskylä).
// This program is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details. You should have received a copy
// of the GNU General Public License along with this program. If not, see
// <http://www.gnu.org/licenses/>.

#ifndef GENERIC_HELPERS_HPP
#define GENERIC_HELPERS_HPP

#include <limits>
#include <random>
#include <vector>

#include "typedefs.hpp"

class helpers
{
public:
    // Reseeds a random engine for one unit of work (an event, a table block) from the user seed.
    // stream separates the independent uses of one seed, index the units of one use. With seed 0
    // the engine is left as it is (seeded from the clock), so the results are not reproducible.
    static auto reseed(
        std::mt19937 &eng,
        const uint_fast64_t &seed,
        const uint_fast32_t &stream,
        const uint_fast64_t &index) noexcept -> void
    {
        if (seed == 0)
        {
            return;
        }
        std::seed_seq seq{static_cast<uint_fast32_t>(seed & 0xffffffffu), static_cast<uint_fast32_t>(seed >> 32),
                          stream, static_cast<uint_fast32_t>(index & 0xffffffffu), static_cast<uint_fast32_t>(index >> 32)};
        eng.seed(seq);
    }

    // return an evenly spaced 1d grid of doubles.
    static auto linspace(
        const double &first,
        const double &last,
        const uint_fast16_t &len) noexcept -> std::vector<double>
    {
        std::vector<double> result(len);
        double step = (last - first) / static_cast<double>(len - 1);
        for (uint_fast16_t i = 0; i < len; i++)
        {
            result[i] = first + static_cast<double>(i) * step;
        }
        return result;
    }

    // return an  1d grid of doubles distributed evenly in logarithm
    static auto loglinspace(
        const double &first,
        const double &last,
        const uint_fast16_t &len) noexcept -> std::vector<double>
    {
        std::vector<double> result(len);
        double step = pow(last / first, 1.0 / static_cast<double>(len - 1));
        result[0] = first;
        for (uint_fast16_t i = 1; i < len; i++)
        {
            result[i] = result[i - 1] * step;
        }
        return result;
    }

    // Finds a zero of the function f
    template <typename F, typename Ret_type, typename Arg_type>
    static auto secant_method(
        Arg_type *const x,
        F f,
        const Ret_type error_tolerance,
        Ret_type *const last_fx,
        const Arg_type lambda = 1.0) -> void
    {
        Arg_type x_n = *x, x_np1 = 1.2 * x_n, x_np2 = 0.0;
        Ret_type fx_n = f(x_n);

        if (std::abs(fx_n) < error_tolerance)
        {
            *last_fx = fx_n;
            return;
        }

        Ret_type fx_np1 = f(x_np1);

        while (std::abs(fx_np1) > error_tolerance)
        {
            x_np2 = x_np1 - lambda * fx_np1 * (x_np1 - x_n) / (fx_np1 - fx_n);

            if (x_np2 <= 0.0)
                x_np2 = 1.0 / std::numeric_limits<Arg_type>::max();

            fx_n = fx_np1;
            x_n = x_np1;
            x_np1 = x_np2;

            fx_np1 = f(x_np1);

            if (x_n == x_np1)
            {
                std::cout << "Doesn't converge!!!" << std::endl;
                std::cout << x_n << ' ' << x_np1 << ' ' << fx_np1 << std::endl;
                return;
            }
        }

        *last_fx = fx_np1;
        *x = x_np1;
        return;
    }
};

#endif // GENERIC_HELPERS_HPP
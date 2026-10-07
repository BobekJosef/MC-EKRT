// Copyright (c) 2025 Mikko Kuha (University of Jyväskylä).
// This program is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details. You should have received a copy
// of the GNU General Public License along with this program. If not, see
// <http://www.gnu.org/licenses/>.
//
// This program uses OpenMP, which is licensed under the Apache License, Version 2.0
// (the "License"); you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TPP_HPP
#define TPP_HPP

#include <array>
#include <chrono>
#include <gsl/gsl_sf_lambert.h>
#include <omp.h>
#include <vector>

#include "generic_helpers.hpp"
#include "nn_coll.hpp"
#include "nucleon.hpp"
#include "nucleus_generator.hpp"
#include "typedefs.hpp"

class Tpp_builder
{
public:
    uint_fast64_t seed{0}; // user seed of the table Monte Carlo, 0 = clock (see helpers::reseed)

    Tpp_builder() : hotspots(false), proton_width_2(0.0), hs_width_2(0.0), func(nullptr) {}
    Tpp_builder(const double &proton_width_2_, const double &hotspot_width, const bool &hotspots_)
        : hotspots(hotspots_), proton_width_2(proton_width_2_)
    {
        if (hotspots)
        {
            this->hs_width_2 = std::pow(hotspot_width, 2);
            this->func = std::function<double(const double &, const std::vector<double> *const)>({[hs_width_2 = this->hs_width_2](const double &leave_zero = 0.0, const std::vector<double> *const hs_bs)
                                                                                                  {
                                                                                                      double Tpp = 0.0;
                                                                                                      uint_fast16_t n_hotspots2 = hs_bs->size(); // N*N
                                                                                                      for (auto bsquared : *hs_bs)
                                                                                                      {
                                                                                                          Tpp += exp(-bsquared / (4 * hs_width_2)) / (40 * M_PI * hs_width_2); // 1/fm² = mb/fm² * 1/mb = 0.1 * 1/mb
                                                                                                      }
                                                                                                      return Tpp / static_cast<double>(n_hotspots2);
                                                                                                  }});
        }
        else
        {
            this->hs_width_2 = 0.0;
            this->func = std::function<double(const double &, const std::vector<double> *const)>({[proton_width_2 = this->proton_width_2](const double &bsquared, const std::vector<double> *const hs_bs = nullptr)
                                                                                                  {
                                                                                                      return exp(-bsquared / (4 * proton_width_2)) / (40 * M_PI * proton_width_2); // 1/fm² = mb/fm² * 1/mb = 0.1 * 1/mb
                                                                                                  }});
        }
    }

    auto at(
        nn_coll &nn_pair) const noexcept -> double
    {
        if (hotspots)
        {
            auto vb2 = nn_pair.calculate_hs_bsquareds();
            return this->func(0.0, &vb2);
        }
        else
        {
            auto b2 = nn_pair.getcr_bsquared();
            return this->func(b2, nullptr);
        }
    }

    auto at(
        const double &b2) const noexcept -> double
    {
        if (hotspots)
        {
            // If at some point we need Tpp at singular b2, let's return the normal one proton Tpp
            std::vector<double> vb2{b2 * this->hs_width_2 / this->proton_width_2};
            return this->func(0.0, &vb2) * this->hs_width_2 / this->proton_width_2;
        }
        else
        {
            return this->func(b2, nullptr);
        }
    }

    auto at(
        const std::vector<double> *const vb2) const noexcept -> double
    {
        if (!hotspots)
        {
            // If we got here, something is very wrong
            std::cout << "HOTSPOT TPP CALLED WITH NO HOTSPOTS!!!!!!!" << std::endl;
            return 0.0;
        }
        else
        {
            return this->func(0.0, vb2);
        }
    }

    auto calculate_sum_tpp(
        const nucleon &nuc,
        const std::vector<nucleon> &nucleus) const noexcept -> double
    {
        double sum_tpp = 0.0;
        if (!this->hotspots)
        {
            uint_fast16_t A = static_cast<uint_fast16_t>(nucleus.size());

            auto [x1, y1, z1] = nuc.co;

            std::vector<double> tpps(A, 0.0);

            for (uint_fast16_t i = 0; i < A; i++)
            {
                if (nuc == nucleus.at(i)) // Do we calculate the effect of the same nucleon to itself?
                {
                    continue;
                }
                auto [x2, y2, z2] = nucleus.at(i).co;
                tpps.at(i) = this->func(pow(x1 - x2, 2) + pow(y1 - y2, 2), nullptr);
            }
            for (uint_fast16_t i = 0; i < A; i++)
            {
                sum_tpp += tpps.at(i);
            }
        }
        else
        {
            uint_fast16_t A = static_cast<uint_fast16_t>(nucleus.size());
            uint_fast16_t N_hs = static_cast<uint_fast16_t>(nucleus.at(0).hotspots.size());

            auto hotspots = nuc.hotspots;

            std::vector<double> tpps(A, 0.0);

            for (uint_fast16_t i = 0; i < A; i++)
            {
                if (nuc == nucleus.at(i)) // Do we calculate the effect of the same nucleon to itself?
                {
                    continue;
                }
                std::vector<double> b_squareds(N_hs * N_hs, 0.0);
                uint_fast16_t ind = 0;
                for (auto hs1 : hotspots)
                {
                    auto [x1, y1, z1] = hs1.co;
                    for (auto hs2 : nucleus.at(i).hotspots)
                    {
                        auto [x2, y2, z2] = hs2.co;
                        b_squareds.at(ind++) = pow(x1 - x2, 2) + pow(y1 - y2, 2);
                    }
                }

                tpps.at(i) = this->func(0.0, &b_squareds);
            }
            for (uint_fast16_t i = 0; i < A; i++)
            {
                sum_tpp += tpps.at(i);
            }
        }
        return sum_tpp;
    }

    auto calculate_TA(
        const double &x,
        const double &y,
        const std::vector<nucleon> &nucleus) const noexcept -> double
    {
        double sum_tp = 0.0;
        if (!this->hotspots)
        {
            uint_fast16_t A = static_cast<uint_fast16_t>(nucleus.size());

            for (uint_fast16_t i = 0; i < A; i++)
            {
                auto [x2, y2, z2] = nucleus.at(i).co;
                // T^i_p(x) = 2*T_pp(2*(x-x_i)^2)
                sum_tp += 2.0 * this->func(2.0 * (pow(x - x2, 2) + pow(y - y2, 2)), nullptr);
            }
        }
        else
        {
            uint_fast16_t A = static_cast<uint_fast16_t>(nucleus.size());
            uint_fast16_t N_hs = static_cast<uint_fast16_t>(nucleus.at(0).hotspots.size());

            for (uint_fast16_t i = 0; i < A; i++)
            {
                std::vector<double> b_squareds(N_hs, 0.0);
                uint_fast16_t ind = 0;
                for (auto hs : nucleus.at(i).hotspots)
                {
                    auto [x2, y2, z2] = hs.co;
                    b_squareds.at(ind++) = 2.0 * (pow(x - x2, 2) + pow(y - y2, 2));
                }
                sum_tp += 2.0 * this->func(0.0, &b_squareds);
            }
        }
        return sum_tp;
    }

    auto calculate_T_AA_0(
        const nucleus_generator::nucleus_params &nuc_params,
        const double &rel_tolerance,
        const bool verbose) -> double
    {
        std::vector<uint_fast8_t> block_indexes(200);
        std::iota(block_indexes.begin(), block_indexes.end(), 0); // generates the list as {0,1,2,3,...}

        uint_fast8_t block_amount = 50;
        std::vector<double> block_averages(block_amount);

#pragma omp parallel
        {
            auto eng = std::make_shared<std::mt19937>(static_cast<ulong>((omp_get_thread_num() + 1)) * static_cast<ulong>(std::chrono::system_clock::now().time_since_epoch().count()));

#pragma omp for
            for (uint_fast8_t block_index = 0; block_index < block_amount; block_index++)
            {
                helpers::reseed(*eng, this->seed, 1u, block_index);
                std::vector<double> T_AA_0s(200);
                for (auto it = block_indexes.begin(); it < block_indexes.end(); it++)
                {
                    std::vector<nucleon> nucl, other;
                    std::vector<uint_fast8_t> nucl_indexes(nuc_params.A);
                    std::iota(nucl_indexes.begin(), nucl_indexes.end(), 0);
                    std::vector<double> sum_tpps(nuc_params.A);

                    nucl = nucleus_generator::generate_nucleus(
                        nuc_params,
                        true,
                        0.0,
                        0.0,
                        eng);

                    for (auto itt = nucl_indexes.begin(); itt < nucl_indexes.end(); itt++)
                    {
                        sum_tpps[*itt] = this->calculate_sum_tpp(nucl[*itt], nucl);
                    }

                    T_AA_0s[*it] = std::reduce(sum_tpps.begin(), sum_tpps.end(), 0.0);
                }

                block_averages[block_index] = std::reduce(T_AA_0s.begin(), T_AA_0s.end(), 0.0) / 200.0;
            }
        }

        double ave_T_AA_0 = std::reduce(block_averages.begin(), block_averages.end(), 0.0) / block_amount;

        std::cout << "A-configs calculated: " << block_amount * 200 << std::endl;

        return ave_T_AA_0;
    }

    auto calculate_R_c_table(
        const nucleus_generator::nucleus_params &nuc_params,
        const double &rel_tolerance,
        const bool verbose) -> std::tuple<std::array<double, 25>, std::array<double, 25>>
    {
        std::vector<uint_fast8_t> block_indexes(200);
        std::iota(block_indexes.begin(), block_indexes.end(), 0); // generates the list as {0,1,2,3,...}

        std::array<double, 25> c_vector{-15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0};
        std::array<double, 25> ave_R_vector{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        uint_fast8_t block_amount = 50;
        std::array<std::array<double, 50>, 25> block_averages;

#pragma omp parallel
        {
            auto eng = std::make_shared<std::mt19937>(static_cast<ulong>((omp_get_thread_num() + 1)) * static_cast<ulong>(std::chrono::system_clock::now().time_since_epoch().count()));

#pragma omp for
            for (uint_fast8_t block_index = 0; block_index < block_amount; block_index++)
            {
                helpers::reseed(*eng, this->seed, 2u, block_index);
                std::array<std::array<double, 200>, 25> R_vectors;
                for (auto it = block_indexes.begin(); it < block_indexes.end(); it++)
                {
                    std::vector<nucleon> nucl, other;
                    std::vector<uint_fast8_t> nucl_indexes(nuc_params.A);
                    std::iota(nucl_indexes.begin(), nucl_indexes.end(), 0);
                    std::array<std::vector<double>, 25> sum_tpps;
                    for (uint_fast8_t i = 0; i < 25; i++)
                    {
                        sum_tpps[i] = std::vector<double>(nuc_params.A);
                    }

                    nucl = nucleus_generator::generate_nucleus(
                        nuc_params,
                        true,
                        0.0,
                        0.0,
                        eng);

                    for (auto itt = nucl_indexes.begin(); itt < nucl_indexes.end(); itt++)
                    {
                        auto dummy = this->calculate_sum_tpp(nucl[*itt], nucl);
                        for (uint_fast8_t i = 0; i < 25; i++)
                        {
                            sum_tpps[i][*itt] = std::exp(c_vector[i] * dummy);
                        }
                    }

                    for (uint_fast8_t i = 0; i < 25; i++)
                    {
                        R_vectors[i][*it] = std::reduce(sum_tpps[i].begin(), sum_tpps[i].end(), 0.0) / static_cast<double>(nuc_params.A);
                    }
                }

                for (uint_fast8_t i = 0; i < 25; i++)
                {
                    block_averages[i][block_index] = std::reduce(R_vectors[i].begin(), R_vectors[i].end(), 0.0) / 200.0;
                }
            }
        }

        for (uint_fast8_t i = 0; i < 25; i++)
        {
            ave_R_vector[i] = std::reduce(block_averages[i].begin(), block_averages[i].end(), 0.0) / block_amount;
        }
        std::cout << "A-configs calculated: " << block_amount * 200 << std::endl;

        return std::make_tuple(ave_R_vector, c_vector);
    }

    auto calculate_R_c_table_new(
        const nucleus_generator::nucleus_params &nuc_params,
        const double &rel_tolerance,
        const bool verbose) -> std::tuple<std::array<double, 101>, std::array<double, 101>>
    {
        constexpr std::size_t table_size = 101;
        std::vector<uint_fast8_t> block_indexes(200);
        std::iota(block_indexes.begin(), block_indexes.end(), 0); // generates the list as {0,1,2,3,...}

        std::array<double, table_size> c_vector;
        c_vector.fill(0.0);
        auto loglin = helpers::loglinspace(1e-3, 150.0, table_size / 2);
        for (uint_fast8_t i = 0, j = table_size / 2 - 1; i < table_size / 2; i++, j--)
        {
            c_vector[i] = -loglin[j];
        }
        loglin = helpers::loglinspace(1e-3, 1e15, table_size / 2);
        for (uint_fast8_t i = (table_size / 2) + 1, j = 0; i < table_size; i++, j++)
        {
            c_vector[i] = loglin[j];
        }

        std::array<double, table_size> ave_R_vector;
        ave_R_vector.fill(0.0);

        uint_fast8_t block_amount = 50;
        std::array<std::array<double, 50>, table_size> block_averages;

#pragma omp parallel
        {
            auto eng = std::make_shared<std::mt19937>(static_cast<ulong>((omp_get_thread_num() + 1)) * static_cast<ulong>(std::chrono::system_clock::now().time_since_epoch().count()));

#pragma omp for
            for (uint_fast8_t block_index = 0; block_index < block_amount; block_index++)
            {
                helpers::reseed(*eng, this->seed, 3u, block_index);
                std::array<std::array<double, 200>, table_size> R_vectors;
                for (auto it = block_indexes.begin(); it < block_indexes.end(); it++)
                {
                    std::vector<nucleon> nucl, other;
                    std::vector<uint_fast8_t> nucl_indexes(nuc_params.A);
                    std::iota(nucl_indexes.begin(), nucl_indexes.end(), 0);
                    std::array<std::vector<double>, table_size> sum_tpps;
                    for (uint_fast8_t i = 0; i < table_size; i++)
                    {
                        sum_tpps[i] = std::vector<double>(nuc_params.A);
                    }

                    nucl = nucleus_generator::generate_nucleus(
                        nuc_params,
                        true,
                        0.0,
                        0.0,
                        eng);

                    for (auto itt = nucl_indexes.begin(); itt < nucl_indexes.end(); itt++)
                    {
                        auto dummy = this->calculate_sum_tpp(nucl[*itt], nucl);
                        for (uint_fast8_t i = 0; i < table_size; i++)
                        {
                            double arg = c_vector[i] * dummy;
                            if (arg >= 0.0)
                            {
                                sum_tpps[i][*itt] = 1.0 + std::log(1.0 + arg);
                                // sum_tpps[i][*itt] = std::exp(arg);
                            }
                            else
                            {
                                sum_tpps[i][*itt] = 1.0 / (1.0 - arg);
                                // sum_tpps[i][*itt] = gsl_sf_lambert_W0(-arg) / (-arg);
                            }
                        }
                    }

                    for (uint_fast8_t i = 0; i < table_size; i++)
                    {
                        R_vectors[i][*it] = std::reduce(sum_tpps[i].begin(), sum_tpps[i].end(), 0.0) / static_cast<double>(nuc_params.A);
                    }
                }

                for (uint_fast8_t i = 0; i < table_size; i++)
                {
                    block_averages[i][block_index] = std::reduce(R_vectors[i].begin(), R_vectors[i].end(), 0.0) / 200.0;
                }
            }
        }

        for (uint_fast8_t i = 0; i < table_size; i++)
        {
            ave_R_vector[i] = std::reduce(block_averages[i].begin(), block_averages[i].end(), 0.0) / block_amount;
        }
        std::cout << "A-configs calculated: " << block_amount * 200 << std::endl;

        return std::make_tuple(ave_R_vector, c_vector);
    }

    auto throw_random_peak_indices(
        const std::vector<hotspot_info> &pro_hotspots,
        const std::vector<hotspot_info> &tar_hotspots,
        std::shared_ptr<std::mt19937> random_generator) -> std::tuple<uint_fast8_t, uint_fast8_t>
    {
        std::vector<std::tuple<double, uint_fast8_t, uint_fast8_t>> tpps;
        double total_tpp = 0.0;

        for (uint_fast8_t i = 0; i < pro_hotspots.size(); i++)
        {
            auto [x1, y1, z1] = pro_hotspots[i].co;
            for (uint_fast8_t j = 0; j < tar_hotspots.size(); j++)
            {
                auto [x2, y2, z2] = tar_hotspots[j].co;
                double bsquared = pow(x1 - x2, 2) + pow(y1 - y2, 2);
                double tpp = exp(-bsquared / (4 * this->hs_width_2)); // Normalization doesn't matter
                total_tpp += tpp;
                tpps.push_back({tpp, i, j});
            }
        }

        std::uniform_real_distribution<> dis(0.0, total_tpp);
        double ran_tpp = dis(*random_generator);

        total_tpp = 0.0;
        for (auto [tpp, i, j] : tpps)
        {
            total_tpp += tpp;
            if (total_tpp >= ran_tpp)
            {
                return std::make_tuple(i, j);
            }
        }
    }

private:
    bool hotspots{false};
    double proton_width_2{0.0};
    double hs_width_2{0.0};
    std::function<double(const double &, const std::vector<double> *const)> func{nullptr};
};

#endif // TPP_HPP
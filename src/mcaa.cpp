// A Monte Carlo EKRT event generator, see Phys. Rev. C 111, no.5, 054914 (2025)
// doi:10.1103/PhysRevC.111.054914 https://doi.org/10.48550/arXiv.2406.17592
// Copyright (c) 2025 Mikko Kuha (University of Jyväskylä).
//
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

#include "mcaa.hpp"

mcaa::mcaa(
    const std::string &initfile)
    : jet_params(pqcd::scaled_from_kt, 1, 1.0, false),
      nuc_params(1u, 1u, 1u, 1u, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, false, false, false, 1u, 1.0)
{
    auto
        [p_name,
         p_sjet_name,
         p_read_sigmajets_from_file,
         p_cent_name,
         p_n_events,
         p_b_max,
         p_b_min,
         p_sqrt_s,
         p_K_factor,
         p_Kappa_factor,
         p_p0,
         p_proton_width_static,
         p_proton_width,
         p_sigma_inel_trigger,
         p_envelope_marginal,
         p_A,
         p_B,
         p_ZA,
         p_ZB,
         p_nuclear_RA,
         p_nuclear_RB,
         p_nuclear_dA,
         p_nuclear_dB,
         p_nuclear_beta2A,
         p_nuclear_beta2B,
         p_nuclear_beta3A,
         p_nuclear_beta3B,
         p_nuclear_beta4A,
         p_nuclear_beta4B,
         p_rad_max,
         p_correct_overlap_bias,
         p_nn_min_dist,
         p_shift_cms,
         p_hotspots,
         p_n_hotspots,
         p_hotspot_width,
         p_is_mc_glauber,
         p_sigma_inel_NN,
         p_only_protons,
         p_use_npdfs,
         p_use_snpdfs,
         p_save_endstate_jets,
         p_end_state_filtering,
         p_is_saturation,
         p_is_mom_cons,
         p_is_val_cons,
         p_seed] = io::read_conf(initfile);

    this->hotspots = p_hotspots;
    this->n_hotspots = p_n_hotspots;
    this->end_state_filtering = p_end_state_filtering;
    this->nPDFs = p_use_npdfs;
    this->MC_Glauber = p_is_mc_glauber;
    this->saturation = p_is_saturation;
    this->mom_cons = p_is_mom_cons;
    this->val_cons = p_is_val_cons;
    this->seed = p_seed;
    this->read_sigmajets_from_file = p_read_sigmajets_from_file;
    this->proton_width_static = p_proton_width_static;
    this->save_endstate_jets = p_save_endstate_jets;
    this->only_protons = p_only_protons;
    this->snPDFs = p_use_snpdfs;
    this->name = p_name;
    this->sigmajet_filename = p_sjet_name;
    this->centrality_filename = p_cent_name;
    this->desired_N_events = p_n_events;
    this->A = p_A;
    this->B = p_B;
    this->ZA = p_ZA;
    this->ZB = p_ZB;
    this->b_max = p_b_max;
    this->b_min = p_b_min;
    this->K_factor = p_K_factor;
    this->Kappa_factor = p_Kappa_factor;
    this->p0 = p_p0;
    this->nn_min_dist = p_nn_min_dist;
    this->rad_max = p_rad_max;
    this->sqrt_s = p_sqrt_s;
    this->envelope_marginal = p_envelope_marginal;
    this->sigma_inel_NN = p_sigma_inel_NN;
    this->sigma_inel_trigger = p_sigma_inel_trigger;
    this->p02 = p_p0 * p_p0;
    this->mand_s = p_sqrt_s * p_sqrt_s;

    // Old, redundant parameters
    this->verbose = false;
    this->calculate_end_state = true;
    this->calculate_tata = true;
    this->sigma_inel_from_sigma_jet = false;
    this->AA_inel_same_as_NN = false;
    this->snPDFs_linear = false;
    this->snPDFs_new = true;
    this->is_sat_y_dep = 0;
    this->pt_ordering = true;
    this->sat_first = true;
    this->t03_ordering = false;
    this->rad_min = 0;
    this->hotspot_trigger = p_hotspots;
    this->T_AA_0 = 0;
    if (p_A == 1 && p_B == 1)
    {
        this->is_pp = true;
        this->is_pA = false;
        this->is_AA = false;
    }
    else if (p_A == 1)
    {
        this->is_pp = false;
        this->is_pA = true;
        this->is_AA = false;
    }
    else
    {
        this->is_pp = false;
        this->is_pA = false;
        this->is_AA = true;
    }

    if (!this->AA_inel_same_as_NN && almostEqual(p_sigma_inel_trigger, 0.0)) // triggering sigma from parametrization
    {
        auto s = this->mand_s;
        // fit by COMPETE https://doi.org/10.1103/PhysRevLett.89.201801
        auto sigmatot = 42.6 * std::pow(s, -0.46) - 33.4 * std::pow(s, -0.545) + 0.307 * std::pow(std::log(s / 29.1), 2) + 35.5;
        // fit by TOTEM https://doi.org/10.1140/epjc/s10052-019-6567-0
        auto sigmael = -1.617 * std::log(s) + 0.1359 * std::pow(std::log(s), 2) + 11.84;

        this->sigma_inel_trigger = sigmatot - sigmael;
        std::cout << "Triggering sigma_inel from parametrization: " << this->sigma_inel_trigger << std::endl;
    }

    if (this->proton_width_static)
    {
        this->proton_width = p_proton_width;
        this->proton_width_2 = p_proton_width * p_proton_width;
    }
    else
    {
        this->proton_width_2 = (4.9 + 4 * 0.06 * std::log(this->sqrt_s / 90.0)) / (FMGEV * FMGEV); // C. A. Flett, PhD thesis, U. Liverpool, 2021
        this->proton_width = std::sqrt(this->proton_width_2);
    }
    std::cout << "Proton width: " << this->proton_width << " fm" << std::endl;

    if (this->hotspots)
    {
        this->hs_dist = std::uniform_int_distribution<>(0, this->n_hotspots - 1);
    }

    if (almostEqual(p_hotspot_width, 0.0))
    {
        p_hotspot_width = 0.2 * p_proton_width;
    }
    this->hotspot_width = p_hotspot_width;

    this->Tpp = std::make_shared<Tpp_builder>(this->proton_width_2, p_hotspot_width, p_hotspots);
    this->Tpp->seed = this->seed;

    // Parameter for the envelope function:
    // sigma_jet < A*pT^(-power_law)
    // Larger number == faster simulation, but the
    // inequality must hold for the results to be correct.
    // These are just some values, experiment freely,
    // the simulation will tell if the inequality breaks.
    if (!this->nPDFs) // free proton PDFs
    {
        this->power_law = 2.8;
    }
    else if (!this->snPDFs) // nPDFs
    {
        this->power_law = 2.2;
    }
    else // spatial nPDFs
    {
        this->power_law = 2.0;
    }

    this->jet_params = pqcd::sigma_jet_params(
        /*scale_choice=             */ pqcd::scaled_from_kt,
        /*scalar=                   */ 1.0,
        /*K_factor=                 */ this->K_factor,
        /*use_ses=                  */ false);

    double hotspot_distr_width = std::sqrt(this->proton_width_2 - std::pow(p_hotspot_width, 2));
    this->nuc_params = nucleus_generator::nucleus_params(
        /*A=                        */ this->A,
        /*ZA=                       */ this->ZA,
        /*B=                        */ this->B,
        /*ZB=                       */ this->ZB,
        /*RA=                       */ p_nuclear_RA,
        /*RB=                       */ p_nuclear_RB,
        /*dA=                       */ p_nuclear_dA,
        /*dB=                       */ p_nuclear_dB,
        /*beta2A=                   */ p_nuclear_beta2A,
        /*beta2B=                   */ p_nuclear_beta2B,
        /*beta3A=                   */ p_nuclear_beta3A,
        /*beta3B=                   */ p_nuclear_beta3B,
        /*beta4A=                   */ p_nuclear_beta4A,
        /*beta4B=                   */ p_nuclear_beta4B,
        /*min_distance=             */ p_nn_min_dist,
        /*shift_cms=                */ p_shift_cms,
        /*correct_overlap_bias=     */ p_correct_overlap_bias,
        /*hotspots=                 */ p_hotspots,
        /*n_hotspots=               */ p_n_hotspots,
        /*hotspot_distr_width=      */ hotspot_distr_width);

    this->pdf = std::make_shared<pdf_builder>(
        /*p_pdf_name,p_pdf_setnumber= */ "CT14lo", 0,
        /*n_pdf_name,n_pdf_setnumber= */ "", 0,
        /*snpdf_spatial_cutoff=       */ 0.0,
        /*snpdf_tAA_0=                */ 0.0,
        /*Tpp=                        */ this->Tpp,
        /*nuc_params=                 */ this->nuc_params,
        /*projectile_with_npdfs=      */ (this->is_AA && p_use_npdfs),
        /*target_with_npdfs=          */ (!this->is_pp && p_use_npdfs),
        /*isoscalar_projectile=       */ false,
        /*isoscalar_target=           */ false,
        /*npdfs_spatial=              */ p_use_snpdfs,
        /*snPDFs_linear=              */ false,
        /*snPDFs_new=                 */ true,
        /*only_protons=               */ p_only_protons,
        /*npdf_setnumber=             */ 1,
        /*A=                          */ this->A,  // Pb
        /*B=                          */ this->B,  // Pb
        /*ZA=                         */ this->ZA, // Pb
        /*ZB=                         */ this->ZB, // Pb
        /*verbose=                    */ this->verbose);
}

auto mcaa::fit_sigma_jet_pt0_cutoff(
    double &pt02,
    const double &target,
    const pqcd::sigma_jet_params &params,
    const bool &verbose) noexcept -> double
{
    double sigma_jet = 0.0;
    pt02 = 2.0;

    auto difference_to_target = [&](const double &_p02)
    {
        auto dummy = pqcd::nn_coll_params(
            0.0,
            0.0,
            false,
            false);
        return pqcd::calculate_sigma_jet(this->pdf, &(this->mand_s), &_p02, params, dummy, false) - target;
    };

    helpers::secant_method(&pt02, difference_to_target, 1e-3, &sigma_jet);

    if (verbose)
        std::cout << sqrt(pt02) << ' ' << sigma_jet + target << ' ' << target << std::endl;

    return pt02;
}

auto mcaa::find_sigma_jet_cutoff_Q(
    const double &pt02,
    const double &target,
    const bool &verbose) noexcept -> double
{
    double sigma_jet = 0.0;
    double scalar = 1.0;

    auto difference_to_target = [&](const double &scalar_)
    {
        double p02dummy = 4.0;
        auto jet_params_ = pqcd::sigma_jet_params(
            /*scale_choice=             */ this->jet_params.scale_c,
            /*scalar=                   */ scalar_,
            /*K_factor=                 */ this->K_factor,
            /*use_ses=                  */ this->jet_params.use_ses);
        auto p02_ = this->fit_sigma_jet_pt0_cutoff(p02dummy, target, jet_params_, verbose);
        return p02_ - pt02;
    };

    helpers::secant_method(&scalar, difference_to_target, 1e-3, &sigma_jet);

    if (verbose)
        std::cout << scalar << ' ' << sigma_jet + target << ' ' << target << std::endl;

    return scalar;
}

auto mcaa::check_and_place_circle_among_others(
    std::tuple<double, double, double, uint_fast16_t, double, double> cand_circle,
    std::vector<std::tuple<double, double, double, uint_fast16_t, double, double>> &final_circles,
    uint_fast16_t is_sat_y_dep) noexcept -> bool
{
    auto &[cand_x, cand_y, cand_r, cand_overlap, cand_y1, cand_y2] = cand_circle;

    for (auto &[circ_x, circ_y, circ_r, circ_overlap, y1, y2] : final_circles)
    {
        // If the any of the circles radii overlap with the candidate, just return false
        if (pow((circ_x - cand_x), 2) + pow((circ_y - cand_y), 2) < pow((circ_r + cand_r), 2))
        {
            if (is_sat_y_dep == 1) // Henry's old idea
            {
                auto miy = std::min(y1, y2);
                auto may = std::max(y1, y2);
                auto mciy = std::min(cand_y1, cand_y2);
                auto mcay = std::max(cand_y1, cand_y2);

                if (!(miy > mcay || may < mciy))
                {
                    return false;
                }
            }
            else if (is_sat_y_dep == 2) // all jets have cones
            {
                double delta = 0.5;
                if ((std::abs(y1 - cand_y1) <= delta) || (std::abs(y1 - cand_y2) <= delta) || (std::abs(y2 - cand_y1) <= delta) || (std::abs(y2 - cand_y2) <= delta))
                {
                    return false;
                }
            }
            else if (is_sat_y_dep == 3) // both dijets have cones
            {
                double delta = 0.5;
                auto dy = 0.5 * (y1 + y2);
                auto cand_dy = 0.5 * (cand_y1 + cand_y2);

                if ((std::abs(dy - cand_dy) <= delta))
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
    }

    // None overlapped, emplace the circle among others and return true
    final_circles.emplace_back(std::move(cand_circle));

    return true;
}

auto mcaa::throw_location_for_dijet(
    const dijet_with_ns &dijet,
    std::normal_distribution<double> &normal_dist,
    std::shared_ptr<std::mt19937> random_generator) noexcept -> std::tuple<double, double, double>
{
    auto dz = 0.0;
    auto retx = 0.0;
    auto rety = 0.0;

    if (!this->hotspots)
    {
        auto param = std::normal_distribution<double>::param_type{0., this->proton_width};
        auto dx = normal_dist(*random_generator, param);
        auto dy = normal_dist(*random_generator, param);

        retx = 0.5 * (dijet.pro_nucleon->co.x + dijet.tar_nucleon->co.x + M_SQRT2 * dx);
        rety = 0.5 * (dijet.pro_nucleon->co.y + dijet.tar_nucleon->co.y + M_SQRT2 * dy);
    }
    else
    {
        auto param = std::normal_distribution<double>::param_type{0., this->hotspot_width};
        auto dx = normal_dist(*random_generator, param);
        auto dy = normal_dist(*random_generator, param);

        auto [fst, snd] = this->Tpp->throw_random_peak_indices(dijet.pro_nucleon->hotspots, dijet.tar_nucleon->hotspots, random_generator);
        retx = 0.5 * (dijet.pro_nucleon->hotspots.at(fst).co.x + dijet.tar_nucleon->hotspots.at(snd).co.x + M_SQRT2 * dx);
        rety = 0.5 * (dijet.pro_nucleon->hotspots.at(fst).co.y + dijet.tar_nucleon->hotspots.at(snd).co.y + M_SQRT2 * dy);
    }

    return std::make_tuple(retx, rety, dz);
}

auto mcaa::sqrtalphas(double Q) noexcept -> double
{
    auto alphas = this->pdf->alphasQ(Q);
    return std::sqrt(alphas);
}

auto mcaa::filter_end_state(
    std::vector<nn_coll> &binary_collisions,
    std::vector<dijet_with_coords> &filtered_scatterings,
    std::shared_ptr<std::mt19937> random_generator,
    const std::vector<nucleon> &pro,
    const std::vector<nucleon> &tar,
    const bool sat_first) noexcept -> void
{
    std::vector<dijet_with_ns> candidates;
    candidates.reserve(binary_collisions.size() * 10); // 10 events on average is just an overhead guess

    std::unordered_map<nucleon *, double> x1s;
    std::unordered_map<nucleon *, double> x2s;

    std::unordered_map<nucleon *, std::tuple<uint_fast8_t, uint_fast8_t>> valence1_nums;
    std::unordered_map<nucleon *, std::tuple<uint_fast8_t, uint_fast8_t>> valence2_nums;

    std::normal_distribution<double> normal_dist(0, 0);

    for (auto &col : binary_collisions)
    {
        for (auto &dij : col.dijets)
        {
            candidates.emplace_back(std::move(dij), col.projectile, col.target, this->pt_ordering, this->t03_ordering);
        }
    }
    candidates.shrink_to_fit();

    std::vector<std::tuple<double, double, double, uint_fast16_t, double, double>> final_circles;
    final_circles.reserve(candidates.size());
    filtered_scatterings.reserve(filtered_scatterings.size() + candidates.size());

    std::sort(candidates.begin(), candidates.end(), // Sort the candidate events so that the one with the smallest t0 is first
              [](dijet_with_ns &s1, dijet_with_ns &s2)
              { return (s1.t0 < s2.t0); });

    double kt;
    double y1, y2;
    double i_x1_sum_to_be, i_x2_sum_to_be;
    bool valence_down;
    bool proton;
    double num_vu1_to_be, num_vd1_to_be;
    double num_vu2_to_be, num_vd2_to_be;
    double tata = 0.0;

    if (!sat_first)
    {
        for (auto &cand : candidates)
        {
            kt = cand.dijet.kt;
            y1 = cand.dijet.y1;
            y2 = cand.dijet.y2;

            if (this->val_cons)
            {
                // VALENCE QUARKS CONSERVATION

                if (cand.dijet.valence1)
                {
                    proton = !cand.pro_nucleon->is_neutron;
                    if (cand.dijet.init1 == 1)
                    {
                        valence_down = true;
                        num_vu1_to_be = 0;
                        num_vd1_to_be = 1;
                    }
                    else if (cand.dijet.init1 == 2)
                    {
                        valence_down = false;
                        num_vu1_to_be = 1;
                        num_vd1_to_be = 0;
                    }
                    else
                    {
                        std::cout << "init1=" << cand.dijet.init1 << " claims to be valence" << std::endl;
                        exit(1);
                    }

                    auto valence_nums = valence1_nums.find(cand.pro_nucleon);
                    if (valence_nums != valence1_nums.end())
                    {
                        auto [num_vu, num_vd] = valence_nums->second;
                        num_vu1_to_be = num_vu + (!valence_down);
                        num_vd1_to_be = num_vd + valence_down;

                        if (num_vu1_to_be > 1 + proton) // Too many valence quarks interacting --> discard
                        {
                            continue;
                        }
                        if (num_vd1_to_be > 1 + (!proton)) // Too many valence quarks interacting --> discard
                        {
                            continue;
                        }
                    }
                }

                if (cand.dijet.valence2)
                {
                    proton = !cand.tar_nucleon->is_neutron;
                    if (cand.dijet.init2 == 1)
                    {
                        valence_down = true;
                        num_vu2_to_be = 0;
                        num_vd2_to_be = 1;
                    }
                    else if (cand.dijet.init2 == 2)
                    {
                        valence_down = false;
                        num_vu2_to_be = 1;
                        num_vd2_to_be = 0;
                    }
                    else
                    {
                        std::cout << "init2=" << cand.dijet.init2 << " claims to be valence" << std::endl;
                        exit(1);
                    }

                    auto valence_nums = valence2_nums.find(cand.tar_nucleon);
                    if (valence_nums != valence2_nums.end())
                    {
                        auto [num_vu, num_vd] = valence_nums->second;
                        num_vu2_to_be = num_vu + (!valence_down);
                        num_vd2_to_be = num_vd + valence_down;

                        if (num_vu2_to_be > 1 + proton) // Too many valence quarks interacting --> discard
                        {
                            continue;
                        }
                        if (num_vd2_to_be > 1 + (!proton)) // Too many valence quarks interacting --> discard
                        {
                            continue;
                        }
                    }
                }
            }

            if (this->mom_cons)
            {
                // MOMENTUM CONSERVATION

                auto x1 = (kt / this->sqrt_s) * (exp(y1) + exp(y2));
                auto x2 = (kt / this->sqrt_s) * (exp(-y1) + exp(-y2));

                i_x1_sum_to_be = x1;
                i_x2_sum_to_be = x2;

                auto i_x1_sum = x1s.find(cand.pro_nucleon);
                if (i_x1_sum != x1s.end())
                {
                    i_x1_sum_to_be = i_x1_sum->second + x1;

                    if (i_x1_sum_to_be > 1.0) // Energy budget broken --> discard
                    {
                        continue;
                    }
                }

                auto i_x2_sum = x2s.find(cand.tar_nucleon);
                if (i_x2_sum != x2s.end())
                {
                    i_x2_sum_to_be = i_x2_sum->second + x2;

                    if (i_x2_sum_to_be > 1.0) // Energy budget broken --> discard
                    {
                        continue;
                    }
                }
            }

            auto [cand_x, cand_y, cand_z] = this->throw_location_for_dijet(cand, normal_dist, random_generator);

            if (this->saturation)
            {
                // SATURATION

                double upstairs = 1.0;
                if (this->is_sat_y_dep == 4)
                {
                    upstairs = this->sqrtalphas(cand.dijet.kt);
                }
                else if (this->is_sat_y_dep == 5)
                {
                    double dy = y1 - y2;
                    double dsigma = std::pow((1 + 2.0 * std::cosh(dy)) / (2 + 2.0 * std::cosh(dy)), 1.5);
                    upstairs = this->sqrtalphas(cand.dijet.kt) * dsigma;
                }

                auto cand_circle = std::make_tuple(cand_x, cand_y, upstairs / (cand.dijet.kt * this->Kappa_factor * FMGEV), 1, y1, y2);

                if (!mcaa::check_and_place_circle_among_others(std::move(cand_circle), final_circles, this->is_sat_y_dep))
                {
                    continue; // Did not fit into saturated PS --> discard
                }
            }

            if (this->calculate_tata)
            {
                // nucleon dummy{coords{cand_x, cand_y, cand_z}, 0};
                // tata = this->Tpp.calculate_sum_tpp(dummy, pro) * this->Tpp.calculate_sum_tpp(dummy, tar);
                tata = this->Tpp->calculate_TA(cand_x, cand_y, pro) * this->Tpp->calculate_TA(cand_x, cand_y, tar);
            }

            if (this->val_cons)
            {
                if (cand.dijet.valence1)
                {
                    valence1_nums.insert_or_assign(cand.pro_nucleon, std::make_tuple(num_vu1_to_be, num_vd1_to_be));
                }
                if (cand.dijet.valence2)
                {
                    valence2_nums.insert_or_assign(cand.tar_nucleon, std::make_tuple(num_vu2_to_be, num_vd2_to_be));
                }
            }

            if (this->mom_cons)
            {
                x1s.insert_or_assign(cand.pro_nucleon, i_x1_sum_to_be);
                x2s.insert_or_assign(cand.tar_nucleon, i_x2_sum_to_be);
            }

            filtered_scatterings.push_back({cand.dijet, coords{cand_x, cand_y, cand_z}, tata, cand.pro_nucleon, cand.tar_nucleon});
        }
    }
    else
    {
        std::vector<std::tuple<dijet_with_ns, double, double, double>> candidates_sat;
        for (auto &cand : candidates)
        {
            kt = cand.dijet.kt;
            y1 = cand.dijet.y1;
            y2 = cand.dijet.y2;

            auto [cand_x, cand_y, cand_z] = this->throw_location_for_dijet(cand, normal_dist, random_generator);

            if (this->saturation)
            {
                // SATURATION

                double upstairs = 1.0;

                if (this->is_sat_y_dep == 4)
                {
                    upstairs = this->sqrtalphas(cand.dijet.kt);
                }
                else if (this->is_sat_y_dep == 5)
                {
                    double dy = y1 - y2;
                    double dsigma = std::pow((1 + 2.0 * std::cosh(dy)) / (2 + 2.0 * std::cosh(dy)), 1.5);
                    upstairs = this->sqrtalphas(cand.dijet.kt) * dsigma;
                }

                auto cand_circle = std::make_tuple(cand_x, cand_y, upstairs / (cand.dijet.kt * this->Kappa_factor * FMGEV), 1, y1, y2);

                if (!mcaa::check_and_place_circle_among_others(std::move(cand_circle), final_circles, this->is_sat_y_dep))
                {
                    continue; // Did not fit into saturated PS --> discard
                }
            }

            candidates_sat.push_back({cand, cand_x, cand_y, cand_z});
        }

        for (auto &[cand, cand_x, cand_y, cand_z] : candidates_sat)
        {
            kt = cand.dijet.kt;
            y1 = cand.dijet.y1;
            y2 = cand.dijet.y2;

            if (this->val_cons)
            {
                // VALENCE QUARKS CONSERVATION

                if (cand.dijet.valence1)
                {
                    proton = !cand.pro_nucleon->is_neutron;
                    if (cand.dijet.init1 == 1)
                    {
                        valence_down = true;
                        num_vu1_to_be = 0;
                        num_vd1_to_be = 1;
                    }
                    else if (cand.dijet.init1 == 2)
                    {
                        valence_down = false;
                        num_vu1_to_be = 1;
                        num_vd1_to_be = 0;
                    }
                    else
                    {
                        std::cout << "init1=" << cand.dijet.init1 << " claims to be valence" << std::endl;
                        exit(1);
                    }

                    auto valence_nums = valence1_nums.find(cand.pro_nucleon);
                    if (valence_nums != valence1_nums.end())
                    {
                        auto [num_vu, num_vd] = valence_nums->second;
                        num_vu1_to_be = num_vu + (!valence_down);
                        num_vd1_to_be = num_vd + valence_down;

                        if (num_vu1_to_be > 1 + proton) // Too many valence quarks interacting --> discard
                        {
                            continue;
                        }
                        if (num_vd1_to_be > 1 + (!proton)) // Too many valence quarks interacting --> discard
                        {
                            continue;
                        }
                    }
                }

                if (cand.dijet.valence2)
                {
                    proton = !cand.tar_nucleon->is_neutron;
                    if (cand.dijet.init2 == 1)
                    {
                        valence_down = true;
                        num_vu2_to_be = 0;
                        num_vd2_to_be = 1;
                    }
                    else if (cand.dijet.init2 == 2)
                    {
                        valence_down = false;
                        num_vu2_to_be = 1;
                        num_vd2_to_be = 0;
                    }
                    else
                    {
                        std::cout << "init2=" << cand.dijet.init2 << " claims to be valence" << std::endl;
                        exit(1);
                    }

                    auto valence_nums = valence2_nums.find(cand.tar_nucleon);
                    if (valence_nums != valence2_nums.end())
                    {
                        auto [num_vu, num_vd] = valence_nums->second;
                        num_vu2_to_be = num_vu + (!valence_down);
                        num_vd2_to_be = num_vd + valence_down;

                        if (num_vu2_to_be > 1 + proton) // Too many valence quarks interacting --> discard
                        {
                            continue;
                        }
                        if (num_vd2_to_be > 1 + (!proton)) // Too many valence quarks interacting --> discard
                        {
                            continue;
                        }
                    }
                }
            }

            if (this->mom_cons)
            {
                // MOMENTUM CONSERVATION
                auto x1 = (kt / this->sqrt_s) * (exp(y1) + exp(y2));
                auto x2 = (kt / this->sqrt_s) * (exp(-y1) + exp(-y2));
                i_x1_sum_to_be = x1;
                i_x2_sum_to_be = x2;

                auto i_x1_sum = x1s.find(cand.pro_nucleon);
                if (i_x1_sum != x1s.end())
                {
                    i_x1_sum_to_be = i_x1_sum->second + x1;

                    if (i_x1_sum_to_be > 1.0) // Energy budget broken --> discard
                    {
                        continue;
                    }
                }

                auto i_x2_sum = x2s.find(cand.tar_nucleon);
                if (i_x2_sum != x2s.end())
                {
                    i_x2_sum_to_be = i_x2_sum->second + x2;

                    if (i_x2_sum_to_be > 1.0) // Energy budget broken --> discard
                    {
                        continue;
                    }
                }
            }

            if (this->calculate_tata)
            {
                // nucleon dummy{coords{cand_x, cand_y, cand_z}, 0};
                // tata = this->Tpp.calculate_sum_tpp(dummy, pro) * this->Tpp.calculate_sum_tpp(dummy, tar);
                tata = this->Tpp->calculate_TA(cand_x, cand_y, pro) * this->Tpp->calculate_TA(cand_x, cand_y, tar);
            }

            if (this->val_cons)
            {
                if (cand.dijet.valence1)
                {
                    valence1_nums.insert_or_assign(cand.pro_nucleon, std::make_tuple(num_vu1_to_be, num_vd1_to_be));
                }
                if (cand.dijet.valence2)
                {
                    valence2_nums.insert_or_assign(cand.tar_nucleon, std::make_tuple(num_vu2_to_be, num_vd2_to_be));
                }
            }

            if (this->mom_cons)
            {
                x1s.insert_or_assign(cand.pro_nucleon, i_x1_sum_to_be);
                x2s.insert_or_assign(cand.tar_nucleon, i_x2_sum_to_be);
            }

            filtered_scatterings.push_back({cand.dijet, coords{cand_x, cand_y, cand_z}, tata, cand.pro_nucleon, cand.tar_nucleon});
        }
    }

    binary_collisions.clear();
}

//--------------------------------------------//
// THESE ARE A HACK TO SAVE THE WORK THIS FAR //
// IN THE CASE OF SUDDEN TERMINATION          //
//--------------------------------------------//
volatile std::atomic_bool user_aborted = false;
volatile std::atomic_bool g_bug_bool = false;
void abort_handler(int num)
{
    user_aborted = true;
    std::cout << std::endl
              << "Abort called " << num << std::endl;
}

auto mcaa::run() -> void
{
    std::cout << std::endl
              << "Doing the run " << this->name << std::endl;

    if (this->verbose)
        std::cout << "Initializing..." << std::flush;

    uint_fast32_t AA_events_done = 0;
    std::mutex AA_events_mutex;

    // auto eng = std::make_shared<std::mt19937>(static_cast<ulong>(1));
    auto eng_shared = std::make_shared<std::mt19937>(static_cast<ulong>(std::chrono::system_clock::now().time_since_epoch().count()));
    helpers::reseed(*eng_shared, this->seed, 5u, 0u);
    std::uniform_real_distribution<double> unirand{0.0, 1.0};

    std::vector<io::Coll> collisions_for_reporting;        /////
    std::vector<io::Coll> collisions_for_reporting_midrap; /////
    std::mutex colls_mutex;
    std::vector<double> kt_bins{helpers::loglinspace(p0, 100, 16u)};
    auto kt_v_dummy = helpers::loglinspace(200, sqrt_s / 2.0, 5u);
    for (auto ktdummy : kt_v_dummy)
    {
        kt_bins.push_back(ktdummy);
    }

    double ylim = static_cast<double>(log(this->sqrt_s / this->p0));
    const std::vector<double> y_bins{helpers::linspace(-ylim, ylim, 40u)};
    const std::vector<double> b_bins{helpers::linspace(this->b_min, this->b_max, 21u)};
    std::vector<double> et_bins{helpers::loglinspace(2 * p0, 30000, 31u)};

    // std::vector<double> Rs_as_vector, cs_as_vector;

    // //if linear snPDFs
    // if (this->snPDFs && (this->T_AA_0 == 0.0) && this->snPDFs_linear)
    // {
    //     std::cout<<"Calculating T_AA(0)"<<std::endl;

    //     this->T_AA_0 = this->Tpp->calculate_T_AA_0
    //     (
    //         this->nuc_params,
    //         1e-5,
    //         verbose
    //     );

    //     std::cout<<"Calculated T_AA(0) = "<< this->T_AA_0 <<std::endl;
    // } //if exponential snPDFs
    // else if (this->snPDFs && (this->T_AA_0 == 0.0))
    // {
    //     std::cout<<"Calculating R_A - c_A table"<<std::endl;

    //     auto [R_A_table, c_A_table] = this->Tpp->calculate_R_c_table
    //     (
    //         this->nuc_params,
    //         1e-5,
    //         verbose
    //     );
    //     std::cout<<"Done! {c,R} pairs:"<<std::endl;

    //     for (uint_fast8_t i=0; i<24; i++)
    //     {
    //         std::cout<<"{"<<c_A_table[i]<<","<<R_A_table[i]<<"},";
    //     }
    //     std::cout<<"{"<<c_A_table[24]<<","<<R_A_table[24]<<"}}"<<std::endl;

    //     for (uint_fast8_t i=0; i<25; i++)
    //     {
    //         Rs_as_vector.push_back(R_A_table[i]);
    //         cs_as_vector.push_back(c_A_table[i]);
    //     }

    //     auto c_A_from_R = linear_interpolator(Rs_as_vector, cs_as_vector);

    //     this->jet_params = pqcd::sigma_jet_params(
    //     /*scale_choice=             */this->jet_params.scale_c,
    //     /*scalar=                   */this->jet_params.scalar,
    //     /*K_factor=                 */this->K_factor,
    //     /*use_ses=                  */this->jet_params.use_ses);
    // }

    auto
        [dijet_norm,
         sigma_jet,
         mand_s_array,
         env_func,
         sqrt_s_array] =
            calcs::prepare_sigma_jets(
                this->read_sigmajets_from_file,
                this->pdf,
                this->mand_s,
                this->sqrt_s,
                this->p02,
                this->p0,
                this->power_law,
                this->envelope_marginal,
                this->jet_params,
                this->sigmajet_filename,
                this->snPDFs);

    if (this->sigma_inel_from_sigma_jet && !this->snPDFs)
    {

        auto dummy = std::get<double>(sigma_jet) / (4 * M_PI * this->proton_width_2);
        dummy = dummy / 10; // mb -> fm²

        this->sigma_inel_NN = (4 * M_PI * this->proton_width_2) * (M_EULER + std::log(dummy) + gsl_sf_expint_E1(dummy)) * 10; // 1 mb = 0.1 fm^2
        std::cout << "sigma_inel_NN = " << this->sigma_inel_NN << std::endl;
    }
    else if (this->sigma_inel_from_sigma_jet)
    {
        auto dummy = dijet_norm / (4 * M_PI * this->proton_width_2);
        dummy = dummy / 10; // mb -> fm²

        this->sigma_inel_NN = (4 * M_PI * this->proton_width_2) * (M_EULER + std::log(dummy) + gsl_sf_expint_E1(dummy)) * 10; // 1 mb = 0.1 fm^2
        std::cout << "sigma_inel_NN = " << this->sigma_inel_NN << std::endl;
    }

    if (this->AA_inel_same_as_NN)
    {
        this->sigma_inel_trigger = this->sigma_inel_NN;
    }

    bool use_nn_b2_max = true; // TODO GET RID OF MAGIC NUMBERS
    double tpp_min = 1e-8;     // TODO GET RID OF MAGIC NUMBERS

    double nn_b2_max = 1.0;
    if (use_nn_b2_max)
    {
        double dummy = 0; // not needed here
        auto difference_to_target = [tpp_min, tpp = this->Tpp](const double &b2)
        {
            return tpp_min - tpp->at(b2);
        };
        helpers::secant_method(&nn_b2_max, difference_to_target, tpp_min, &dummy);
        if (this->verbose)
        {
            std::cout << "Found b2max = " << nn_b2_max << ", Tpp(b2max) = " << this->Tpp->at(nn_b2_max) << std::endl;
        }
    }

    if (this->verbose)
        std::cout << "Done!" << std::endl;

    AA_collision_params coll_params{
        /*mc_glauber_mode=          */ this->MC_Glauber,
        /*pp_scattering=            */ this->is_pp,
        /*pA_scattering=            */ this->is_pA,
        /*spatial_pdfs=             */ this->snPDFs,
        /*calculate_end_state=      */ this->calculate_end_state,
        /*use_nn_b2_max=            */ use_nn_b2_max,
        /*sigma_inel=               */ this->sigma_inel_NN,
        /*Tpp=                      */ this->Tpp,
        /*normalize_to=             */ B2_normalization_mode::inelastic,
        /*sqrt_s=                   */ this->sqrt_s,
        /*energy_threshold=         */ this->p0,
        /*nn_b2_max=                */ nn_b2_max,
        /*T_AA_0=                   */ this->T_AA_0};

    std::vector<std::tuple<io::Coll, std::vector<dijet_with_coords>>> colls_scatterings;

    // auto cmpLambda = [](const std::tuple<double, double> &lhs, const std::tuple<double, double> &rhs) { return std::get<0>(lhs) < std::get<0>(rhs); }; /////
    // std::map<io::Coll, std::tuple<double, double>, decltype(cmpLambda)> colls_sumet_midet(cmpLambda); /////

    //    OBSERVABLES TO BE SAVED
    //
    //    histo_2d jets{kt_bins, y_bins};
    //    histo_2d dijets{kt_bins, y_bins};
    //    std::mutex jets_mutex;
    //    std::mutex dijets_mutex;
    //
    //    histo_1d dETdy{y_bins};
    //    histo_1d dEdy{y_bins};
    //    histo_1d dNdy{y_bins};
    //    histo_1d dNdET{et_bins};
    //    std::mutex dETdy_mutex;
    //    std::mutex dEdy_mutex;
    //    std::mutex dNdy_mutex;
    //    std::mutex dNdET_mutex;
    //
    //    histo_1d dETdeta{y_bins};
    //    histo_1d dEdeta{y_bins};
    //    std::mutex dETdeta_mutex;
    //    std::mutex dEdeta_mutex;
    //
    //    histo_1d dETdb{b_bins};
    //    histo_1d dEdb{b_bins};
    //    std::mutex dETdb_mutex;
    //    std::mutex dEdb_mutex;

    std::ofstream total_energy;    /////
    std::mutex total_energy_mutex; /////

    total_energy.open("total_energies_" + this->name + ".dat"); /////
    total_energy << "///Sum E_T Sum E" << std::endl;            /////

    std::vector<uint_fast64_t> event_indexes(this->desired_N_events);
    std::iota(event_indexes.begin(), event_indexes.end(), 0); // generates the list as {0,1,2,3,...}
    std::atomic<uint_fast64_t> running_count{desired_N_events};

    //--------------------------------------------//
    // THESE ARE A HACK TO SAVE THE WORK THIS FAR //
    // IN THE CASE OF A UNEXPECTED TERMINATION    //
    //--------------------------------------------//
    std::signal(SIGINT, abort_handler);
    std::set_terminate([]()
                       {
                           std::cout << std::endl
                                     << "Unhandled exception" << std::endl;
                           // g_bug_bool = true;
                       });

    try
    {
        //----------------------------------------------------------------//
        // THIS IS THE PARALLELLIZED MAIN LOOP FOR ALL OF THE EVENTS      //
        // INCLUDES NUCLEUS GENERATION, COLLISION CALCULATIONS, FILTERING //
        // AND OBSERVABLE CALCULATIONS                                    //
        //----------------------------------------------------------------//

#pragma omp parallel
        {
            auto eng = std::make_shared<std::mt19937>(static_cast<ulong>((omp_get_thread_num() + 1)) * static_cast<ulong>(std::chrono::system_clock::now().time_since_epoch().count()));

#pragma omp for
            for (auto it = event_indexes.begin(); it < event_indexes.end(); it++)
            {
                if (user_aborted)
                {
                    continue;
                }
                // one random stream per event index: with a seed, the events do not depend on the threads
                helpers::reseed(*eng, this->seed, 4u, *it);
                do // while (g_bug_bool)
                {
                    uint_fast32_t NColl = 0;
                    std::vector<nn_coll> binary_collisions;
                    std::vector<dijet_with_coords> filtered_scatterings;
                    double impact_parameter;
                    double b_min2 = this->b_min * this->b_min;
                    double b_max2 = this->b_max * this->b_max;

                    std::vector<nucleon> pro, tar;
                    uint_fast16_t times_discarded = 0;

                    // Keep generating nuclei until the triggering condition is met
                    bool bugged, triggered;
                    // "ball" diameter = distance at which two nucleons interact in MC Glauber
                    const double d2 = this->sigma_inel_trigger / (M_PI * 10.0); // in fm^2
                    do                                                          // while (!triggered || bugged)
                    {
                        // B^2 from a uniform distribution
                        impact_parameter = sqrt(b_min2 + unirand(*eng) * (b_max2 - b_min2));

                        times_discarded++;
                        if (times_discarded > 1000)
                        {
                            std::cout << std::endl
                                      << "Generated nuclei discarded over 1000 times. "
                                      << "Check impact parameters and/or collsion probabilities."
                                      << std::endl;
                            times_discarded = 0;
                        }

                        bugged = false;
                        triggered = false;
                        try
                        {
                            auto [pro_dummy, tar_dummy] = calcs::generate_nuclei(
                                this->nuc_params,
                                this->sqrt_s,
                                impact_parameter,
                                eng,
                                verbose);
                            pro = std::move(pro_dummy);
                            tar = std::move(tar_dummy);
                        }
                        catch (const std::exception &e)
                        {
                            std::cout << e.what() << " in main, trying again" << std::endl;
                            bugged = true;
                        }

                        for (auto A : pro)
                        {
                            for (auto B : tar)
                            {
                                // "ball" diameter = distance at which two nucleons interact
                                double dij2 = 100.0;

                                // Nucleon distance is the distance of the closest hotspots
                                if (hotspot_trigger)
                                {
                                    dij2 = A.calculate_bsquared_hotspot(B);
                                }
                                else // Nucleon distance is the distance of the nucleon centers
                                {
                                    dij2 = A.calculate_bsquared(B);
                                }

                                if (dij2 <= d2) // triggering condition
                                {
                                    triggered = true;
                                    continue;
                                }
                            }
                            if (triggered)
                            {
                                continue;
                            }
                        }

                    } while (!triggered || bugged);

                    // Demand at least one hard scattering
                    uint_fast16_t times_collided = 0;
                    do // while (NColl<1)
                    {
                        times_collided++;
                        binary_collisions.clear();
                        if (verbose)
                            std::cout << "impact_parameter: " << impact_parameter << std::endl;

                        NColl = calcs::collide_nuclei(
                            pro,
                            tar,
                            binary_collisions,
                            sigma_jet,
                            unirand,
                            eng,
                            coll_params,
                            this->jet_params,
                            this->p0,
                            this->pdf,
                            this->power_law,
                            env_func,
                            verbose,
                            this->Kappa_factor,
                            this->proton_width,
                            this->is_sat_y_dep);

                        if (times_collided > 1000)
                        {
                            std::cout << std::endl
                                      << "The same collision is collided 1000 times with no "
                                      << "jets in the output. Check the triggering condition."
                                      << std::endl;
                            times_collided = 0;
                        }
                    } while (NColl < 1);

                    double sum_ET = 0;
                    double sum_ET_midrap = 0;

                    if (end_state_filtering)
                    {
                        double sum_E = 0;
                        this->filter_end_state(
                            binary_collisions,
                            filtered_scatterings,
                            eng,
                            pro,
                            tar,
                            this->sat_first);

                        // std::vector<std::tuple<double, double> > new_jets;
                        // std::vector<std::tuple<double, double> > new_dijets;
                        // std::vector<std::tuple<double, double> > new_ET_y;
                        // std::vector<std::tuple<double, double> > new_E_y;
                        // std::vector<std::tuple<double, double> > new_N_y;
                        // std::vector<std::tuple<double, double> > new_ET_eta;
                        // std::vector<std::tuple<double, double> > new_E_eta;

                        for (auto e_co : filtered_scatterings)
                        {
                            auto e = e_co.dijet;

                            // new_jets.emplace_back(e.kt, e.y1);
                            // new_jets.emplace_back(e.kt, e.y2);

                            // new_dijets.emplace_back(e.kt, 0.5*(e.y1+e.y2));

                            // new_ET_y.emplace_back(e.y1, e.kt);
                            // new_ET_y.emplace_back(e.y2, e.kt);

                            // new_ET_eta.emplace_back(0.5*(e.y1+e.y2), 2*e.kt);

                            // new_E_y.emplace_back(e.y1, e.kt*cosh(e.y1));
                            // new_E_y.emplace_back(e.y2, e.kt*cosh(e.y2));

                            // new_N_y.emplace_back(e.y1, 1);
                            // new_N_y.emplace_back(e.y2, 1);

                            // new_E_eta.emplace_back(0.5*(e.y1+e.y2), e.kt*(cosh(e.y1) + cosh(e.y2)));

                            sum_ET += 2 * e.kt;
                            sum_E += e.kt * (cosh(e.y1) + cosh(e.y2)); /////

                            if (e.y1 >= -0.5 && e.y1 <= 0.5)
                            {
                                sum_ET_midrap += e.kt;
                            }
                            if (e.y2 >= -0.5 && e.y2 <= 0.5)
                            {
                                sum_ET_midrap += e.kt;
                            }
                        }

                        // {
                        //    const std::lock_guard<std::mutex> lock(jets_mutex);
                        //    jets.add(new_jets);
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dijets_mutex);
                        //    dijets.add(new_dijets);
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dETdy_mutex);
                        //    dETdy.add(new_ET_y);
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dEdy_mutex);
                        //    dEdy.add(new_E_y);
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dNdy_mutex);
                        //    dNdy.add(new_N_y);
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dETdeta_mutex);
                        //    dETdeta.add(new_ET_eta);
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dEdeta_mutex);
                        //    dEdeta.add(new_E_eta);
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dNdET_mutex);
                        //    dNdET.add(std::make_tuple(sum_ET, 1));
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dETdb_mutex);
                        //    dETdb.add(std::make_tuple(impact_parameter, sum_ET));
                        // }

                        // {
                        //    const std::lock_guard<std::mutex> lock(dEdb_mutex);
                        //    dEdb.add(std::make_tuple(impact_parameter, sum_E));
                        // }

                        {                                                               /////
                            const std::lock_guard<std::mutex> lock(total_energy_mutex); /////
                            total_energy << sum_ET << ' ' << sum_E << std::endl;        /////
                        } /////
                    }

                    {
                        const std::lock_guard<std::mutex> lock(AA_events_mutex);
                        AA_events_done++;
                        if (AA_events_done % 100 == 0)
                        {
                            std::cout << "\rA+A collisions calculated: " << AA_events_done << std::flush;
                        }
                    }

                    uint_fast32_t Npart = 0;
                    for (auto &A : pro)
                    {
                        if (A.wounded)
                        {
                            Npart++;
                        }
                    }
                    for (auto &B : tar)
                    {
                        if (B.wounded)
                        {
                            Npart++;
                        }
                    }

                    {
                        const std::lock_guard<std::mutex> lock(colls_mutex);
                        io::Coll coll(NColl, Npart, 2 * filtered_scatterings.size(), impact_parameter, sum_ET);
                        io::Coll coll_midrap(NColl, Npart, 2 * filtered_scatterings.size(), impact_parameter, sum_ET_midrap);
                        collisions_for_reporting.push_back(coll);
                        collisions_for_reporting_midrap.push_back(coll_midrap);

                        if (save_endstate_jets)
                        {
                            colls_scatterings.push_back({coll, filtered_scatterings});
                        }
                    }
                } while (g_bug_bool);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::endl
                  << "Threw " << e.what() << std::endl;
    }
    std::cout << " ...done!" << std::endl;

    // io::print_histos
    //(
    //     name_postfix,
    //     jets,
    //     dijets,
    //     dETdy,
    //     dEdy,
    //     dNdy,
    //     dNdET,
    //     dETdeta,
    //     dEdeta,
    //     dETdb,
    //     dEdb,
    //     dijet_norm,
    //     AA_events_done
    //);
    if (collisions_for_reporting.size() >= 50)
    {
        uint_fast8_t nBins = 18;
        double binsLow[] = {0., 0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 0.0, 0.0, 0.0, 0.0};
        double binsHigh[] = {0.02, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 0.05, 0.1, 0.8, 1.0};
        std::ofstream glauber_report_file;
        std::string g_name{"g_report_" + this->name + ".dat"};
        std::string g_name_midrap{"g_report_midrap_" + this->name + ".dat"};

        glauber_report_file.open(g_name, std::ios::out);
        io::mc_glauber_style_report(collisions_for_reporting, this->sigma_inel_NN, collisions_for_reporting.size(), nBins, binsLow, binsHigh, glauber_report_file);
        glauber_report_file.close();
        glauber_report_file.open(g_name_midrap, std::ios::out);
        io::mc_glauber_style_report(collisions_for_reporting_midrap, this->sigma_inel_NN, collisions_for_reporting_midrap.size(), nBins, binsLow, binsHigh, glauber_report_file);
        glauber_report_file.close();

        if (save_endstate_jets)
        {
            std::vector<std::tuple<double, double>> centBins;
            std::ifstream cents_input(this->centrality_filename);
            if (!cents_input.is_open())
            {
                std::cout << "Could not open " << this->centrality_filename << std::endl;
                centBins.push_back(std::tuple<double, double>{0.0, 1.0});
                return;
            }
            std::string line;
            for (; std::getline(cents_input, line);)
            {
                std::stringstream ss(line);
                std::string token;
                std::getline(ss, token, ',');
                double centLow = std::stod(token);
                std::getline(ss, token, ',');
                double centHigh = std::stod(token);
                centBins.push_back(std::tuple<double, double>{centLow, centHigh});
            }
            cents_input.close();

            std::string name_pfs{this->name + ".dat"};

            std::ofstream jet_file;

            std::sort(colls_scatterings.begin(), colls_scatterings.end(), [](const std::tuple<io::Coll, std::vector<dijet_with_coords>> &a, const std::tuple<io::Coll, std::vector<dijet_with_coords>> &b)
                      {
                auto [ac, as] = a;
                auto [bc, bs] = b;
                return ac.getET() < bc.getET(); });

            for (auto [centLow, centHigh] : centBins)
            {
                std::stringstream jetsname{""};
                jetsname << "jets_" << static_cast<uint_fast16_t>(centLow * 100) << "_" << static_cast<uint_fast16_t>(centHigh * 100) << "_" << name_pfs;
                jet_file.open(jetsname.str(), std::ios::out | std::ios::binary);

                histo_1d dETdy_by_cent{y_bins};
                histo_1d dEdy_by_cent{y_bins};
                std::vector<std::tuple<double, double>> new_ET_y;
                std::vector<std::tuple<double, double>> new_E_y;

                uint_fast64_t N_evts_tot = colls_scatterings.size();
                // Make sure that no rounding downwards.
                double eps = 0.1 / static_cast<double>(N_evts_tot);

                uint_fast64_t lower_ind = static_cast<uint_fast64_t>(centLow * static_cast<double>(N_evts_tot) + eps);
                uint_fast64_t upper_ind = static_cast<uint_fast64_t>(centHigh * static_cast<double>(N_evts_tot) + eps);
                uint_fast64_t n_in_bin = upper_ind - lower_ind;

                // total number of events in this bin
                jet_file.write(reinterpret_cast<char *>(&n_in_bin), sizeof n_in_bin);

                auto it = colls_scatterings.crbegin();
                std::advance(it, lower_ind);

                for (uint_fast64_t ii = 0; ii < n_in_bin; it++, ii++)
                {
                    auto [co, sc] = *it;
                    for (auto e_co : sc)
                    {
                        auto e = e_co.dijet;

                        new_ET_y.emplace_back(e.y1, e.kt);
                        new_ET_y.emplace_back(e.y2, e.kt);

                        new_E_y.emplace_back(e.y1, e.kt * cosh(e.y1));
                        new_E_y.emplace_back(e.y2, e.kt * cosh(e.y2));
                    }
                    io::append_single_coll_binary(jet_file, sc, unirand, eng_shared);
                }
                jet_file.close();

                dETdy_by_cent.add(new_ET_y);
                dEdy_by_cent.add(new_E_y);

                std::stringstream outname{""};

                outname << "dEdy_" << static_cast<uint_fast16_t>(centLow * 100) << "_" << static_cast<uint_fast16_t>(centHigh * 100) << "_" << name_pfs;
                io::print_1d_histo(
                    dEdy_by_cent,
                    outname.str(),
                    1.0 / static_cast<double>(n_in_bin),
                    false);
                outname.seekp(0);
                outname << "dETdy_" << static_cast<uint_fast16_t>(centLow * 100) << "_" << static_cast<uint_fast16_t>(centHigh * 100) << "_" << name_pfs;
                io::print_1d_histo(
                    dETdy_by_cent,
                    outname.str(),
                    1.0 / static_cast<double>(n_in_bin),
                    false);
            }
        }
    }
    else
    {
        std::cout << "Simulation finished succesfully with " << collisions_for_reporting.size() << " collisions. Run >50 to produce output files" << std::endl;
    }
}
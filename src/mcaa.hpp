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

#ifndef MCAA_HPP
#define MCAA_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <gsl/gsl_sf_expint.h>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <random>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "generic_helpers.hpp"
#include "high_level_calcs.hpp"
#include "histo.hpp"
#include "io_helpers.hpp"
#include "nucleus_generator.hpp"
#include "pdf_builder.hpp"
#include "pqcd.hpp"
#include "Tpp_builder.hpp"
#include "typedefs.hpp"

class mcaa
{
public:
    // PUBLIC ATTRIBUTES //////////////////////////////////////////////////////////////////////////////////////////////
    // technical flags
    bool hotspots{false};                  // false=protons don't have hotspots
    uint_fast16_t n_hotspots{3};           // the number of proton hotspots
    bool calculate_end_state{true};        // false=no jets will be calculated, only collisions
    bool calculate_tata{true};             // if TA(x)TA(x) should be calculated for each dijet (takes time)
    bool end_state_filtering{true};        // false=no any filtering in the end state
    bool is_AA{true};                      // if the run is nucleus-nucleus
    bool is_pA{false};                     // if the run is proton-nucleus
    bool is_pp{false};                     // if the run is proton-proton
    bool nPDFs{true};                      // if there should be nuclear corrections in nucleus' PDFs
    bool MC_Glauber{true};                 // if true, nucleon-nucleon collisions are decided like with hard spheres
    bool val_cons{true};                   // if the valence quark number should be conserved
    bool mom_cons{true};                   // if the momentum should be conserved
    bool read_sigmajets_from_file{false};  // if the sigma_jets are precalculated
    bool proton_width_static{false};       // if the proton width should be a static value or depend on sqrt(s)
    bool saturation{true};                 // if EKRT saturation is enforced
    bool save_endstate_jets{true};         // if all the jets should be saved in binary (see jet_reader.cpp)
    bool sigma_inel_from_sigma_jet{false}; // is sigma_inel static or from eikonal model
    bool AA_inel_same_as_NN{false};        // is the triggering sigma the same as NN-sigma
    bool only_protons{false};              // if the nucleons are all protons
    bool snPDFs{true};                     // if the used nPDF should be spatially dependent or average
    bool snPDFs_linear{false};             // if the used spatial nuclear modification should be (1+cT) (true) or exp(cT) (false)
    bool snPDFs_new{true};                 // if the used spatial nuclear modification should be exp(cT) (false) or the new model (true)
    bool verbose{false};                   // lots of printing all around
    uint_fast16_t is_sat_y_dep{0};         // if the saturation criterion should be y-dependent
    bool pt_ordering{true};                // if the jets should be ordered by p_T
    bool sat_first{false};                 // if the saturation filter is to be done before all the other filters
    bool t03_ordering{false};              // if the jets should be ordered by t03
    bool hotspot_trigger{false};           // if the AA triggering is done with the minimal distance of two hotspots or two nucleon centers
    // simulation parameters
    std::string name{"example_name"};                       // name of the run, affects output filenames and such
    std::string sigmajet_filename{"sigma_jet.dat"};         // filename for the spatially dependent sigma_jet grid
    std::string centrality_filename{"centrality_bins.csv"}; // filename for the centrality bins for the reporting
    uint_fast32_t desired_N_events{2000};                   // how many events should be simulated
    uint_fast16_t A{208};                                   // mass number of the colliding nucleus
    uint_fast16_t B{208};                                   // mass number of the colliding nucleus
    uint_fast16_t ZA{82};                                   // atomic number of the colliding nucleus
    uint_fast16_t ZB{82};                                   // atomic number of the colliding nucleus
    double b_max{20.0};                                     // (fm) maximum of the impact parameter
    double b_min{0.0};                                      // (fm) minimum of the impact parameter
    double K_factor{2.0};                                   // pQCD K-factor to account for the higher order corrections
    double p0{1.0};                                         // (GeV) jet p_T lower cutoff
    double p02{1.0};                                        // (GeV^2) p_T lower cutoff squared
    double Kappa_factor{2.0};                               // saturation criterion for jets i, j: d_ij < (1/p_Ti + 1/p_Tj)/M_sat
    double nn_min_dist{0.4};                                // (fm) minimum distance between the nucleons in a nucleus
    double mand_s{5020.0 * 5020.0};                         // (GeV^2) mandelstam s for the hard process
    double hotspot_width{0.0};
    uint_fast64_t seed{0};                                  // random seed, 0 = from the clock (not reproducible)
    double proton_width{0.573};                    // (fm) proton width
    double proton_width_2{0.573 * 0.573};          // (fm^2) proton width squared
    double rad_max{30.0};                          // (fm) nucleons' maximum distance from the center of the nucleus
    double rad_min{0.0};                           // (fm) nucleons' minimum distance from the center of the nucleus
    double sigma_inel_NN{70.0};                    // (mb) inelastic cross section of the NN event
    double sigma_inel_trigger{0.0};                // (mb) inelastic cross section for the triggering
    double sqrt_s{5020.0};                         // (GeV) sqrt(s) for the hard process
    double T_AA_0{0.0};                            // (fm^-2) T_AA(0) for the normalization of c_A(x):s in snPDFs. 0 = calculate at the start.
    double envelope_marginal{1.05};                // How tight we want the envelope of dsigma to be, lower values == faster but more prone to error
    std::uniform_int_distribution<> hs_dist{0, 0}; // distribution for choosing a random hotspot

    // PUBLIC METHODS /////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief Construct a new mcaa object, some of the parameters
     * are taken from a parameter file. See "params_template" for
     * an example file.
     *
     * @param initfile input parameter file name
     */
    mcaa(const std::string &initfile);

    /**
     * @brief Find k_T^2 cutoff that fits sigma_jet to a given
     * target number using secant method.
     *
     * @param pt02 k_T^2 cutoff will be saved to this (GeV^2)
     * @param target the target value for sigma_jet (mb)
     * @param verbose if true, prints all the intermediate results
     * @return double the found k_T^2 cutoff (GeV^2)
     */
    auto fit_sigma_jet_pt0_cutoff(
        double &pt02,
        const double &target,
        const pqcd::sigma_jet_params &params,
        const bool &verbose = true) noexcept -> double;

    /**
     * @brief Find scale choice Q/k_T that fits sigma_jet to a given
     * target number, for a given target k_T^2 cutoff, using secant method.
     *
     * @param pt02 The target k_T^2 cutoff value (GeV^2)
     * @param target the target value for sigma_jet (mb)
     * @param verbose if true, prints all the intermediate results
     * @return double the found scale choice Q/k_T
     */
    auto find_sigma_jet_cutoff_Q(
        const double &pt02,
        const double &target,
        const bool &verbose = true) noexcept -> double;

    /**
     * @brief Runs the simulator with the given parameters. The reporting is
     * also built in this function, so that if one would like to change the
     * outputs for example, this is the function to modify. This is done to
     * keep in memory only the things that are actually needed.
     *
     */
    auto run() -> void;

    // PRIVATE ////////////////////////////////////////////////////////////////////////////////////////////////////////////
private:
    // PRIVATE ATTRIBUTES /////////////////////////////////////////////////////////////////////////////////////////////
    std::shared_ptr<pdf_builder> pdf{};           // pointer to the LHAPDF PDF-object of the proton PDF
    std::shared_ptr<Tpp_builder> Tpp{};           // nucleon overlap function
    double power_law{2.0};                        // parameter for the envelope function: sigma_jet < A*pT^(-power_law)
    pqcd::sigma_jet_params jet_params;            // the struct of sigma_jet parameters
    nucleus_generator::nucleus_params nuc_params; // the struct of parameters for the nucleus generation

    // PRIVATE STRUCTS ////////////////////////////////////////////////////////////////////////////////////////////////

    // PRIVATE METHODS ////////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * @brief Checks if the candidate circle overlaps with the previous circles. If
     * it does not, add it to the collection of accepted circles.
     *
     * @param cand_circle (x,y,r,n), where x and y are coordinates (fm), r is
     * radius (fm) and n is for supporting, say, two circles overlapping but no
     * more (not implemented).
     * @param final_circles A collection of the previously accepted circles.
     * @return true The circle was accepted and added to the collection.
     * @return false The circle was not accepted.
     */
    static auto check_and_place_circle_among_others(
        std::tuple<double, double, double, uint_fast16_t, double, double> cand_circle,
        std::vector<std::tuple<double, double, double, uint_fast16_t, double, double>> &final_circles,
        uint_fast16_t is_sat_y_dep) noexcept -> bool;

    auto sqrtalphas(double Q) noexcept -> double;

    /**
     * @brief Sample the coordinates for a given dijet from the Gaussian distribution
     * that is the product of the two Gaussian distributions that are associated with
     * the mother nucleons.
     *
     * @param dijet The dijet in question
     * @param normal_dist Normal distribution object
     * @param random_generator Random generator object
     * @return std::tuple<double, double, double> (x,y,z=0) coordinates of the dijet (fm)
     */
    auto throw_location_for_dijet(
        const dijet_with_ns &dijet,
        std::normal_distribution<double> &normal_dist,
        std::shared_ptr<std::mt19937> random_generator) noexcept -> std::tuple<double, double, double>;

    /**
     * @brief Takes the unphysical final state of the hard process and filters it with the
     * momentum conservation criterion and EKRT saturation criterion.
     *
     * @param binary_collisions The set of (unphysical) pQCD final states
     * @param filtered_scatterings The output, fully filtered dijets
     * @param random_generator Random generator object
     * @param pro Projectile nucleus' nucleon information
     * @param tar Target nucleus' nucleon information
     */
    auto filter_end_state(
        /*uint_fast64_t &nall,
        uint_fast64_t &nmc,
        uint_fast64_t &nsat,*/
        std::vector<nn_coll> &binary_collisions,
        std::vector<dijet_with_coords> &filtered_scatterings,
        std::shared_ptr<std::mt19937> random_generator,
        const std::vector<nucleon> &pro = {},
        const std::vector<nucleon> &tar = {},
        const bool sat_first = false) noexcept -> void;
};

#endif // MCAA_HPP

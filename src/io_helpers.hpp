// Copyright (c) 2025 Mikko Kuha (University of Jyväskylä).
// This program is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or (at your option) any later version.
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
// without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details. You should have received a copy
// of the GNU General Public License along with this program. If not, see
// <http://www.gnu.org/licenses/>.

#ifndef IO_HELPERS_HPP
#define IO_HELPERS_HPP

#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "histo.hpp"
#include "nucleon.hpp"
#include "typedefs.hpp"

/**
 * @brief struct for the end state filtering functions. In the constructor
 * calculates the formation time t0 = y/max(u,t) = (1/Q)*(E/Q) (in the collider frame)
 * for the jets.
 *
 */
struct dijet_with_ns
{
    dijet_specs dijet;
    nucleon *pro_nucleon;
    nucleon *tar_nucleon;
    double t01;
    double t02;
    double t0; // Larger of the formation times := the formation time of the process
    dijet_with_ns(dijet_specs dijet_, nucleon *pro_nucleon_, nucleon *tar_nucleon_, bool pt_ordering, bool t03_ordering)
        : dijet(std::move(dijet_)), pro_nucleon(pro_nucleon_), tar_nucleon(tar_nucleon_)
    {
        if (pt_ordering)
        {
            t0 = 1.0 / dijet.kt;
        }
        else
        {
            if (t03_ordering)
            {
                auto y1 = dijet.y1;
                auto y2 = dijet.y2;
                t0 = std::cosh((y1 + y2) / 2) / (dijet.kt * std::sqrt(1 + std::exp(-std::abs(y1 - y2))));
            }
            else
            {
                auto y1 = dijet.y1;
                auto y2 = dijet.y2;

                auto max_t_u = dijet.kt * (1 + std::exp(-std::abs(y1 - y2)));
                t01 = std::cosh(std::abs(y1)) / max_t_u;
                t02 = std::cosh(std::abs(y2)) / max_t_u;
                t0 = std::max(t01, t02);
            }
        }

        // t0 = (std::cosh(y1)+std::cosh(y2))/(2*dijet.kt*(1+std::cosh(y1-y2)));
    }
};

struct dijet_with_coords
{
    dijet_specs dijet;
    coords co;
    double tata;
    uint_fast16_t ia;
    uint_fast16_t ib;
    double xa;
    double ya;
    double za;
    double xb;
    double yb;
    double zb;
    bool a_is_neutron;
    bool b_is_neutron;
    dijet_with_coords(const dijet_specs &dijet_, const coords &co_, const double &tata_, const nucleon *const pro_nucleon_, const nucleon *const tar_nucleon_)
        : dijet(dijet_), co(co_), tata(tata_), ia(pro_nucleon_->index), ib(tar_nucleon_->index),
          xa(pro_nucleon_->co.x), ya(pro_nucleon_->co.y), za(pro_nucleon_->co.z), xb(tar_nucleon_->co.x), yb(tar_nucleon_->co.y), zb(tar_nucleon_->co.z),
          a_is_neutron(pro_nucleon_->is_neutron), b_is_neutron(tar_nucleon_->is_neutron) {}
};

class io
{
public:
    class Coll
    {

    private:
        double b, x, nTwo, sumET;
        ulong nColl, nPart, nEnd;
        bool alice = true;

    public:
        Coll()
        {
            b = 0.;
            x = 0.;
            nTwo = 0.;
            nColl = 0;
            nPart = 0;
            nEnd = 0;
            sumET = 0.;
        }
        Coll(ulong nColl_, ulong nPart_, ulong nEnd_, double b_, double sumET_)
        {
            b = b_;
            nColl = nColl_;
            nPart = nPart_;
            nEnd = nEnd_;
            sumET = sumET_;
            // ATLAS definitions.
            if (!alice)
            {
                x = 0.09;
                nTwo = 0.5 * (1. - x) * static_cast<double>(nPart) + x * static_cast<double>(nColl);
                // ALICE definitions.
            }
            else
            {
                x = 0.801;
                nTwo = x * static_cast<double>(nPart) + (1. - x) * static_cast<double>(nColl);
            }
        }
        Coll(ulong nCollIn, ulong nPartIn, ulong nEndIn, double bIn, std::shared_ptr<std::mt19937> random_generator)
        {
            b = bIn;
            nColl = nCollIn;
            nPart = nPartIn;
            nEnd = nEndIn;
            sumET = 0.;
            // ATLAS definitions.
            if (!alice)
            {
                x = 0.09;
                nTwo = 0.5 * (1. - x) * static_cast<double>(nPart) + x * static_cast<double>(nColl);
                sampleET(random_generator);
                // ALICE definitions.
            }
            else
            {
                x = 0.801;
                nTwo = x * static_cast<double>(nPart) + (1. - x) * static_cast<double>(nColl);
                sampleET(random_generator);
            }
        }
        double getB() const { return b; }
        ulong getNcoll() const { return nColl; }
        ulong getNpart() const { return nPart; }
        ulong getNend() const { return nEnd; }
        double getNtwo() const { return nTwo; }
        double getET() const { return sumET; }
        void sampleET(std::shared_ptr<std::mt19937> random_generator)
        {
            double mu = 46.4;
            double k = 1.5;
            double p = (mu / k) / (mu / k + 1.);
            uint_fast32_t nAncestors = static_cast<uint_fast32_t>(floor(nTwo));
            uint_fast32_t mult = 0;
            // std::cout << (*random_generator)() << ' ' << std::flush;
            std::negative_binomial_distribution<uint_fast32_t> distr(static_cast<uint_fast32_t>(k), 1. - p);
            for (uint_fast32_t i = 0; i < nAncestors; ++i)
            {
                mult += distr(*random_generator);
                // std::cout << mult << ' ' << std::flush;
            }
            sumET = static_cast<double>(mult);
            // std::cout << std::endl << nAncestors << ' ' << sumET/nAncestors << std::endl << std::endl;
            //  cout << endl << nAncestors << ' ' << sumET/nAncestors << endl << endl;
        }
    };
    static bool compNpart(Coll c1, Coll c2)
    {
        return c1.getNpart() < c2.getNpart();
    }
    static bool compNcoll(Coll c1, Coll c2)
    {
        return c1.getNcoll() < c2.getNcoll();
    }
    static bool compNend(Coll c1, Coll c2)
    {
        return c1.getNend() < c2.getNend();
    }
    static bool compNtwo(Coll c1, Coll c2)
    {
        return c1.getNtwo() < c2.getNtwo();
    }
    static bool compET(Coll c1, Coll c2)
    {
        return c1.getET() < c2.getET();
    }
    template <typename T>
    static double calc_ave(std::vector<Coll> &collisions, T (Coll::*func)() const)
    {
        std::vector<Coll>::iterator it;
        T tot = 0;
        for (it = collisions.begin(); it != collisions.end(); ++it)
        {
            tot += ((*it).*func)();
        }
        return static_cast<double>(tot) / static_cast<double>(collisions.size());
    }

    static auto mc_glauber_style_report(
        std::vector<Coll> &collisions,
        const double &sigma_inel,
        const uint_fast64_t &N_events,
        const uint_fast64_t &nBins,
        const double *const binsLow,
        const double *const binsHigh,
        std::ostream &out_stream) noexcept -> void
    {
        // Make sure that no rounding downwards.
        double eps = 0.1 / static_cast<double>(N_events);

        // int comp = 1;
        for (uint_fast8_t comp = 1; comp <= 5; ++comp)
        {

            // Print out the header.
            out_stream << std::endl;
            out_stream << "# sigma_inel = " << sigma_inel << " mb" << std::endl;

            switch (comp)
            {
            case 1:
                std::sort(collisions.begin(), collisions.end(), compNpart);
                out_stream << "Using Npart for centrality determination"
                           << std::endl
                           << std::endl
                           << "#  cent     nPmax     nPmin       <b>   <nPart>   <nColl>    <nEnd>    <T_AA>     <E_T>"
                           << std::endl;
                break;
            case 2:
                std::sort(collisions.begin(), collisions.end(), compNcoll);
                out_stream << "Using Ncoll for centrality determination"
                           << std::endl
                           << std::endl
                           << "#  cent     nCmax     nCmin       <b>   <nPart>   <nColl>    <nEnd>    <T_AA>     <E_T>"
                           << std::endl;
                break;
            case 3:
                std::sort(collisions.begin(), collisions.end(), compNtwo);
                out_stream << "Using two-component (ancestors) model for centrality determination"
                           << std::endl
                           << std::endl
                           << "#  cent     nAmax     nAmin       <b>   <nPart>   <nColl>    <nEnd>    <T_AA>     <E_T>"
                           << std::endl;
                break;
            case 4:
                std::sort(collisions.begin(), collisions.end(), compET);
                out_stream << "Using sumET model for centrality determination"
                           << std::endl
                           << std::endl
                           << "#  cent     ETmax     ETmin       <b>   <nPart>   <nColl>    <nEnd>    <T_AA>     <E_T>"
                           << std::endl;
                break;
            default: // case 5
                std::sort(collisions.begin(), collisions.end(), compET);
                out_stream << "Using the number of produced jets for centrality determination"
                           << std::endl
                           << std::endl
                           << "#  cent     nJmax     nJmin       <b>   <nPart>   <nColl>    <nEnd>    <T_AA>     <E_T>"
                           << std::endl;
            }
            std::reverse(collisions.begin(), collisions.end());

            std::vector<Coll> centrality;
            // Number of collisions within centrality class i.
            for (uint_fast8_t i = 0; i < nBins; i++)
            {
                uint_fast64_t lower = static_cast<uint_fast64_t>(binsLow[i] * static_cast<double>(N_events) + eps);
                uint_fast64_t upper = static_cast<uint_fast64_t>(binsHigh[i] * static_cast<double>(N_events) + eps);
                centrality.clear();
                while (lower < upper)
                {
                    centrality.push_back(collisions.at(lower++));
                }

                // Print the centrality selection.
                out_stream << std::setw(2) << binsLow[i] * 100 << " " << std::setw(4) << binsHigh[i] * 100;

                switch (comp)
                {
                case 1:
                    out_stream << std::setw(10) << centrality[0].getNpart()
                               << std::setw(10) << centrality[centrality.size() - 1].getNpart();
                    break;
                case 2:
                    out_stream << std::setw(10) << centrality[0].getNcoll()
                               << std::setw(10) << centrality[centrality.size() - 1].getNcoll();
                    break;
                case 3:
                    out_stream << std::setw(10) << centrality[0].getNtwo()
                               << std::setw(10) << centrality[centrality.size() - 1].getNtwo();
                    break;
                case 4:
                    out_stream << std::setw(10) << centrality[0].getET()
                               << std::setw(10) << centrality[centrality.size() - 1].getET();
                    break;
                default: // case 5
                    out_stream << std::setw(10) << centrality[0].getNend()
                               << std::setw(10) << centrality[centrality.size() - 1].getNend();
                    break;
                }

                out_stream << std::setw(10) << io::calc_ave<double>(centrality, &Coll::getB)
                           << std::setw(10) << io::calc_ave<ulong>(centrality, &Coll::getNpart)
                           << std::setw(10) << io::calc_ave<ulong>(centrality, &Coll::getNcoll)
                           << std::setw(10) << io::calc_ave<ulong>(centrality, &Coll::getNend)
                           << std::setw(10) << io::calc_ave<ulong>(centrality, &Coll::getNcoll) / sigma_inel
                           << std::setw(10) << calc_ave<double>(centrality, &Coll::getET)
                           << std::endl;
            }
        }
    }

    static auto print_3d_histo(
        const histo_3d &histo_,
        const std::string &filename,
        const double normalization = 1.0,
        const bool divide_tot = true) noexcept -> void
    {
        std::ofstream file;

        file.open(filename);

        auto [histo, uf, of, xs, tot] = histo_.project_2d(1);
        auto [xs0, xs1] = xs;

        file << "///Total count: " << tot << std::endl;
        file << "///underflow: " << uf << std::endl;
        file << "///overflow: " << of << std::endl;

        file << "///y bin walls:  ";
        for (auto y : xs1)
        {
            file << y << ' ';
        }
        file << std::endl;

        auto norm = (divide_tot) ? normalization : normalization * static_cast<double>(tot);

        ulong n_x_bins = xs0.size() - 1;
        ulong n_y_bins = xs1.size() - 1;

        for (ulong i = 0; i < n_x_bins; i++)
        {
            file << xs0[i] << ' ' << xs0[i + 1] << ' ';

            for (ulong j = 0; j < n_y_bins; j++)
            {
                file << histo[i][j] * norm << ' ';
            }
            file << std::endl;
        }

        auto [histo2, uf2, of2, xs2, tot2] = histo_.project_2d(2);
        auto [xs02, xs12] = xs2;
        file << "///y bin walls:  ";
        for (auto y : xs12)
        {
            file << y << ' ';
        }
        file << std::endl;

        norm = (divide_tot) ? normalization : normalization * static_cast<double>(tot2);

        n_x_bins = xs02.size() - 1;
        n_y_bins = xs12.size() - 1;

        for (ulong i = 0; i < n_x_bins; i++)
        {
            file << xs02[i] << ' ' << xs02[i + 1] << ' ';

            for (ulong j = 0; j < n_y_bins; j++)
            {
                file << histo2[i][j] * norm << ' ';
            }
            file << std::endl;
        }
        std::cout << "printed to " << filename << std::endl;
        file.close();
    }

    static auto print_2d_histo(
        const histo_2d &histo_,
        const std::string &filename,
        const double normalization = 1.0,
        const bool divide_tot = true) noexcept -> void
    {
        std::ofstream file;

        file.open(filename);

        auto [histo, uf, of, xs, tot] = histo_.get_histo();
        auto [uf0, uf1] = uf;
        auto [of0, of1] = of;
        auto [xs0, xs1] = xs;

        file << "///Total count: " << tot << std::endl;
        file << "///underflow: " << uf0 << ' ' << uf1 << std::endl;
        file << "///overflow: " << of0 << ' ' << of1 << std::endl;
        file << "///normalization: " << normalization << std::endl;

        file << "///y bin walls:  ";
        for (auto y : xs1)
        {
            file << y << ' ';
        }
        file << std::endl;

        auto norm = (divide_tot) ? normalization : normalization * static_cast<double>(tot);

        auto n_x_bins = histo.size();
        auto n_y_bins = histo[0].size();

        for (uint_fast8_t i = 0; i < n_x_bins; i++)
        {
            file << xs0[i] << ' ' << xs0[i + 1] << ' ';

            for (uint_fast8_t j = 0; j < n_y_bins; j++)
            {
                file << histo[i][j] * norm << ' ';
            }
            file << std::endl;
        }
        std::cout << "printed to " << filename << std::endl;
        file.close();
    }

    static auto save_event(
        std::ofstream &event_file,
        const std::vector<nucleon> &pro,
        const std::vector<nucleon> &tar,
        const double &impact_parameter) noexcept -> void
    {
        event_file << "{" << std::endl;
        event_file << "    " << impact_parameter << ',' << std::endl;
        event_file << "    {" << std::endl;
        for (const auto &n : pro)
        {
            event_file << "        {" << std::endl;
            event_file << "            " << n.co << ',' << std::endl;
            event_file << "            " << n.mom << ',' << std::endl;
            event_file << "            " << n.wounded << ',' << std::endl;
            event_file << "            " << n.is_neutron << std::endl;
            if (&n != &pro.back())
            {
                event_file << "        }," << std::endl;
            }
            else
            {
                event_file << "        }" << std::endl;
            }
        }
        event_file << "    }," << std::endl;
        event_file << "    {" << std::endl;
        for (auto n : tar)
        {
            event_file << "        {" << std::endl;
            event_file << "            " << n.co << std::endl;
            event_file << "            " << n.mom << std::endl;
            event_file << "            " << n.wounded << std::endl;
            event_file << "            " << n.is_neutron << std::endl;
            if (&n != &tar.back())
            {
                event_file << "        }," << std::endl;
            }
            else
            {
                event_file << "        }" << std::endl;
            }
        }
        event_file << "    }" << std::endl;
        event_file << "}," << std::endl;
    }

    static auto save_event(
        std::ofstream &event_file,
        const std::vector<nucleon> &pro,
        const std::vector<nucleon> &tar,
        const double &impact_parameter,
        const std::vector<dijet_with_coords> &filtered_scatterings) noexcept -> void
    {
        event_file << "{" << std::endl;
        event_file << "    " << impact_parameter << ',' << std::endl;
        event_file << "    {" << std::endl;
        for (const auto &n : pro)
        {
            event_file << "        {" << std::endl;
            event_file << "            " << n.co << ',' << std::endl;
            event_file << "            " << n.mom << ',' << std::endl;
            event_file << "            " << n.wounded << ',' << std::endl;
            event_file << "            " << n.is_neutron << std::endl;
            if (&n != &pro.back())
            {
                event_file << "        }," << std::endl;
            }
            else
            {
                event_file << "        }" << std::endl;
            }
        }
        event_file << "    }," << std::endl;
        event_file << "    {" << std::endl;
        for (const auto &n : tar)
        {
            event_file << "        {" << std::endl;
            event_file << "            " << n.co << ',' << std::endl;
            event_file << "            " << n.mom << ',' << std::endl;
            event_file << "            " << n.wounded << ',' << std::endl;
            event_file << "            " << n.is_neutron << std::endl;
            if (&n != &tar.back())
            {
                event_file << "        }," << std::endl;
            }
            else
            {
                event_file << "        }" << std::endl;
            }
        }
        event_file << "    }," << std::endl;
        event_file << "    {" << std::endl;
        for (const auto &scc : filtered_scatterings)
        {
            const auto sc = scc.dijet;

            event_file << "        {" << std::endl;
            event_file << "            " << sc.kt << ',' << std::endl;
            event_file << "            " << sc.y1 << ',' << std::endl;
            event_file << "            " << sc.y2 << ',' << std::endl;
            event_file << "            " << sc.init1 << ',' << std::endl;
            event_file << "            " << sc.init2 << ',' << std::endl;
            event_file << "            " << sc.final1 << ',' << std::endl;
            event_file << "            " << sc.final2 << std::endl;
            if (&scc != &filtered_scatterings.back())
            {
                event_file << "        }," << std::endl;
            }
            else
            {
                event_file << "        }" << std::endl;
            }
        }
        event_file << "    }" << std::endl;
        event_file << "}," << std::endl;
    }

    /*
        static auto save_single_coll_csv
        (
            const std::array<std::vector<dijet_with_coords>, 4> &filtered_scatterings,
            std::uniform_real_distribution<double> &unirand,
            std::shared_ptr<std::mt19937> random_generator,
            bool include_TATA = false
        ) noexcept -> void
        {
            std::ofstream event_file;
            std::string name = "jets";
            std::string separator = ",";
            std::array<std::string, 4> names{name+".csv",
                                             name+"_MC.csv",
                                             name+"_SAT.csv",
                                             name+"_SAT_MC.csv"};

            double pt, energy, px, py, pz;
            double y1, y2;
            double theta;

            for (size_t i=0; i<4; i++)
            {
                event_file.open(names[i]);
                //Output header
                event_file << "t0" << separator;
                event_file << "x"   << separator;
                event_file << "y"   << separator;
                event_file << "z"   << separator;
                event_file << "E"   << separator;
                event_file << "p_x" << separator;
                event_file << "p_y" << separator;
                event_file << "p_z";
                if(include_TATA)
                {
                    event_file << separator;
                    event_file << "p_T" << separator;
                    event_file << "T_AT_A" << std::endl;
                }
                else
                {
                    event_file << std::endl;
                }
                event_file << std::setprecision(std::numeric_limits<double>::digits10 + 1);

                for (auto e_co : filtered_scatterings[i])
                {
                    pt = e_co.dijet.kt;
                    y1 = e_co.dijet.y1;
                    y2 = e_co.dijet.y2;
                    // jet 1
                    energy = pt*std::cosh(y1);
                    theta = 2*M_PI*unirand(*random_generator);
                    px = pt*std::cos(theta);
                    py = pt*std::sin(theta);
                    pz = pt*std::sinh(y1);
                    event_file << e_co.t0  << separator; //t0
                    event_file << e_co.co.x << separator; //x
                    event_file << e_co.co.y << separator; //y
                    event_file << e_co.co.z << separator; //z
                    event_file << energy    << separator; //E
                    event_file << px        << separator; //p_x
                    event_file << py        << separator; //p_y
                    event_file << pz                    ; //p_z
                    if(include_TATA)
                    {
                        event_file << separator;
                        event_file << pt << separator;        //p_T
                        event_file << e_co.tata << std::endl; //T_A * T_A
                    }
                    else
                    {
                        event_file << std::endl;
                    }
                    auto m = energy*energy-(px*px+py*py+pz*pz);
                    if (m >= 0.0001)
                    {
                        std::cout<<"1: "<<m<<std::endl;
                    }


                    // jet 2
                    energy = pt*std::cosh(y2);
                    pz = pt*std::sinh(y2);
                    event_file << e_co.t0  << separator; //t0
                    event_file << e_co.co.x << separator; //x
                    event_file << e_co.co.y << separator; //y
                    event_file << e_co.co.z << separator; //z
                    event_file << energy    << separator; //E
                    event_file << -px       << separator; //p_x
                    event_file << -py       << separator; //p_y
                    event_file << pz                    ; //p_z
                    if(include_TATA)
                    {
                        event_file << separator;
                        event_file << pt << separator;        //p_T
                        event_file << e_co.tata << std::endl; //T_A * T_A
                    }
                    else
                    {
                        event_file << std::endl;
                    }
                    m = energy*energy-(px*px+py*py+pz*pz);
                    if (m >= 0.0001)
                    {
                        std::cout<<"2: "<<m<<std::endl;
                    }
                }
                event_file.close();
            }
        }


        static auto save_single_coll_binary
        (
            const std::array<std::vector<dijet_with_coords>, 4> &filtered_scatterings,
            std::uniform_real_distribution<double> &unirand,
            std::shared_ptr<std::mt19937> random_generator
        ) noexcept -> void
        {
            std::ofstream jet_file;
            std::string name = "jets";
            std::array<std::string, 4> names{name+".dat",
                                             name+"_MC.dat",
                                             name+"_SAT.dat",
                                             name+"_SAT_MC.dat"};

            double pt, y1, y2, t0, x, y, z, energy, px, py, pz, tata, theta;
            uint_fast64_t n_jets;

            for (uint_fast8_t i=0; i<4; i++)
            {
                n_jets = filtered_scatterings[i].size()*2;

                jet_file.open(names[i], std::ios::out | std::ios::binary);
                jet_file.write(reinterpret_cast<char*>(&n_jets), sizeof n_jets); //total number of jets

                for (auto e_co : filtered_scatterings[i])
                {
                    pt = e_co.dijet.kt;
                    y1 = e_co.dijet.y1;
                    y2 = e_co.dijet.y2;
                    t0 = e_co.t0;
                    x = e_co.co.x;
                    y = e_co.co.y;
                    z = e_co.co.z;
                    tata = e_co.tata;

                    // jet 1
                    energy = pt*std::cosh(y1);
                    theta = 2*M_PI*unirand(*random_generator);
                    px = pt*std::cos(theta);
                    py = pt*std::sin(theta);
                    pz = pt*std::sinh(y1);
                    jet_file.write(reinterpret_cast<char*>(&t0)     , sizeof t0);     //t0
                    jet_file.write(reinterpret_cast<char*>(&x)      , sizeof x);      //x
                    jet_file.write(reinterpret_cast<char*>(&y)      , sizeof y);      //y
                    jet_file.write(reinterpret_cast<char*>(&z)      , sizeof z);      //z
                    jet_file.write(reinterpret_cast<char*>(&energy) , sizeof energy); //E
                    jet_file.write(reinterpret_cast<char*>(&px)     , sizeof px);     //p_x
                    jet_file.write(reinterpret_cast<char*>(&py)     , sizeof py);     //p_y
                    jet_file.write(reinterpret_cast<char*>(&pz)     , sizeof pz);     //p_z
                    jet_file.write(reinterpret_cast<char*>(&tata)   , sizeof tata);   //T_A * T_A

                    // jet 2
                    energy = pt*std::cosh(y2);
                    px = -px;
                    py = -py;
                    pz = pt*std::sinh(y2);
                    jet_file.write(reinterpret_cast<char*>(&t0)     , sizeof t0);     //t0
                    jet_file.write(reinterpret_cast<char*>(&x)      , sizeof x);      //x
                    jet_file.write(reinterpret_cast<char*>(&y)      , sizeof y);      //y
                    jet_file.write(reinterpret_cast<char*>(&z)      , sizeof z);      //z
                    jet_file.write(reinterpret_cast<char*>(&energy) , sizeof energy); //E
                    jet_file.write(reinterpret_cast<char*>(&px)     , sizeof px);     //p_x
                    jet_file.write(reinterpret_cast<char*>(&py)     , sizeof py);     //p_y
                    jet_file.write(reinterpret_cast<char*>(&pz)     , sizeof pz);     //p_z
                    jet_file.write(reinterpret_cast<char*>(&tata)   , sizeof tata);   //T_A * T_A
                }
                jet_file.close();
            }
        }
    */
    static auto parse_output_params() noexcept
        -> std::tuple<bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool, bool>
    {
        bool w_t01 = true;
        bool w_t02 = true;
        bool w_x = true;
        bool w_y = true;
        bool w_pt = true;
        bool w_y1 = true;
        bool w_y2 = true;
        bool w_phi = true;
        bool w_tata = true;
        bool w_init1 = true;
        bool w_init2 = true;
        bool w_final1 = true;
        bool w_final2 = true;
        bool w_ia = true;
        bool w_ib = true;
        bool w_xa = true;
        bool w_ya = true;
        bool w_za = true;
        bool w_xb = true;
        bool w_yb = true;
        bool w_zb = true;
        bool w_a_is_neutron = true;
        bool w_b_is_neutron = true;

        std::ifstream output_params("output_params");
        if (!output_params.is_open())
        {
            std::cout << "Could not open output_params, will output default parameters " << std::endl;
        }
        else
        {
            w_t01 = false;
            w_t02 = false;
            w_x = false;
            w_y = false;
            w_pt = false;
            w_y1 = false;
            w_y2 = false;
            w_phi = false;
            w_tata = false;
            w_init1 = false;
            w_init2 = false;
            w_final1 = false;
            w_final2 = false;
            w_ia = false;
            w_ib = false;
            w_xa = false;
            w_ya = false;
            w_za = false;
            w_xb = false;
            w_yb = false;
            w_zb = false;
            w_a_is_neutron = false;
            w_b_is_neutron = false;
            std::string line;
            for (; std::getline(output_params, line);)
            {
                std::istringstream line_stream(line);
                std::string param_name;
                line_stream >> param_name;
                if (param_name == "t01")
                {
                    w_t01 = true;
                }
                else if (param_name == "t02")
                {
                    w_t02 = true;
                }
                else if (param_name == "x")
                {
                    w_x = true;
                }
                else if (param_name == "y")
                {
                    w_y = true;
                }
                else if (param_name == "pt")
                {
                    w_pt = true;
                }
                else if (param_name == "y1")
                {
                    w_y1 = true;
                }
                else if (param_name == "y2")
                {
                    w_y2 = true;
                }
                else if (param_name == "phi")
                {
                    w_phi = true;
                }
                else if (param_name == "tata")
                {
                    w_tata = true;
                }
                else if (param_name == "init1")
                {
                    w_init1 = true;
                }
                else if (param_name == "init2")
                {
                    w_init2 = true;
                }
                else if (param_name == "final1")
                {
                    w_final1 = true;
                }
                else if (param_name == "final2")
                {
                    w_final2 = true;
                }
                else if (param_name == "ia")
                {
                    w_ia = true;
                }
                else if (param_name == "ib")
                {
                    w_ib = true;
                }
                else if (param_name == "xa")
                {
                    w_xa = true;
                }
                else if (param_name == "ya")
                {
                    w_ya = true;
                }
                else if (param_name == "za")
                {
                    w_za = true;
                }
                else if (param_name == "xb")
                {
                    w_xb = true;
                }
                else if (param_name == "yb")
                {
                    w_yb = true;
                }
                else if (param_name == "zb")
                {
                    w_zb = true;
                }
                else if (param_name == "a_is_neutron")
                {
                    w_a_is_neutron = true;
                }
                else if (param_name == "b_is_neutron")
                {
                    w_b_is_neutron = true;
                }
                else
                {
                    std::cout << "Unknown parameter: " << param_name << std::endl;
                }
            }
            output_params.close();
        }
        return std::make_tuple(
            w_t01,
            w_t02,
            w_x,
            w_y,
            w_pt,
            w_y1,
            w_y2,
            w_phi,
            w_tata,
            w_init1,
            w_init2,
            w_final1,
            w_final2,
            w_ia,
            w_ib,
            w_xa,
            w_ya,
            w_za,
            w_xb,
            w_yb,
            w_zb,
            w_a_is_neutron,
            w_b_is_neutron);
    }

    static auto append_single_coll_binary(
        std::ofstream &jet_file,
        const std::vector<dijet_with_coords> &filtered_scatterings,
        std::uniform_real_distribution<double> &unirand,
        std::shared_ptr<std::mt19937> random_generator) noexcept -> void
    {
        auto [w_t01,
              w_t02,
              w_x,
              w_y,
              w_pt,
              w_y1,
              w_y2,
              w_phi,
              w_tata,
              w_init1,
              w_init2,
              w_final1,
              w_final2,
              w_ia,
              w_ib,
              w_xa,
              w_ya,
              w_za,
              w_xb,
              w_yb,
              w_zb,
              w_a_is_neutron,
              w_b_is_neutron] = parse_output_params();

        // double pt, y1, y2, t0, x, y, z, energy, px, py, pz, tata, phi;
        double pt, y1, y2, t01, t02, x, y, tata, phi;
        particle_id init1, init2, final1, final2;
        uint_fast16_t ia, ib;
        uint_fast64_t n_dijets;
        double xa, ya, za, xb, yb, zb;
        bool a_is_neutron, b_is_neutron;
        n_dijets = filtered_scatterings.size();
        jet_file.write(reinterpret_cast<char *>(&n_dijets), sizeof n_dijets); // total number of dijets

        for (auto e_co : filtered_scatterings)
        {
            pt = e_co.dijet.kt;
            y1 = e_co.dijet.y1;
            y2 = e_co.dijet.y2;
            t01 = 1 / pt;
            t02 = 1 / pt;
            x = e_co.co.x;
            y = e_co.co.y;
            // z = e_co.co.z;
            tata = e_co.tata;

            init1 = e_co.dijet.init1;
            init2 = e_co.dijet.init2;
            final1 = e_co.dijet.final1;
            final2 = e_co.dijet.final2;
            ia = e_co.ia;
            ib = e_co.ib;

            xa = e_co.xa;
            ya = e_co.ya;
            za = e_co.za;
            xb = e_co.xb;
            yb = e_co.yb;
            zb = e_co.zb;
            a_is_neutron = e_co.a_is_neutron;
            b_is_neutron = e_co.b_is_neutron;

            // jet 1
            // energy = pt*std::cosh(y1);
            phi = 2 * M_PI * unirand(*random_generator);
            // px = pt*std::cos(theta);
            // py = pt*std::sin(theta);
            // pz = pt*std::sinh(y1);
            if (w_t01)
            {
                jet_file.write(reinterpret_cast<char *>(&t01), sizeof t01);
            } // t01
            if (w_t02)
            {
                jet_file.write(reinterpret_cast<char *>(&t02), sizeof t02);
            } // t02
            if (w_x)
            {
                jet_file.write(reinterpret_cast<char *>(&x), sizeof x);
            } // x
            if (w_y)
            {
                jet_file.write(reinterpret_cast<char *>(&y), sizeof y);
            } // y
            if (w_pt)
            {
                jet_file.write(reinterpret_cast<char *>(&pt), sizeof pt);
            } // p_T
            if (w_y1)
            {
                jet_file.write(reinterpret_cast<char *>(&y1), sizeof y1);
            } // y1
            if (w_y2)
            {
                jet_file.write(reinterpret_cast<char *>(&y2), sizeof y2);
            } // y2
            if (w_phi)
            {
                jet_file.write(reinterpret_cast<char *>(&phi), sizeof phi);
            } // phi
            if (w_tata)
            {
                jet_file.write(reinterpret_cast<char *>(&tata), sizeof tata);
            } // T_A * T_A

            if (w_init1)
            {
                jet_file.write(reinterpret_cast<char *>(&init1), sizeof init1);
            } // flavour of incoming 1
            if (w_init2)
            {
                jet_file.write(reinterpret_cast<char *>(&init2), sizeof init2);
            } // flavour of incoming 2
            if (w_final1)
            {
                jet_file.write(reinterpret_cast<char *>(&final1), sizeof final1);
            } // flavour of outgoing 1
            if (w_final2)
            {
                jet_file.write(reinterpret_cast<char *>(&final2), sizeof final2);
            } // flavour of outgoing 2

            if (w_ia)
            {
                jet_file.write(reinterpret_cast<char *>(&ia), sizeof ia);
            } // index of mother a
            if (w_ib)
            {
                jet_file.write(reinterpret_cast<char *>(&ib), sizeof ib);
            } // index of mother b
            if (w_xa)
            {
                jet_file.write(reinterpret_cast<char *>(&xa), sizeof xa);
            }
            if (w_ya)
            {
                jet_file.write(reinterpret_cast<char *>(&ya), sizeof ya);
            }
            if (w_za)
            {
                jet_file.write(reinterpret_cast<char *>(&za), sizeof za);
            }
            if (w_xb)
            {
                jet_file.write(reinterpret_cast<char *>(&xb), sizeof xb);
            }
            if (w_yb)
            {
                jet_file.write(reinterpret_cast<char *>(&yb), sizeof yb);
            }
            if (w_zb)
            {
                jet_file.write(reinterpret_cast<char *>(&zb), sizeof zb);
            }
            if (w_a_is_neutron)
            {
                jet_file.write(reinterpret_cast<char *>(&a_is_neutron), sizeof a_is_neutron);
            }
            if (w_b_is_neutron)
            {
                jet_file.write(reinterpret_cast<char *>(&b_is_neutron), sizeof b_is_neutron);
            }

            // jet 2
            // energy = pt*std::cosh(y2);
            // px = -px;
            // py = -py;
            // pz = pt*std::sinh(y2);
            // jet_file.write(reinterpret_cast<char*>(&t0)     , sizeof t0);     //t0
            // jet_file.write(reinterpret_cast<char*>(&x)      , sizeof x);      //x
            // jet_file.write(reinterpret_cast<char*>(&y)      , sizeof y);      //y
            // jet_file.write(reinterpret_cast<char*>(&z)      , sizeof z);      //z
            // jet_file.write(reinterpret_cast<char*>(&energy) , sizeof energy); //E
            // jet_file.write(reinterpret_cast<char*>(&px)     , sizeof px);     //p_x
            // jet_file.write(reinterpret_cast<char*>(&py)     , sizeof py);     //p_y
            // jet_file.write(reinterpret_cast<char*>(&pz)     , sizeof pz);     //p_z
            // jet_file.write(reinterpret_cast<char*>(&tata)   , sizeof tata);   //T_A * T_A
        }
    }

    static auto print_1d_histo(
        const histo_1d &histo_,
        const std::string &filename,
        const double normalization = 1.0,
        const bool divide_tot = true) noexcept -> void
    {
        std::ofstream file;

        file.open(filename);

        auto [histo, uf, of, xs, tot] = histo_.get_histo();

        file << "///Total count: " << tot << std::endl;
        file << "///underflow: " << uf << std::endl;
        file << "///overflow: " << of << std::endl;

        file << "///x bin walls:  ";
        for (auto x : xs)
        {
            file << x << ' ';
        }
        file << std::endl;

        auto norm = (divide_tot) ? normalization : normalization * static_cast<double>(tot);

        for (auto e : histo)
        {
            file << e * norm << ' ';
        }
        file << std::endl;

        std::cout << "printed to " << filename << std::endl;
        file.close();
    }

    enum Param
    {
        name,
        sigmajet_filename,
        read_sigmajets_from_file,
        centrality_filename,
        n_events,
        b_max,
        b_min,
        sqrt_s,
        K_factor,
        Kappa_factor,
        p0,
        proton_width_static,
        proton_width,
        sigma_inel_trigger,
        envelope_marginal,
        A,
        B,
        ZA,
        ZB,
        nuclear_RA,
        nuclear_RB,
        nuclear_dA,
        nuclear_dB,
        nuclear_beta2A,
        nuclear_beta2B,
        nuclear_beta3A,
        nuclear_beta3B,
        nuclear_beta4A,
        nuclear_beta4B,
        rad_max,
        correct_overlap_bias,
        nn_min_dist,
        shift_cms,
        hotspots,
        n_hotspots,
        hotspot_width,
        is_mc_glauber,
        sigma_inel_NN,
        only_protons,
        use_npdfs,
        use_snpdfs,
        save_endstate_jets,
        end_state_filtering,
        is_saturation,
        is_mom_cons,
        is_val_cons,
        seed
    };

    static auto read_conf(
        const std::string filename) noexcept
    {
        std::ifstream input(filename);
        const std::map<std::string, io::Param> stringToParam{
            {"name", name},
            {"sigmajet_filename", sigmajet_filename},
            {"read_sigmajets_from_file", read_sigmajets_from_file},
            {"centrality_filename", centrality_filename},
            {"n_events", n_events},
            {"b_max", b_max},
            {"b_min", b_min},
            {"sqrt_s", sqrt_s},
            {"K_factor", K_factor},
            {"Kappa_factor", Kappa_factor},
            {"p0", p0},
            {"proton_width_static", proton_width_static},
            {"proton_width", proton_width},
            {"sigma_inel_trigger", sigma_inel_trigger},
            {"envelope_marginal", envelope_marginal},
            {"A", A},
            {"B", B},
            {"ZA", ZA},
            {"ZB", ZB},
            {"nuclear_RA", nuclear_RA},
            {"nuclear_RB", nuclear_RB},
            {"nuclear_dA", nuclear_dA},
            {"nuclear_dB", nuclear_dB},
            {"nuclear_beta2A", nuclear_beta2A},
            {"nuclear_beta2B", nuclear_beta2B},
            {"nuclear_beta3A", nuclear_beta3A},
            {"nuclear_beta3B", nuclear_beta3B},
            {"nuclear_beta4A", nuclear_beta4A},
            {"nuclear_beta4B", nuclear_beta4B},
            {"rad_max", rad_max},
            {"correct_overlap_bias", correct_overlap_bias},
            {"nn_min_dist", nn_min_dist},
            {"shift_cms", shift_cms},
            {"hotspots", hotspots},
            {"n_hotspots", n_hotspots},
            {"hotspot_width", hotspot_width},
            {"is_mc_glauber", is_mc_glauber},
            {"sigma_inel_NN", sigma_inel_NN},
            {"only_protons", only_protons},
            {"use_npdfs", use_npdfs},
            {"use_snpdfs", use_snpdfs},
            {"save_endstate_jets", save_endstate_jets},
            {"end_state_filtering", end_state_filtering},
            {"is_saturation", is_saturation},
            {"is_mom_cons", is_mom_cons},
            {"is_val_cons", is_val_cons},
            {"seed", seed}};
        std::string name{"example_name"};
        std::string sigmajet_filename{"example_sigma_jet.dat"};
        bool read_sigmajets_from_file{false};
        std::string centrality_filename{"centrality_bins.csv"};
        uint_fast32_t n_events{2000};
        double b_max{20.0};
        double b_min{0.0};
        double sqrt_s{5020.0};
        double K_factor{2.0};
        double Kappa_factor{2.0};
        double p0{1.0};
        bool proton_width_static{false};
        double proton_width{0.573};
        double sigma_inel_trigger{0.0};
        double envelope_marginal{1.05};
        uint_fast16_t A{208};
        uint_fast16_t B{208};
        uint_fast16_t ZA{82};
        uint_fast16_t ZB{82};
        double nuclear_RA(6.62435);
        double nuclear_RB(6.62435);
        double nuclear_dA(0.5498);
        double nuclear_dB(0.5498);
        double nuclear_beta2A(0.0);
        double nuclear_beta2B(0.0);
        double nuclear_beta3A(0.0);
        double nuclear_beta3B(0.0);
        double nuclear_beta4A(0.0);
        double nuclear_beta4B(0.0);
        double rad_max{30.0};
        bool correct_overlap_bias{true};
        double nn_min_dist{0.4};
        bool shift_cms{true};
        bool hotspots{false};
        uint_fast16_t n_hotspots{3};
        double hotspot_width{0.0};
        bool is_mc_glauber{false};
        double sigma_inel_NN{70.0};
        bool only_protons{false};
        bool use_npdfs{true};
        bool use_snpdfs{true};
        bool save_endstate_jets{true};
        bool end_state_filtering{true};
        bool is_saturation{true};
        bool is_mom_cons{true};
        bool is_val_cons{true};
        uint_fast64_t seed{0};

        uint_fast16_t count = 0;

        if (input.is_open())
        {
            std::string line;

            for (; std::getline(input, line);)
            {
                std::istringstream line_stream(line);
                std::string param_name;
                line_stream >> param_name;
                if (param_name[0] == '#')
                {
                    continue;
                }

                io::Param p{io::Param::name};
                try
                {
                    p = stringToParam.at(param_name);
                }
                catch (const std::exception &e)
                {
                    std::cout << "READ UNKNOWN PARAMETER: " << param_name << std::endl;
                    exit(1);
                }

                switch (p)
                {
                case io::Param::name:
                    line_stream >> name;
                    break;
                case io::Param::sigmajet_filename:
                    line_stream >> sigmajet_filename;
                    break;
                case io::Param::read_sigmajets_from_file:
                    line_stream >> std::boolalpha >> read_sigmajets_from_file;
                    break;
                case io::Param::centrality_filename:
                    line_stream >> centrality_filename;
                    break;
                case io::Param::n_events:
                    line_stream >> n_events;
                    break;
                case io::Param::b_max:
                    line_stream >> b_max;
                    break;
                case io::Param::b_min:
                    line_stream >> b_min;
                    break;
                case io::Param::sqrt_s:
                    line_stream >> sqrt_s;
                    break;
                case io::Param::K_factor:
                    line_stream >> K_factor;
                    break;
                case io::Param::Kappa_factor:
                    line_stream >> Kappa_factor;
                    break;
                case io::Param::p0:
                    line_stream >> p0;
                    break;
                case io::Param::proton_width_static:
                    line_stream >> std::boolalpha >> proton_width_static;
                    break;
                case io::Param::proton_width:
                    line_stream >> proton_width;
                    break;
                case io::Param::sigma_inel_trigger:
                    line_stream >> sigma_inel_trigger;
                    break;
                case io::Param::envelope_marginal:
                    line_stream >> envelope_marginal;
                    break;
                case io::Param::A:
                    line_stream >> A;
                    break;
                case io::Param::B:
                    line_stream >> B;
                    break;
                case io::Param::ZA:
                    line_stream >> ZA;
                    break;
                case io::Param::ZB:
                    line_stream >> ZB;
                    break;
                case io::Param::nuclear_RA:
                    line_stream >> nuclear_RA;
                    break;
                case io::Param::nuclear_RB:
                    line_stream >> nuclear_RB;
                    break;
                case io::Param::nuclear_dA:
                    line_stream >> nuclear_dA;
                    break;
                case io::Param::nuclear_dB:
                    line_stream >> nuclear_dB;
                    break;
                case io::Param::nuclear_beta2A:
                    line_stream >> nuclear_beta2A;
                    break;
                case io::Param::nuclear_beta2B:
                    line_stream >> nuclear_beta2B;
                    break;
                case io::Param::nuclear_beta3A:
                    line_stream >> nuclear_beta3A;
                    break;
                case io::Param::nuclear_beta3B:
                    line_stream >> nuclear_beta3B;
                    break;
                case io::Param::nuclear_beta4A:
                    line_stream >> nuclear_beta4A;
                    break;
                case io::Param::nuclear_beta4B:
                    line_stream >> nuclear_beta4B;
                    break;
                case io::Param::rad_max:
                    line_stream >> rad_max;
                    break;
                case io::Param::correct_overlap_bias:
                    line_stream >> std::boolalpha >> correct_overlap_bias;
                    break;
                case io::Param::nn_min_dist:
                    line_stream >> nn_min_dist;
                    break;
                case io::Param::shift_cms:
                    line_stream >> std::boolalpha >> shift_cms;
                    break;
                case io::Param::hotspots:
                    line_stream >> std::boolalpha >> hotspots;
                    break;
                case io::Param::n_hotspots:
                    line_stream >> n_hotspots;
                    break;
                case io::Param::hotspot_width:
                    line_stream >> hotspot_width;
                    break;
                case io::Param::is_mc_glauber:
                    line_stream >> std::boolalpha >> is_mc_glauber;
                    break;
                case io::Param::sigma_inel_NN:
                    line_stream >> sigma_inel_NN;
                    break;
                case io::Param::only_protons:
                    line_stream >> std::boolalpha >> only_protons;
                    break;
                case io::Param::use_npdfs:
                    line_stream >> std::boolalpha >> use_npdfs;
                    break;
                case io::Param::use_snpdfs:
                    line_stream >> std::boolalpha >> use_snpdfs;
                    break;
                case io::Param::save_endstate_jets:
                    line_stream >> std::boolalpha >> save_endstate_jets;
                    break;
                case io::Param::end_state_filtering:
                    line_stream >> std::boolalpha >> end_state_filtering;
                    break;
                case io::Param::is_saturation:
                    line_stream >> std::boolalpha >> is_saturation;
                    break;
                case io::Param::is_mom_cons:
                    line_stream >> std::boolalpha >> is_mom_cons;
                    break;
                case io::Param::is_val_cons:
                    line_stream >> std::boolalpha >> is_val_cons;
                    break;
                case io::Param::seed:
                    line_stream >> seed;
                    break;
                default:
                    continue;
                }
                count++;
            }
        }
        else
        {
            std::cout << "Error opening the configuration file!!" << std::endl;
            std::terminate();
        }

        std::cout << "Read " << count << " parameters from a file " << filename << std::endl;

        return std::make_tuple(
            name,
            sigmajet_filename,
            read_sigmajets_from_file,
            centrality_filename,
            n_events,
            b_max,
            b_min,
            sqrt_s,
            K_factor,
            Kappa_factor,
            p0,
            proton_width_static,
            proton_width,
            sigma_inel_trigger,
            envelope_marginal,
            A,
            B,
            ZA,
            ZB,
            nuclear_RA,
            nuclear_RB,
            nuclear_dA,
            nuclear_dB,
            nuclear_beta2A,
            nuclear_beta2B,
            nuclear_beta3A,
            nuclear_beta3B,
            nuclear_beta4A,
            nuclear_beta4B,
            rad_max,
            correct_overlap_bias,
            nn_min_dist,
            shift_cms,
            hotspots,
            n_hotspots,
            hotspot_width,
            is_mc_glauber,
            sigma_inel_NN,
            only_protons,
            use_npdfs,
            use_snpdfs,
            save_endstate_jets,
            end_state_filtering,
            is_saturation,
            is_mom_cons,
            is_val_cons,
            seed);
    }

    static auto print_histos(
        const std::string &name_postfix,
        const histo_2d &jets,
        const histo_2d &dijets,
        const histo_1d &dETdy,
        const histo_1d &dEdy,
        const histo_1d &dNdy,
        const histo_1d &dNdET,
        const histo_1d &dETdeta,
        const histo_1d &dEdeta,
        const histo_1d &dETdb,
        const histo_1d &dEdb,
        const double &dijet_norm,
        const uint_fast32_t &AA_events) noexcept -> void
    {
        std::string true_postfix{name_postfix + ".dat"};

        // sigma1jet
        print_2d_histo(
            jets,
            "sigma1jet_sim_" + true_postfix,
            2.0 * dijet_norm);

        // dNdpTdy
        print_2d_histo(
            jets,
            "dNdpTdy_sim_" + true_postfix,
            1.0,
            false);

        // sigmadijet
        print_2d_histo(
            dijets,
            "sigmadijet_sim_" + true_postfix,
            dijet_norm);

        // dETdy
        print_1d_histo(
            dETdy,
            "dETdy_sim_" + true_postfix,
            1.0 / static_cast<double>(AA_events),
            false);

        // dEdy
        print_1d_histo(
            dEdy,
            "dEdy_sim_" + true_postfix,
            1.0 / static_cast<double>(AA_events),
            false);

        // dNdy
        print_1d_histo(
            dNdy,
            "dNdy_sim_" + true_postfix,
            1.0 / static_cast<double>(AA_events),
            false);

        // dNdET
        print_1d_histo(
            dNdET,
            "dNdET_sim_" + true_postfix,
            1.0,
            false);

        // dETdeta
        print_1d_histo(
            dETdeta,
            "dETdeta_sim_" + true_postfix,
            1.0 / static_cast<double>(AA_events),
            false);

        // dEdeta
        print_1d_histo(
            dEdeta,
            "dEdeta_sim_" + true_postfix,
            1.0 / static_cast<double>(AA_events),
            false);

        // dETdb
        print_1d_histo(
            dETdb,
            "dETdb_sim_" + true_postfix,
            1.0,
            true);

        // dEdb
        print_1d_histo(
            dEdb,
            "dEdb_sim_" + true_postfix,
            1.0,
            true);
    }
};

#endif // IO_HELPERS_HPP
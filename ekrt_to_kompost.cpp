// Converts MC-EKRT end-state minijets (jets_<bin>_<name>.dat) into the energy-momentum tensor
// T^{mu nu}(tau0) on a 3D grid, the input of KoMPoST3D (KoMPoST run on every eta_s slice).
//
// Initial-state prescription of Kuha et al., PRC 111 (2025) 054914 [arXiv:2406.17592], Sec. V.B:
//  - every parton is produced at (x0, y0, z = 0) at t = 0 and free-streams to tau0, so eta_s = y and
//    x_perp(tau0) = x0 + tau0 * pT_vec / pT
//  - the spatial delta functions are smeared with Gaussians of widths sigma_perp (fm) and
//    sigma_eta, cut at +-3 sigma and renormalized on the grid (C_perp, C_par in Eqs. 66-67)
// Each parton contributes p^mu p^nu / (tau0 p^tau) times the smearing kernel. With eta_s = y its
// Milne momentum is (pT, pT cos phi, pT sin phi, 0), so T^{tau eta} = T^{x eta} = T^{y eta} =
// T^{eta eta} = 0 and T^{tautau} = 1/tau0 sum_i pT_i g_perp g_par (Eq. 69).
//
// Build: part of the MC-EKRT CMake project (target ekrt_to_kompost), or
//   g++ -O2 -std=c++17 ekrt_to_kompost.cpp -o ekrt_to_kompost

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace std;

static constexpr double hbarc = 0.1973269804; // GeV fm

// ---------------------------------------------------------------- MC-EKRT binary jet file

// Fields in the order io::append_single_coll_binary writes them, with their sizes on disk.
// Which of them are present is decided by the output_params file MC-EKRT was run with.
struct Field
{
    const char *name;
    size_t size;
};
static const vector<Field> all_fields = {
    {"t01", sizeof(double)}, {"t02", sizeof(double)}, {"x", sizeof(double)}, {"y", sizeof(double)},
    {"pt", sizeof(double)}, {"y1", sizeof(double)}, {"y2", sizeof(double)}, {"phi", sizeof(double)},
    {"tata", sizeof(double)}, {"init1", sizeof(int_fast16_t)}, {"init2", sizeof(int_fast16_t)},
    {"final1", sizeof(int_fast16_t)}, {"final2", sizeof(int_fast16_t)}, {"ia", sizeof(uint_fast16_t)},
    {"ib", sizeof(uint_fast16_t)}, {"xa", sizeof(double)}, {"ya", sizeof(double)}, {"za", sizeof(double)},
    {"xb", sizeof(double)}, {"yb", sizeof(double)}, {"zb", sizeof(double)},
    {"a_is_neutron", sizeof(bool)}, {"b_is_neutron", sizeof(bool)}};

struct Dijet
{
    double x, y, pt, y1, y2, phi;
    bool has_phi;
};

// Byte offsets of the fields we need inside one dijet record (-1 = not written).
struct Layout
{
    size_t record_size = 0;
    long off_x = -1, off_y = -1, off_pt = -1, off_y1 = -1, off_y2 = -1, off_phi = -1;
};

// Mirrors io::parse_output_params: no file -> every field is written; otherwise only the
// fields named (first word of each line) are.
Layout read_layout(const string &output_params)
{
    vector<bool> present(all_fields.size(), true);
    ifstream fin(output_params);
    if (!fin.is_open())
        cerr << "Warning: cannot open " << output_params
             << ", assuming MC-EKRT wrote all fields (its default)\n";
    else
    {
        present.assign(all_fields.size(), false);
        string line;
        while (getline(fin, line))
        {
            istringstream ls(line);
            string name;
            ls >> name;
            for (size_t i = 0; i < all_fields.size(); i++)
                if (name == all_fields[i].name)
                    present[i] = true;
        }
    }

    Layout l;
    for (size_t i = 0; i < all_fields.size(); i++)
    {
        if (!present[i])
            continue;
        const string name = all_fields[i].name;
        const long off = static_cast<long>(l.record_size);
        if (name == "x") l.off_x = off;
        else if (name == "y") l.off_y = off;
        else if (name == "pt") l.off_pt = off;
        else if (name == "y1") l.off_y1 = off;
        else if (name == "y2") l.off_y2 = off;
        else if (name == "phi") l.off_phi = off;
        l.record_size += all_fields[i].size;
    }
    if (l.off_x < 0 || l.off_y < 0 || l.off_pt < 0 || l.off_y1 < 0 || l.off_y2 < 0)
    {
        cerr << "Error: the jet file must contain x, y, pt, y1 and y2 (check " << output_params << ")\n";
        exit(1);
    }
    return l;
}

// File: uint64 n_events, then per event: uint64 n_dijets, n_dijets records.
vector<vector<Dijet>> read_jets(const string &filename, const Layout &l, uint64_t max_events)
{
    ifstream fin(filename, ios::binary);
    if (!fin.is_open())
    {
        cerr << "Error: cannot open " << filename << "\n";
        exit(1);
    }
    auto get = [](const vector<char> &rec, long off)
    {
        double v;
        memcpy(&v, rec.data() + off, sizeof v);
        return v;
    };

    uint_fast64_t n_events = 0;
    fin.read(reinterpret_cast<char *>(&n_events), sizeof n_events);
    if (!fin)
    {
        cerr << "Error: " << filename << " is empty\n";
        exit(1);
    }
    const uint64_t n_read = max_events > 0 ? min<uint64_t>(n_events, max_events) : n_events;

    vector<vector<Dijet>> events(n_read);
    vector<char> rec(l.record_size);
    for (uint64_t ev = 0; ev < n_read; ev++)
    {
        uint_fast64_t n_dijets = 0;
        fin.read(reinterpret_cast<char *>(&n_dijets), sizeof n_dijets);
        events[ev].reserve(n_dijets);
        for (uint64_t j = 0; j < n_dijets; j++)
        {
            fin.read(rec.data(), rec.size());
            Dijet d;
            d.x = get(rec, l.off_x);
            d.y = get(rec, l.off_y);
            d.pt = get(rec, l.off_pt);
            d.y1 = get(rec, l.off_y1);
            d.y2 = get(rec, l.off_y2);
            d.has_phi = l.off_phi >= 0;
            d.phi = d.has_phi ? get(rec, l.off_phi) : 0.0;
            events[ev].push_back(d);
        }
        if (!fin)
        {
            cerr << "Error: " << filename << " ended inside event " << ev
                 << " - does output_params match the run that wrote it?\n";
            exit(1);
        }
    }
    // A layout mismatch that happens to stay in bounds shows up as leftover bytes.
    if (n_read == n_events && fin.peek() != EOF)
    {
        cerr << "Error: " << filename << " has data after the last event"
             << " - does output_params match the run that wrote it?\n";
        exit(1);
    }
    cout << "Read " << n_read << " of " << n_events << " events from " << filename << "\n";
    return events;
}

// ---------------------------------------------------------------- grid and smearing

// Same convention as TRENTo3d / IcTrento3d: n points from -max to +max inclusive.
// Components of T^{mu nu} that a parton with eta_s = y can produce, in units of pT / tau0.
enum Comp { TT, TX, TY, XX, XY, YY, NCOMP_MAX };

struct Grid
{
    int nxy, neta;
    double xy_max, eta_max, dxy, deta;
    vector<vector<double>> t; // t[c]: component c [GeV/fm^3], index (ix * nxy + iy) * neta + ieta
    vector<double> &e;        // t[TT] = T^{tautau}, the energy density at zero flow

    Grid(int nxy_, double xy_max_, int neta_, double eta_max_)
        : nxy(nxy_), neta(neta_), xy_max(xy_max_), eta_max(eta_max_),
          dxy(2 * xy_max_ / (nxy_ - 1)), deta(2 * eta_max_ / (neta_ - 1)),
          t(NCOMP_MAX, vector<double>(static_cast<size_t>(nxy_) * nxy_ * neta_, 0.0)), e(t[TT]) {}

    double x(int i) const { return -xy_max + i * dxy; }
    double eta(int i) const { return -eta_max + i * deta; }
    size_t index(int ix, int iy, int ieta) const { return (static_cast<size_t>(ix) * nxy + iy) * neta + ieta; }
    void clear()
    {
        for (auto &c : t)
            fill(c.begin(), c.end(), 0.0);
    }
};

struct Settings
{
    double tau0 = hbarc / 1.0; // 1/p0 with p0 = 1 GeV, as in the paper
    double sigma_perp = 0.15;  // fm
    double sigma_eta = 0.15;
    double cutoff = 3.0;       // in units of sigma
};

struct Bookkeeping
{
    uint64_t partons = 0;
    double sum_pt = 0;        // sum pT: should equal tau0 * int e dx dy deta
    double sum_E = 0;         // sum pT cosh y: the partons' energy
    double lost_pt = 0;       // pT smeared outside the grid
    double sum_pt_midrap = 0; // sum pT for |y| < 0.5
};

// Adds pT/tau0 * f_c * g_perp(x - xi) * g_par(eta - etai) of one parton to every component c of the
// grid, with f = (1, cos phi, sin phi, cos^2 phi, cos phi sin phi, sin^2 phi). The discrete
// Gaussians are normalized over their full +-cutoff stencil (C_perp, C_par), so a parton fully
// inside the grid deposits exactly pT/tau0 * (dx dy deta)^-1 summed over cells; the part of a
// stencil outside the grid is dropped and counted in lost_pt.
void deposit(Grid &g, const Settings &s, double xi, double yi, double etai, double pt, double cphi,
             double sphi, Bookkeeping &bk)
{
    const int rxy = static_cast<int>(ceil(s.cutoff * s.sigma_perp / g.dxy));
    const int reta = static_cast<int>(ceil(s.cutoff * s.sigma_eta / g.deta));
    const int cx = static_cast<int>(lround((xi + g.xy_max) / g.dxy));
    const int cy = static_cast<int>(lround((yi + g.xy_max) / g.dxy));
    const int ce = static_cast<int>(lround((etai + g.eta_max) / g.deta));

    auto gauss_weights = [](int c, int r, double centre, double sigma, double step, double min_coord, double max_dist)
    {
        vector<double> w(2 * r + 1);
        double norm = 0;
        for (int k = -r; k <= r; k++)
        {
            const double d = min_coord + (c + k) * step - centre;
            const double v = fabs(d) <= max_dist ? exp(-0.5 * d * d / (sigma * sigma)) : 0.0;
            w[k + r] = v;
            norm += v * step;
        }
        for (double &v : w)
            v = norm > 0 ? v / norm : 0.0;
        return w;
    };
    const auto wx = gauss_weights(cx, rxy, xi, s.sigma_perp, g.dxy, -g.xy_max, s.cutoff * s.sigma_perp);
    const auto wy = gauss_weights(cy, rxy, yi, s.sigma_perp, g.dxy, -g.xy_max, s.cutoff * s.sigma_perp);
    const auto we = gauss_weights(ce, reta, etai, s.sigma_eta, g.deta, -g.eta_max, s.cutoff * s.sigma_eta);

    const double amp = pt / s.tau0;
    const double f[NCOMP_MAX] = {1.0, cphi, sphi, cphi * cphi, cphi * sphi, sphi * sphi};
    double deposited = 0; // in units of pT
    for (int kx = -rxy; kx <= rxy; kx++)
    {
        const int ix = cx + kx;
        if (ix < 0 || ix >= g.nxy || wx[kx + rxy] == 0)
            continue;
        for (int ky = -rxy; ky <= rxy; ky++)
        {
            const int iy = cy + ky;
            if (iy < 0 || iy >= g.nxy || wy[ky + rxy] == 0)
                continue;
            const double wxy = wx[kx + rxy] * wy[ky + rxy];
            for (int ke = -reta; ke <= reta; ke++)
            {
                const int ie = ce + ke;
                if (ie < 0 || ie >= g.neta)
                    continue;
                const double w = wxy * we[ke + reta];
                const size_t i = g.index(ix, iy, ie);
                for (int c = 0; c < NCOMP_MAX; c++)
                    g.t[c][i] += amp * w * f[c];
                deposited += w * g.dxy * g.dxy * g.deta;
            }
        }
    }
    bk.lost_pt += pt * (1.0 - deposited);
}

void add_event(Grid &g, const Settings &s, const vector<Dijet> &event, mt19937_64 &rng, Bookkeeping &bk)
{
    uniform_real_distribution<double> uni(0.0, 2 * M_PI);
    for (const Dijet &d : event)
    {
        // MC-EKRT samples phi uniformly for parton 1; parton 2 is back to back
        const double phi = d.has_phi ? d.phi : uni(rng);
        const double cphi = cos(phi), sphi = sin(phi);
        const double rap[2] = {d.y1, d.y2};
        const double dir[2] = {1.0, -1.0};
        for (int p = 0; p < 2; p++)
        {
            const double xi = d.x + s.tau0 * dir[p] * cphi;
            const double yi = d.y + s.tau0 * dir[p] * sphi;
            deposit(g, s, xi, yi, rap[p], d.pt, dir[p] * cphi, dir[p] * sphi, bk);
            bk.partons++;
            bk.sum_pt += d.pt;
            bk.sum_E += d.pt * cosh(rap[p]);
            if (fabs(rap[p]) < 0.5)
                bk.sum_pt_midrap += d.pt;
        }
    }
}

// ---------------------------------------------------------------- output

// TMUNU3D binary file, read by KoMPoST/src/Main3D.cpp:
//   char magic[8] = "TMUNU3D", int32 version = 1, ncomp = 10, nxy, neta,
//   double tau [fm], xy_max [fm], eta_max, then nxy*nxy*neta*ncomp doubles ordered
//   ieta (slowest), iy, ix, component (fastest).
// x, y run over nxy points from -xy_max to +xy_max, eta_s over neta points from -eta_max to +eta_max.
// Components are T^{mu nu} [GeV/fm^3] with upper indices in Milne coordinates, the eta index
// multiplied by tau (as vHLLE stores pi^{mu nu}), in the order
//   tt, tx, ty, tz, xx, xy, xz, yy, yz, zz    (z = tau * eta)
void write_tmunu(const string &filename, const Grid &g, double scale, double tau0)
{
    ofstream out(filename, ios::binary);
    if (!out)
    {
        cerr << "Error: cannot write " << filename << "\n";
        exit(1);
    }
    const char magic[8] = "TMUNU3D";
    const int32_t head[4] = {1, 10, g.nxy, g.neta};
    const double geom[3] = {tau0, g.xy_max, g.eta_max};
    out.write(magic, sizeof magic);
    out.write(reinterpret_cast<const char *>(head), sizeof head);
    out.write(reinterpret_cast<const char *>(geom), sizeof geom);
    vector<double> row(static_cast<size_t>(g.nxy) * 10);
    for (int ie = 0; ie < g.neta; ie++)
        for (int iy = 0; iy < g.nxy; iy++)
        {
            for (int ix = 0; ix < g.nxy; ix++)
            {
                const size_t i = g.index(ix, iy, ie);
                double *v = &row[static_cast<size_t>(ix) * 10];
                v[0] = scale * g.t[TT][i];
                v[1] = scale * g.t[TX][i];
                v[2] = scale * g.t[TY][i];
                v[3] = 0.0;
                v[4] = scale * g.t[XX][i];
                v[5] = scale * g.t[XY][i];
                v[6] = 0.0;
                v[7] = scale * g.t[YY][i];
                v[8] = 0.0;
                v[9] = 0.0;
            }
            out.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
        }
    if (!out)
    {
        cerr << "Error: writing " << filename << " failed\n";
        exit(1);
    }
}

void report(const Grid &g, const Settings &s, const Bookkeeping &bk, int nevents)
{
    // tau0 * int e cosh(eta) dx dy deta is the fluid energy; with eta_s = y it differs from the
    // partons' energy only through the eta smearing (paper: ~1% for sigma_eta = 0.15)
    double int_e = 0, int_E = 0;
    for (int ix = 0; ix < g.nxy; ix++)
        for (int iy = 0; iy < g.nxy; iy++)
            for (int ie = 0; ie < g.neta; ie++)
            {
                const double v = g.e[(static_cast<size_t>(ix) * g.nxy + iy) * g.neta + ie];
                int_e += v;
                int_E += v * cosh(g.eta(ie));
            }
    const double cell = s.tau0 * g.dxy * g.dxy * g.deta / nevents;
    double emax = 0;
    for (double v : g.e)
        emax = max(emax, v / nevents);
    cout << fixed << setprecision(4)
         << "  partons/event                 " << double(bk.partons) / nevents << "\n"
         << "  sum pT/event      [GeV]       " << bk.sum_pt / nevents << "\n"
         << "  tau0 int e        [GeV]       " << int_e * cell
         << "   (pT smeared off-grid: " << 100 * bk.lost_pt / bk.sum_pt << " %)\n"
         << "  parton E/event    [GeV]       " << bk.sum_E / nevents << "\n"
         << "  fluid E/event     [GeV]       " << int_E * cell
         << "   (" << showpos << 100 * (int_E * cell / (bk.sum_E / nevents) - 1) << noshowpos << " %)\n"
         << "  dET/dy |y|<0.5    [GeV]       " << bk.sum_pt_midrap / nevents << "\n"
         << "  max e(tau0)       [GeV/fm^3]  " << emax << "\n"
         << defaultfloat;
}

// ---------------------------------------------------------------- main

void usage(const char *prog)
{
    cerr << "Usage: " << prog << " <jets_file> [options]\n"
         << "\nConverts MC-EKRT minijets into T^{mu nu}(tau0) on a 3D grid, a binary TMUNU3D file for KoMPoST3D.\n"
         << "\nOutput (one of):\n"
         << "  -o, --output FILE        average all events of the file into FILE (default: <jets_file stem>_tmunu.bin)\n"
         << "  --each DIR               one file per event, DIR/tmunu_<k>.bin\n"
         << "\nPhysics:\n"
         << "  --tau0 FM                proper time of the T^{mu nu} [fm], KoMPoST's tIn\n"
         << "                           (default hbar c / p0 = 0.1973, p0 = 1 GeV)\n"
         << "  --p0 GEV                 set tau0 = hbar c / p0 instead\n"
         << "  --sigma-perp FM          transverse Gaussian width [fm] (default 0.15)\n"
         << "  --sigma-eta X            longitudinal Gaussian width in eta_s (default 0.15)\n"
         << "  --cutoff N               smearing cut-off in sigmas (default 3)\n"
         << "\nGrid (n points from -max to +max; KoMPoST3D and vHLLE use the same grid):\n"
         << "  --nxy N                  transverse points (default 201)\n"
         << "  --xy-max FM              transverse half width [fm] (default 15)\n"
         << "  --neta N                 eta_s points (default 135)\n"
         << "  --eta-max X              eta_s half width (default 10)\n"
         << "\nOther:\n"
         << "  --output-params FILE     MC-EKRT output_params the jets were written with\n"
         << "                           (default: output_params next to the jets file)\n"
         << "  --events N               use only the first N events (default all)\n"
         << "  --seed N                 seed for the azimuth when phi is not in the file (default 1)\n";
}

int main(int argc, char *argv[])
{
    if (argc < 2 || string(argv[1]) == "-h" || string(argv[1]) == "--help")
    {
        usage(argv[0]);
        return argc < 2;
    }
    const string jets_file = argv[1];
    Settings s;
    int nxy = 201, neta = 135;
    double xy_max = 15.0, eta_max = 10.0;
    string output, each_dir, output_params;
    uint64_t max_events = 0;
    uint64_t seed = 1;

    for (int i = 2; i < argc; i++)
    {
        const string a = argv[i];
        auto next = [&]() -> string
        {
            if (i + 1 >= argc)
            {
                cerr << "Error: " << a << " needs a value\n";
                exit(1);
            }
            return argv[++i];
        };
        if (a == "-o" || a == "--output") output = next();
        else if (a == "--each") each_dir = next();
        else if (a == "--tau0") s.tau0 = stod(next());
        else if (a == "--p0") s.tau0 = hbarc / stod(next());
        else if (a == "--sigma-perp") s.sigma_perp = stod(next());
        else if (a == "--sigma-eta") s.sigma_eta = stod(next());
        else if (a == "--cutoff") s.cutoff = stod(next());
        else if (a == "--nxy") nxy = stoi(next());
        else if (a == "--xy-max") xy_max = stod(next());
        else if (a == "--neta") neta = stoi(next());
        else if (a == "--eta-max") eta_max = stod(next());
        else if (a == "--output-params") output_params = next();
        else if (a == "--events") max_events = stoull(next());
        else if (a == "--seed") seed = stoull(next());
        else
        {
            cerr << "Error: unknown option " << a << "\n";
            usage(argv[0]);
            return 1;
        }
    }
    if (!output.empty() && !each_dir.empty())
    {
        cerr << "Error: give either --output or --each\n";
        return 1;
    }
    if (nxy < 2 || neta < 2 || xy_max <= 0 || eta_max <= 0 || s.tau0 <= 0 || s.sigma_perp <= 0 ||
        s.sigma_eta <= 0 || s.cutoff <= 0)
    {
        cerr << "Error: grid sizes, tau0, widths and cut-off must be positive (n >= 2)\n";
        return 1;
    }
    if (output_params.empty())
        output_params = (fs::path(jets_file).parent_path() / "output_params").string();
    if (output.empty() && each_dir.empty())
        output = fs::path(jets_file).stem().string() + "_tmunu.bin";

    Grid g(nxy, xy_max, neta, eta_max);
    const Layout layout = read_layout(output_params);
    const auto events = read_jets(jets_file, layout, max_events);
    if (events.empty())
    {
        cerr << "Error: no events in " << jets_file << "\n";
        return 1;
    }
    if (layout.off_phi < 0)
        cout << "phi is not in the jet file: sampling it uniformly (seed " << seed << "), as MC-EKRT does\n";

    cout << "tau0 = " << s.tau0 << " fm, sigma_perp = " << s.sigma_perp << " fm, sigma_eta = " << s.sigma_eta
         << ", cut-off " << s.cutoff << " sigma\n"
         << "grid: " << nxy << " x " << nxy << " x " << neta << ", x,y in [-" << xy_max << ", " << xy_max
         << "] fm (step " << g.dxy << "), eta_s in [-" << eta_max << ", " << eta_max << "] (step " << g.deta << ")\n";
    if (s.sigma_perp < g.dxy || s.sigma_eta < g.deta)
        cout << "Warning: a smearing width is below the grid step; the Gaussians are under-resolved\n";

    mt19937_64 rng(seed);

    if (!each_dir.empty())
    {
        fs::create_directories(each_dir);
        // zero-padded so that a shell glob lists the events in order
        const int width = static_cast<int>(to_string(events.size() - 1).size());
        for (size_t ev = 0; ev < events.size(); ev++)
        {
            g.clear();
            Bookkeeping bk;
            add_event(g, s, events[ev], rng, bk);
            ostringstream name;
            name << "tmunu_" << setw(width) << setfill('0') << ev << ".bin";
            write_tmunu((fs::path(each_dir) / name.str()).string(), g, 1.0, s.tau0);
            if (ev == 0)
            {
                cout << "event 0:\n";
                report(g, s, bk, 1);
            }
        }
        cout << "Wrote " << events.size() << " files to " << each_dir << "\n";
    }
    else
    {
        Bookkeeping bk;
        for (const auto &ev : events)
            add_event(g, s, ev, rng, bk);
        cout << "average over " << events.size() << " events:\n";
        report(g, s, bk, static_cast<int>(events.size()));
        write_tmunu(output, g, 1.0 / events.size(), s.tau0);
        cout << "Wrote " << output << "\n";
    }
    cout << "KoMPoST3D: tIn = " << s.tau0 << " fm and the grid are taken from the file\n";
    return 0;
}

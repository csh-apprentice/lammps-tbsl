// Isolated tests of the production pair style. No particle integration, wall,
// continuum solver, input data files, or production simulation is involved.
#include "atom.h"
#include "force.h"
#include "input.h"
#include "lammps.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair_lj_cutio_coul_dsf.h"

#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace LAMMPS_NS;

namespace {
int rank, nranks, failures = 0, checks = 0;
std::string current;
using Vec = std::array<double, 3>;

std::string number(double value)
{
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

void check(bool condition, const std::string &message)
{
  ++checks;
  if (!condition) {
    ++failures;
    if (rank == 0) std::cerr << "FAIL " << current << ": " << message << '\n';
  }
}

void near(double got, double expected, const std::string &message, double tolerance = 2e-12)
{
  check(std::isfinite(got) && std::abs(got - expected) <=
    tolerance * std::max(1.0, std::abs(expected)),
    message + ": got " + number(got) + ", expected " + number(expected));
}

struct Particle {
  Vec x, v;
  double mass = 2.0, q = 0.0;
};

// Exercise the CPU ionization pipeline with the same list topology requested
// by the GPU wrapper, without needing a GPU or testing GPU force arithmetic.
class PairFullList : public PairLJCutIOCoulDSF {
 public:
  explicit PairFullList(LAMMPS *lmp) : PairLJCutIOCoulDSF(lmp) {}
  void init_style() override
  {
    PairLJCutIOCoulDSF::init_style();
    neighbor->find_request(this)->enable_full();
  }
};

struct Fixture {
  std::unique_ptr<LAMMPS> lmp;
  std::vector<Particle> initial;
  PairLJCutIOCoulDSF *pair;
  int size;

  Fixture(std::vector<Particle> particles, bool newton = true,
          const std::vector<double> &thresholds = {10000.0, 1e9}, double g = 1.0,
          bool full_list = false)
    : initial(std::move(particles)), size(initial.size())
  {
    const char *args[] = {"tbsl-test", "-screen", "none", "-log", "none", "-nocite"};
    lmp.reset(new LAMMPS(6, const_cast<char **>(args), MPI_COMM_WORLD));
    command("units real");
    command("atom_style charge");
    command("atom_modify map array sort 0 0.0");
    command("boundary f f f");
    command(std::string("newton ") + (newton ? "on" : "off"));
    command("processors " + std::to_string(nranks) + " 1 1");
    command("region box block 0 12 0 12 0 12");
    command("create_box 1 box");
    for (const auto &p : initial)
      command("create_atoms 1 single " + number(p.x[0]) + " " + number(p.x[1]) +
              " " + number(p.x[2]) + " units box");
    command("mass 1 2");
    command("fix mass all property/atom rmass ghost yes");
    command("fix tlast all property/atom i_tlast");
    command("fix label all property/atom i_label ghost yes");
    command("fix distance all property/atom d_mindist ghost yes");
    command("set atom * i_label -1");
    command("set atom * d_mindist 1e20");
    for (int k = 0; k < size; ++k) {
      const auto id = std::to_string(k + 1);
      const auto &p = initial[k];
      command("set atom " + id + " mass " + number(p.mass));
      command("set atom " + id + " charge " + number(p.q));
      command("group selected id " + id);
      command("velocity selected set " + number(p.v[0]) + " " + number(p.v[1]) +
              " " + number(p.v[2]) + " units box");
      command("group selected delete");
    }
    command("comm_modify vel yes");
    command("neighbor 0.2 bin");
    command("variable active equal 0");
    (*lmp->force->pair_map)["test/ion/full"] = [](LAMMPS *lmp) -> Pair * {
      return new PairFullList(lmp);
    };
    command(std::string("pair_style ") + (full_list ? "test/ion/full" : "lj/cutio/coul/dsf") +
            " 1.5 2 2 2");
    std::string coeff = "pair_coeff * * " + std::to_string(thresholds.size());
    for (double threshold : thresholds) coeff += " " + number(threshold);
    command(coeff + " 0 1 v_active " + number(g));
    command("run 0"); // Build ghosts and neighbor lists with ionization inactive.
    command("variable active equal 1");
    pair = dynamic_cast<PairLJCutIOCoulDSF *>(lmp->force->pair);
  }

  void command(const std::string &line) { lmp->input->one(line); }
  void evaluate() { pair->compute(0, 0); }

  // Read owners only. A correct-looking ghost velocity cannot hide a missing
  // update on the rank that actually integrates the particle.
  std::vector<Particle> state()
  {
    std::vector<double> local(8 * size, 0.0), global(8 * size);
    const auto atom = lmp->atom;
    for (int i = 0; i < atom->nlocal; ++i) {
      const int k = atom->tag[i] - 1;
      for (int d = 0; d < 3; ++d) {
        local[8*k + d] = atom->x[i][d];
        local[8*k + 3 + d] = atom->v[i][d];
      }
      local[8*k + 6] = atom->q[i];
      local[8*k + 7] = 1.0;
    }
    MPI_Allreduce(local.data(), global.data(), global.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    auto result = initial;
    for (int k = 0; k < size; ++k) {
      near(global[8*k + 7], 1.0, "one owner per particle");
      for (int d = 0; d < 3; ++d) {
        result[k].x[d] = global[8*k + d];
        result[k].v[d] = global[8*k + 3 + d];
      }
      result[k].q = global[8*k + 6];
    }
    return result;
  }
};

std::vector<Particle> pair_at_boundary(Vec boost = {{0, 0, 0}})
{
  std::vector<Particle> p(2);
  p[0].x = {{5.75, 6, 6}};
  p[1].x = {{6.25, 6, 6}};
  p[0].v = p[1].v = boost;
  p[0].v[0] += 1;
  p[1].v[0] -= 1;
  return p;
}

double kinetic(const std::vector<Particle> &p)
{
  double total = 0;
  for (const auto &a : p)
    for (int d = 0; d < 3; ++d) total += 0.5 * 1e4 * a.mass * a.v[d] * a.v[d];
  return total;
}

void invariant_event(Vec boost, bool newton, double g = 1, bool full_list = false)
{
  auto p = pair_at_boundary(boost);
  for (auto &a : p) a.mass *= g;
  Fixture f(p, newton, {10000*g, 1e9*g}, g, full_list);
  f.evaluate(); // Select candidates only.
  auto selected = f.state();
  near(selected[0].q + selected[1].q, 0, "selection does not ionize");
  f.evaluate(); // Execute the previously selected event.
  const auto out = f.state();
  near(out[0].q, 1, "lower-tag particle ionized once");
  near(out[1].q, 0, "partner charge unchanged");
  const double speed = std::sqrt(1.0/3.0);
  for (int d = 0; d < 3; ++d) {
    near(out[0].v[d], boost[d] + (d == 0 ? speed : 0), "first owner velocity");
    near(out[1].v[d], boost[d] - (d == 0 ? speed : 0), "second owner velocity");
    near(p[0].mass*out[0].v[d] + p[1].mass*out[1].v[d],
         p[0].mass*p[0].v[d] + p[1].mass*p[1].v[d], "pair momentum");
  }
  near(kinetic(p) - kinetic(out), 40000*g/3, "ionization plus electron heating loss");
}

void no_relative_motion()
{
  auto p = pair_at_boundary();
  p[0].v = p[1].v = {{8, -4, 2}};
  Fixture f(p);
  f.evaluate();
  f.evaluate();
  const auto out = f.state();
  for (int k = 0; k < 2; ++k) {
    near(out[k].q, 0, "common translation cannot ionize");
    for (int d = 0; d < 3; ++d) near(out[k].v[d], p[k].v[d], "velocity unchanged");
  }
}

void threshold_edges()
{
  for (double chi : {20000.0 - 1e-6, 20000.0, 20000.0 + 1e-6}) {
    Fixture f(pair_at_boundary(), true, {chi, 1e9});
    f.evaluate();
    f.evaluate();
    const auto out = f.state();
    near(out[0].q + out[1].q, chi < 20000 ? 1 : 0, "strict COM threshold");
  }
}

void unequal_masses()
{
  auto p = pair_at_boundary();
  p[0].mass = 1;
  p[1].mass = 3;
  p[0].v = {{3, 2, -1}};
  p[1].v = {{1, 2, -1}};
  Fixture f(p, true, {7500, 1e9}); // E_rel = 15000; COM = (1.5,2,-1).
  f.evaluate();
  f.evaluate();
  const auto out = f.state();
  const double s = std::sqrt(1.0/3.0);
  near(out[0].v[0], 1.5 + 1.5*s, "mass-weighted first velocity");
  near(out[1].v[0], 1.5 - 0.5*s, "mass-weighted second velocity");
  near(kinetic(p) - kinetic(out), 10000, "unequal-mass kinetic loss");
}

void saturated_states()
{
  for (int saturated = 0; saturated < 3; ++saturated) {
    auto p = pair_at_boundary();
    if (saturated == 0) p[0].q = p[1].q = 1; // Last transition 1 -> 2.
    if (saturated == 1) p[0].q = 2;          // Saturated atom can be a partner.
    if (saturated == 2) p[0].q = p[1].q = 2;
    Fixture f(p, true, {5000, 10000});
    f.evaluate();
    f.evaluate();
    const auto out = f.state();
    near(out[0].q, 2, "valid terminal charge");
    near(out[1].q, saturated == 2 ? 2 : 1, "partner charge");
  }
}

void partner_index_zero()
{
  auto p = pair_at_boundary();
  const double subdomain_width = 12.0 / nranks;
  p[0].x[0] = 0.2 * subdomain_width;
  p[1].x[0] = p[0].x[0] + std::min(0.5, 0.4 * subdomain_width);
  // Both endpoints must be owned by rank 0: this deliberately invokes the
  // local-pair event routine on that rank alone, without its MPI pipeline.
  Fixture f(p);
  check(f.lmp->atom->nlocal == (rank == 0 ? 2 : 0),
        "index-zero fixture keeps both endpoints on rank zero");
  f.evaluate();
  const int old_inum = f.pair->list->inum;
  if (rank == 0) {
    near(f.lmp->atom->map(1), 0, "fixture partner is actually at index zero");
    const int old_first = f.pair->list->ilist[0];
    f.pair->list->ilist[0] = f.lmp->atom->map(2);
    f.pair->list->inum = 1;
    // Call the production event routine from this endpoint only. Otherwise
    // the opposite ilist row could mask the old jlocal > 0 bug.
    f.pair->ionization(f.lmp->atom->v, f.lmp->atom->q,
                      f.lmp->atom->ivector[1], f.lmp->atom->dvector[0]);
    f.pair->list->ilist[0] = old_first;
    f.pair->list->inum = old_inum;
  }
  const auto out = f.state();
  near(out[0].q, 1, "reordered ilist processes index-zero partner");
  near(kinetic(out), 20000.0/3, "both velocities updated once");
}

void recheck_after_labeling()
{
  Fixture f(pair_at_boundary());
  f.evaluate();
  f.command("velocity all set 8 -4 2 units box");
  f.evaluate();
  const auto out = f.state();
  near(out[0].q + out[1].q, 0, "event rechecks current COM energy after labeling");
  for (const auto &p : out) near(p.v[0], 8, "rejected event leaves velocity unchanged");
}

void eligible_partner_selection()
{
  auto p = pair_at_boundary();
  p[0].v = p[1].v = {{8, 0, 0}};
  p[1].x[0] = 6;
  Particle third;
  third.x = {{6.75, 6, 6}};
  third.v = {{6, 0, 0}};
  p.push_back(third);
  Fixture f(p);
  f.evaluate();
  f.evaluate();
  const auto out = f.state();
  near(out[0].q, 0, "nearest comoving neighbor is ineligible");
  near(out[0].v[0], 8, "non-mutual particle untouched");
  near(out[1].q, 1, "eligible relative-energy partner selected");
  near(out[1].v[0], 7 + std::sqrt(1.0/3.0), "selected pair cooled in its COM frame");
}

void tied_partners()
{
  auto p = pair_at_boundary();
  p[0].x[0] = 6;
  p[1].x[0] = 5.5;
  Particle third;
  third.x = {{6.5, 6, 6}};
  third.v = {{-1, 0, 0}};
  p.push_back(third);
  Fixture f(p);
  f.evaluate();
  std::reverse(f.pair->list->ilist, f.pair->list->ilist + f.pair->list->inum);
  f.evaluate();
  const auto out = f.state();
  near(out[0].q, 1, "one event for tied partners");
  near(out[1].v[0], -std::sqrt(1.0/3.0), "lower-tag equal-distance partner wins");
  near(out[2].v[0], -1, "other equal-distance candidate untouched");
}

void one_event_per_round()
{
  Fixture f(pair_at_boundary(), true, {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000});
  f.evaluate();
  f.evaluate();
  auto out = f.state();
  near(out[0].q, 1, "first accepted event increments one atom");
  near(out[1].q, 0, "no second event in the same round despite excess energy");
  f.evaluate();
  out = f.state();
  near(out[0].q, 1, "first atom retains its charge");
  near(out[1].q, 1, "next round selects the lower next threshold");
}

void migrating_pair()
{
  Fixture f(pair_at_boundary());
  f.evaluate();
  f.command("variable active equal 0");
  f.command("set atom 1 x 6.125");
  f.command("run 0"); // Atom 1 changes owner while retaining its partner label.
  f.command("variable active equal 1");
  f.evaluate();
  const auto out = f.state();
  near(out[0].q, 1, "migrated atom ionized once");
  near(out[1].q, 0, "migrated partner unchanged");
  near(out[0].v[0], std::sqrt(1.0/3.0), "new owner receives correct velocity");
  near(out[1].v[0], -std::sqrt(1.0/3.0), "partner receives correct velocity");
}

void competing_partners()
{
  auto p = pair_at_boundary();
  Particle third;
  third.x = {{5.0, 6, 6}};
  third.v = {{-1, 0, 0}};
  p.push_back(third);
  Fixture f(p);
  f.evaluate();
  f.evaluate();
  const auto out = f.state();
  near(out[0].q, 1, "mutual nearest pair accepted");
  near(out[1].v[0], -std::sqrt(1.0/3.0), "nearest partner cooled");
  near(out[2].v[0], -1, "non-mutual candidate untouched");
}

void stale_partner()
{
  Fixture f(pair_at_boundary());
  f.evaluate();
  f.command("variable active equal 0");
  // Brick exchange passes atoms to adjacent ranks. Move by less than half a
  // subdomain per rebuild instead of teleporting across several ranks when
  // this fixture is extended beyond its original 1/2/4-rank matrix.
  const int moves = static_cast<int>(std::ceil((10.5 - 6.25) / (0.5 * 12.0 / nranks)));
  for (int k = 1; k <= moves; ++k) {
    f.command("set atom 2 x " + number(6.25 + (10.5 - 6.25) * k / moves));
    f.command("run 0");
  }
  f.command("variable active equal 1");
  f.evaluate();
  const auto out = f.state();
  near(out[0].q + out[1].q, 0, "stale out-of-range pair does not ionize");
}
} // namespace

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);
  const std::string filter = argc > 1 ? argv[1] : "";
  const auto run = [&](const std::string &name, void (*test)()) {
    if (!filter.empty() && name.find(filter) == std::string::npos) return;
    current = name;
    const int before = failures;
    test();
    if (rank == 0) std::cout << (failures == before ? "PASS " : "FAIL ") << name << '\n';
  };
  run("zero_COM", [] { invariant_event({{0, 0, 0}}, true); });
  run("boost", [] { invariant_event({{3, -2, 1}}, true); });
  run("newton_off", [] { invariant_event({{3, -2, 1}}, false); });
  run("full_list", [] { invariant_event({{3, -2, 1}}, true, 1, true); });
  run("full_list_newton_off", [] { invariant_event({{3, -2, 1}}, false, 1, true); });
  run("EP_scaling", [] { invariant_event({{3, -2, 1}}, true, 1000); });
  run("stationary_particle", [] { invariant_event({{1, 0, 0}}, true); });
  run("no_relative_motion", no_relative_motion);
  run("threshold_edges", threshold_edges);
  run("unequal_masses", unequal_masses);
  run("saturated_states", saturated_states);
  run("partner_index_zero", partner_index_zero);
  run("competing_partners", competing_partners);
  run("recheck_after_labeling", recheck_after_labeling);
  run("eligible_partner_selection", eligible_partner_selection);
  run("tied_partners", tied_partners);
  run("one_event_per_round", one_event_per_round);
  run("migrating_pair", migrating_pair);
  run("stale_partner", stale_partner);
  int failed;
  MPI_Allreduce(&failures, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (rank == 0) std::cout << checks << " checks, " << failed << " failures on " << nranks << " rank(s)\n";
  MPI_Finalize();
  return failed ? 1 : 0;
}

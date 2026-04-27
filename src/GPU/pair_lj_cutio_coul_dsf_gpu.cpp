/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   GPU-accelerated lj/cutio/coul/dsf for TBSL sonoluminescence.

   The O(N²) force/energy loop runs on GPU (reusing lal_lj_cutio_dsf kernel).
   The ionization pipeline (three phases + MPI communications) runs on CPU
   before the GPU force call, because it requires random write access to
   per-atom state (q, label, mindist, v) and depends on CPU-side neighbor
   list iteration.

   Only GPU_FORCE mode is supported; GPU_NEIGH / GPU_HYB_NEIGH are rejected
   at init time because ionization must use the CPU-built neighbor list.
------------------------------------------------------------------------- */

#include "pair_lj_cutio_coul_dsf_gpu.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "input.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "platform.h"
#include "suffix.h"
#include "timer.h"
#include "variable.h"

#include <cmath>

#define MY_PIS  1.77245385090551602729
#define EWALD_P 0.3275911
#define A1      0.254829592
#define A2     -0.284496736
#define A3      1.421413741
#define A4     -1.453152027
#define A5      1.061405429

using namespace LAMMPS_NS;

// External functions from CUDA library (lal_lj_cutio_dsf_ext.cpp)
int ljiod_gpu_init(const int ntypes, double **cutsq, double **host_lj1,
                   double **host_lj2, double **host_lj3, double **host_lj4,
                   double **offset, double *special_lj, const int nlocal,
                   const int nall, const int max_nbors, const int maxspecial,
                   const double cell_size, int &gpu_mode, FILE *screen,
                   double **host_cut_ljsq, const double host_cut_coulsq,
                   double *host_special_coul, const double qqrd2e,
                   const double e_shift, const double f_shift,
                   const double alpha, const double spair, const double sself);
void ljiod_gpu_clear();
int **ljiod_gpu_compute_n(const int ago, const int inum, const int nall,
                          double **host_x, int *host_type, double *sublo,
                          double *subhi, tagint *tag, int **nspecial,
                          tagint **special, const bool eflag, const bool vflag,
                          const bool eatom, const bool vatom, int &host_start,
                          int **ilist, int **jnum, const double cpu_time,
                          bool &success, double *host_q, double *boxlo,
                          double *prd);
void ljiod_gpu_compute(const int ago, const int inum, const int nall,
                       double **host_x, int *host_type, int *ilist, int *numj,
                       int **firstneigh, const bool eflag, const bool vflag,
                       const bool eatom, const bool vatom, int &host_start,
                       const double cpu_time, bool &success, double *host_q,
                       const int nlocal, double *boxlo, double *prd);
double ljiod_gpu_bytes();

/* ---------------------------------------------------------------------- */

PairLJCutIOCoulDSFGPU::PairLJCutIOCoulDSFGPU(LAMMPS *lmp)
    : PairLJCutIOCoulDSF(lmp), gpu_mode(GPU_FORCE)
{
  respa_enable = 0;
  reinitflag = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ---------------------------------------------------------------------- */

PairLJCutIOCoulDSFGPU::~PairLJCutIOCoulDSFGPU()
{
  ljiod_gpu_clear();
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSFGPU::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  // Check whether ionization/pair is active this timestep
  double active = input->variable->compute_equal(activevar);
  if (std::abs(active) < 1e-4) return;

  // -----------------------------------------------------------------------
  // Phase 1-3: ionization pipeline (runs on CPU, uses CPU neighbor list)
  // These modify q, v, label, mindist and require MPI comms between phases.
  // -----------------------------------------------------------------------
  double **v = atom->v;
  double *q = atom->q;
  int *label = atom->ivector[1];
  double *mindist = atom->dvector[0];
  int newton_pair = force->newton_pair;

  ionization(v, q, label, mindist);

  pack_flag = 1;
  comm->forward_comm(this);
  if (newton_pair) {
    pack_flag = 1;
    comm->reverse_comm(this);
  }

  reset(label, mindist);

  pack_flag = 2;
  comm->forward_comm(this);
  if (newton_pair) {
    pack_flag = 2;
    comm->reverse_comm(this);
  }

  labelnext(v, q, label, mindist);

  pack_flag = 3;
  comm->forward_comm(this);
  if (newton_pair) {
    pack_flag = 3;
    comm->reverse_comm(this);
  }
  timer->stamp(Timer::IONIZATION);

  // -----------------------------------------------------------------------
  // Phase 4: GPU force computation (uses updated q after ionization)
  // Only GPU_FORCE mode is supported; neighbor list is CPU-built.
  // -----------------------------------------------------------------------
  int nall = atom->nlocal + atom->nghost;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int host_start;

  bool success = true;

  ljiod_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->type,
                    ilist, numneigh, firstneigh, eflag, vflag,
                    eflag_atom, vflag_atom, host_start, cpu_time, success,
                    atom->q, atom->nlocal, domain->boxlo, domain->prd);

  if (!success) error->one(FLERR, "Insufficient memory on accelerator");

  if (atom->molecular != Atom::ATOMIC && neighbor->ago == 0)
    neighbor->build_topology();

  // Handle any atoms not processed by GPU (host_start < inum means GPU
  // handled [0, host_start) and CPU must handle [host_start, inum))
  if (host_start < inum) {
    cpu_time = platform::walltime();
    cpu_compute(host_start, inum, eflag, vflag, ilist, numneigh, firstneigh);
    cpu_time = platform::walltime() - cpu_time;
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSFGPU::init_style()
{
  if (!atom->q_flag)
    error->all(FLERR, "Pair style lj/cutio/coul/dsf/gpu requires atom attribute q");

  // Only GPU_FORCE is supported because ionization uses the CPU neighbor list
  gpu_mode = GPU_FORCE;

  // Compute cut_coulsq, f_shift, e_shift (same as CPU parent)
  double maxcut = -1.0;
  double cut;
  for (int i = 1; i <= atom->ntypes; i++) {
    for (int j = i; j <= atom->ntypes; j++) {
      if (setflag[i][j] != 0 || (setflag[i][i] != 0 && setflag[j][j] != 0)) {
        cut = init_one(i, j);
        cut *= cut;
        if (cut > maxcut) maxcut = cut;
        cutsq[i][j] = cutsq[j][i] = cut;
      } else
        cutsq[i][j] = cutsq[j][i] = 0.0;
    }
  }
  double cell_size = sqrt(maxcut) + neighbor->skin;

  cut_coulsq = cut_coul * cut_coul;
  double erfcc = erfc(alpha * cut_coul);
  double erfcd = exp(-alpha * alpha * cut_coul * cut_coul);
  f_shift = -(erfcc / cut_coulsq + 2.0 / MY_PIS * alpha * erfcd / cut_coul);
  e_shift = erfcc / cut_coul - f_shift * cut_coul;

  // MPI comm sizes (same as CPU parent)
  comm_forward = 3;
  comm_reverse = 3;
  comm_reverse_off = 3;

  int maxspecial = 0;
  if (atom->molecular != Atom::ATOMIC) maxspecial = atom->maxspecial;
  int mnf = static_cast<int>(5e-2 * neighbor->oneatom);

  int success =
      ljiod_gpu_init(atom->ntypes + 1, cutsq, lj1, lj2, lj3, lj4, offset,
                     force->special_lj, atom->nlocal,
                     atom->nlocal + atom->nghost, mnf, maxspecial, cell_size,
                     gpu_mode, screen, cut_ljsq, cut_coulsq,
                     force->special_coul, force->qqrd2e, e_shift, f_shift,
                     alpha, spair, sself);
  GPU_EXTRA::check_flag(success, error, world);

  // ljiod_gpu_init() overwrites gpu_mode with the device's actual mode.
  // This pair style requires GPU_FORCE because ionization runs on CPU.
  // Use "package gpu N neigh no" in the input to select GPU_FORCE mode.
  if (gpu_mode != GPU_FORCE)
    error->all(FLERR, "Pair lj/cutio/coul/dsf/gpu requires GPU_FORCE mode; "
               "add 'neigh no' to the package gpu command");

  // GPU_FORCE requires a full neighbor list (every pair listed from both sides)
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ---------------------------------------------------------------------- */

double PairLJCutIOCoulDSFGPU::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes + ljiod_gpu_bytes();
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSFGPU::cpu_compute(int start, int inum, int eflag,
                                         int /*vflag*/, int *ilist,
                                         int *numneigh, int **firstneigh)
{
  int i, j, ii, jj, jnum, itype, jtype;
  double qtmp, xtmp, ytmp, ztmp, delx, dely, delz, evdwl, ecoul, fpair;
  double r, rsq, r2inv, r6inv, forcecoul, forcelj, factor_coul, factor_lj;
  double prefactor, erfcc, erfcd, t;
  int *jlist;

  evdwl = ecoul = 0.0;

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  double qqrd2e = force->qqrd2e;

  for (ii = start; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    // DSF self-energy with sself scaling
    if (evflag) {
      double e_self = -(e_shift / 2.0 + alpha / MY_PIS) * qtmp * qtmp * qqrd2e * sself;
      ev_tally(i, i, nlocal, 0, 0.0, e_self, 0.0, 0.0, 0.0, 0.0);
    }

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        } else
          forcelj = 0.0;

        if (rsq < cut_coulsq) {
          r = sqrt(rsq);
          // Pair Coulomb with spair scaling
          prefactor = spair * qqrd2e * qtmp * q[j] / r;
          erfcd = exp(-alpha * alpha * r * r);
          t = 1.0 / (1.0 + EWALD_P * alpha * r);
          erfcc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * erfcd;
          forcecoul = prefactor * (erfcc / r + 2.0 * alpha / MY_PIS * erfcd +
                                   r * f_shift) * r;
          if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;
        } else {
          forcecoul = 0.0;
        }

        fpair = (forcecoul + factor_lj * forcelj) * r2inv;
        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;

        if (eflag) {
          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) -
                    offset[itype][jtype];
            evdwl *= factor_lj;
          } else
            evdwl = 0.0;

          if (rsq < cut_coulsq) {
            ecoul = prefactor * (erfcc - r * e_shift - rsq * f_shift);
            if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
          } else
            ecoul = 0.0;
        }

        if (evflag) ev_tally_full(i, evdwl, ecoul, fpair, delx, dely, delz);
      }
    }
  }
}

/* -*- c++ -*- ----------------------------------------------------------
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
   GPU-accelerated pair style for lj/cutio/coul/dsf (TBSL sonoluminescence).

   Force computation is offloaded to GPU; the ionization pipeline
   (ionization, reset, labelnext + MPI comms) runs on CPU before each
   GPU force call.  Only GPU_FORCE mode is supported because ionization
   requires CPU-side access to the neighbor list.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cutio/coul/dsf/gpu,PairLJCutIOCoulDSFGPU);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUTIO_COUL_DSF_GPU_H
#define LMP_PAIR_LJ_CUTIO_COUL_DSF_GPU_H

#include "pair_lj_cutio_coul_dsf.h"

namespace LAMMPS_NS {

class PairLJCutIOCoulDSFGPU : public PairLJCutIOCoulDSF {
 public:
  PairLJCutIOCoulDSFGPU(LAMMPS *lmp);
  ~PairLJCutIOCoulDSFGPU() override;
  void compute(int, int) override;
  void init_style() override;
  double memory_usage() override;
  void cpu_compute(int, int, int, int, int *, int *, int **);

  enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

 private:
  int gpu_mode;
  double cpu_time;
};

}    // namespace LAMMPS_NS
#endif
#endif

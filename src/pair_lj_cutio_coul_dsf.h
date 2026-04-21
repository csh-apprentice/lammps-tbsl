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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cutio/coul/dsf,PairLJCutIOCoulDSF);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUTIO_COUL_DSF_H
#define LMP_PAIR_LJ_CUTIO_COUL_DSF_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJCutIOCoulDSF : public Pair {
 public:
  PairLJCutIOCoulDSF(class LAMMPS *);
  ~PairLJCutIOCoulDSF() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void *extract(const char *, int &) override;
  void reset(int* label, double* mindist);
  void ionization(double** v, double* q, int *label, double* mindist);
  void labelnext(double**v, double* q, int * label, double* mindist);

  // communication
  
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

  void variable_check();

 protected:
  int pack_flag;
  double cut_lj_global;
  double **cut_lj, **cut_ljsq;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;

  double cut_coul, cut_coulsq;
  double alpha;
  double f_shift, e_shift;

  // Ionization
  int num_io;
  double ***iopo; //ionization potential

  int activevar;
  char *activestr;

  double scalefactor;
  double spair=1.0;
  double sself=1.0;

  double cut_io; // ionization cutoff

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif

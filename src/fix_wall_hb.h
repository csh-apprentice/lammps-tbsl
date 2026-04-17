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

#ifdef FIX_CLASS
// clang-format off
FixStyle(wall/hb,FixWallHeatbath);
// clang-format on
#else

#ifndef LMP_FIX_WALL_HB_H
#define LMP_FIX_WALL_HB_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallHeatbath : public Fix {
 public:
  FixWallHeatbath(class LAMMPS *, int, char **);
  ~FixWallHeatbath() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void min_setup(int) override;
  void post_force(int) override;
  void post_force_respa(int, int, int) override;
  void min_post_force(int) override;
  double compute_scalar() override;
  double compute_vector(int) override;

 private:
  int style;
  double epsilon, sigma, cutoff;
  double alpha;

  double vwall;   // the bubble wall velocity 
  double Tbl;     // the wall temperature
  double mix_coeff;  // weight parameter of heat bath condition and adiabatic condition, 1 refers to hb, 0 refers to adiabatic

  int Tblstyle,Tblvar;
  int varshape;   // 1 if change over time 
  char *Tblstr;
  double scalefactor;   // scale in ensemble particles

  int eflag;
  double ewall[4], ewall_all[4];
  int ilevel_respa;
  char *idregion;
  class Region *region;

  double coeff1, coeff2, coeff3, coeff4, offset;
  double coeff5, coeff6, coeff7;
  double eng, fwall;

  void lj93(double);
  void lj126(double);
  void lj1043(double);
  void morse(double);
  void colloid(double, double);
  void harmonic(double);

  void variable_check();
  void temp_update();
};

}    // namespace LAMMPS_NS

#endif
#endif

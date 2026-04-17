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

#ifdef REGION_CLASS
// clang-format off
RegionStyle(shell,RegShell);
// clang-format on
#else

#ifndef LMP_REGION_SHELL_H
#define LMP_REGION_SHELL_H

#include "region.h"

namespace LAMMPS_NS {

class RegShell : public Region {
 public:
  RegShell(class LAMMPS *, int, char **);
  ~RegShell() override;
  void init() override;
  int inside(double, double, double) override;
  int surface_interior(double *, double) override;
  int surface_exterior(double *, double) override;
  void shape_update() override;
  void set_velocity_shape() override;
  void velocity_contact_shape(double *, double *) override;

 private:
  double xc, yc, zc;

  int xstyle, xvar;
  int ystyle, yvar;
  int zstyle, zvar;
  int rstyle, svar;
  char *xstr, *ystr, *zstr, *sstr;

  double radius_out;  //outer radius of teh shell
  double radius_in;  //innner radius of the shell
  double vradius;  // dR
  double CB; //the velocity of sound in mediin 
  double rho;  // the density of the medium
  double Pinfty;  //the medium pressure
  double sigma; //surface tension
  double  miu;  //dynamics viscosity of liquid
  double cutoff;  //cutoff used as the thickness of the shell
  //auto  compute_press;
  double stresssum; // sum of the stress
  double press;  //current pressure
 
  void variable_check();
};

}    // namespace LAMMPS_NS

#endif
#endif

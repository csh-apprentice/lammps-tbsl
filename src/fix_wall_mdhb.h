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
FixStyle(wall/mdhb,FixWallMDHeatbath);
// clang-format on
#else

#ifndef LMP_FIX_WALL_MDHB_H
#define LMP_FIX_WALL_MDHB_H

#include "fix.h"

namespace LAMMPS_NS {

class FixWallMDHeatbath : public Fix {
 public:
  FixWallMDHeatbath(class LAMMPS *, int, char **);
  ~FixWallMDHeatbath() override;
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
  double Tblgasinit;  // intial gas shell temperature
  double Tbl;     // the wall temperature
  double Tblold;   // the wall temperature of last frame
  double mix_coeff;  // weight parameter of heat bath condition and adiabatic condition, 1 refers to hb, 0 refers to adiabatic


  double temp_shell; // the temperature of the shell
  double vel_change;  // the sum of thevelocity change
  double fix_stress[6]; //stress contribute to fix

  double pair_virial[6]; //pair virial in differnent procs;
  double pair_stress[6]; //stress contribute to pair

  double kspace_virial[6]; // kspace virial in diffrent procs;
  double kspace_stress[6]; // stress contribute to the kspace

  double ke_virial[6]; // kinetic in diffrent procs;
  double ke_stress[6]; //stress contribute to the kinetic

  int sumnumatoms;
  int sumlostparticles;  // sum of the losing particles
  double volume; //volume of the 
  double stress_all;
  double sumenergyloss;  // loss of the particles' energy when hitting the wall
  double gasshellenergy;  // total energy in the gas shell

  int Tblstyle,Tblvar;
  int varshape;   // 1 if change over time 
  char *Tblstr;
  double scalefactor;   // scale in ensemble particles
  double density;  // density of the liquid
  double capacity;  // heat capacity of liquid
  double Tinfty;   // medium temperature
  double liquidenergy;  // energy of the liquid

  int shellpressurevar; 
  char *shellpressurestr;

  int shellkestressvar; 
  char *shellkestressstr;

  bool kspaceflag;  // whether count in kstress

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
  void Tbl_update(double R, double oldR, double delta, double olddelta); 
  double cal_liquid_energy(double R,double delta);  //R is the bubble radius
};

}    // namespace LAMMPS_NS

#endif
#endif
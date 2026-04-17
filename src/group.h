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

#ifndef LMP_GROUP_H
#define LMP_GROUP_H

#include "pointers.h"

#include <map>

namespace LAMMPS_NS {
class Region;

class Group : protected Pointers {
 public:
  int ngroup;          // # of defined groups
  char **names;        // name of each group
  int *bitmask;        // one-bit mask for each group
  int *inversemask;    // inverse mask for each group
  int *dynamic;        // 1 if dynamic, 0 if not

  Group(class LAMMPS *);
  ~Group() override;
  void assign(int, char **);                  // assign atoms to a group
  void assign(const std::string &);           // convenience function
  void create(const std::string &, int *);    // add flagged atoms to a group
  int find(const std::string &);              // lookup name in list of groups
  int find_or_create(const char *);           // lookup name or create new group
  void write_restart(FILE *);
  void read_restart(FILE *);

  bigint count_all();             // count atoms in group all
  bigint count(int);              // count atoms in group
  bigint count(int, Region *);    // count atoms in group & region
  double mass(int);               // total mass of atoms in group
  double mass(int, Region *);
  double charge(int);    // total charge of atoms in group
  double charge(int, Region *);
  void bounds(int, double *);    // bounds of atoms in group
  void bounds(int, double *, Region *);
  void xcm(int, double, double *);    // center-of-mass coords of group
  void xcm(int, double, double *, Region *);
  void vcm(int, double, double *);    // center-of-mass velocity of group
  void vcm(int, double, double *, Region *);
  void fcm(int, double *);    // total force on group
  void fcm(int, double *, Region *);
  double ke(int);    // kinetic energy of group
  double ke(int, Region *);
  double gyration(int, double, double *);    // radius-of-gyration of group
  double gyration(int, double, double *, Region *);
  void angmom(int, double *, double *);    // angular momentum of group
  void angmom(int, double *, double *, Region *);
  void torque(int, double *, double *);    // torque on group
  void torque(int, double *, double *, Region *);
  void inertia(int, double *, double[3][3]);    // inertia tensor
  void inertia(int, double *, double[3][3], Region *);
  void omega(double *, double[3][3], double *);    // angular velocity

  // code for SL
  double SL_radius(int, Region *);   //radius of the bubble
  double SL_radius_old(int, Region *);   //radius of the bubble from previous step
  double SL_vradius(int, Region *);  // vradius of the bubble
  double SL_count(int, Region *);   // number of atoms in the gas shell
  double SL_pB(int, Region *);  // pressure of gas shell from outside
  double SL_pB_old(int, Region *);  // pressure of gas shell from outside from last step
  double SL_pb(int, Region *);  // pressure of gas shell from inside
  double SL_gastemp(int, Region *);  // temperature of gas shell
  double SL_walltemp(int, Region *);  // temperature of liquid wall
  double SL_delta(int, Region *);  // thickness of the liquid shell
  double SL_delta_old(int, Region *);  // thickness of the liquid shell from last step
  double SL_vdelta(int, Region *);  // velocity of thickness of the liquid shell
  double SL_numatoms(int, Region *);  // number of atoms in the gas shell
  double SL_debug(int, Region *); // print out debug variable
  double SL_steps(int, Region *); // accumulate steps in SL simulation
  double TH_radius(int, Region *);  // Radius in Theory Calcultion
  double TH_vradius(int, Region *);  // Velocity in Theory Calcultion
  double TH_aradius(int, Region *);  // Accerleration in Theory Calcultion
  double TH_Tb0(int, Region *); //Temperature of Gas shell in theory
  double TH_delta(int, Region *); //thickness of Gas shell in theory

  int SL_lost(int, Region *); // print out lost particles
 private:
  int me;
  std::map<tagint, int> *hash;

  int find_unused();
  void add_molecules(int, int);

  // callback functions for ring communication

  static void molring(int, char *, void *);
  int molbit;
};

}    // namespace LAMMPS_NS

#endif

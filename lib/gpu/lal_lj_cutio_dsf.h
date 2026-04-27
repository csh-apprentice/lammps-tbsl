/***************************************************************************
                            lj_cutio_dsf.h
                             -------------------
              GPU accelerator class for lj/cutio/coul/dsf pair style.

  Based on lal_lj_dsf.h (W. Michael Brown, ORNL).
  Extended for TBSL: separate qqrd2e_self parameter for DSF self-energy
  to correctly apply the sself ensemble scaling factor.

 __________________________________________________________________________
    This file is part of the LAMMPS Accelerator Library (LAMMPS_AL)
 ***************************************************************************/

#ifndef LAL_LJ_CUTIO_DSF_H
#define LAL_LJ_CUTIO_DSF_H

#include "lal_base_charge.h"

namespace LAMMPS_AL {

template <class numtyp, class acctyp>
class LJCutIODSF : public BaseCharge<numtyp, acctyp> {
 public:
  LJCutIODSF();
  ~LJCutIODSF();

  /// Clear any previous data and set up for a new LAMMPS run
  /** \param max_nbors initial number of rows in the neighbor matrix
    * \param cell_size cutoff + skin
    * \param gpu_split fraction of particles handled by device
    * \param spair   ensemble scale factor for Coulomb pair interactions
    * \param sself   ensemble scale factor for DSF self-energy
    *
    * Returns:
    * -  0 if successful
    * - -1 if fix gpu not found
    * - -3 if there is an out of memory error
    * - -4 if the GPU library was not compiled for GPU
    * - -5 Double precision is not supported on card **/
  int init(const int ntypes, double **host_cutsq, double **host_lj1,
           double **host_lj2, double **host_lj3, double **host_lj4,
           double **host_offset, double *host_special_lj,
           const int nlocal, const int nall, const int max_nbors,
           const int maxspecial, const double cell_size,
           const double gpu_split, FILE *screen, double **host_cut_ljsq,
           const double host_cut_coulsq, double *host_special_coul,
           const double qqrd2e, const double e_shift, const double f_shift,
           const double alpha, const double spair, const double sself);

  /// Clear all host and device data
  void clear();

  /// Returns memory usage on device per atom
  int bytes_per_atom(const int max_nbors) const;

  /// Total host memory used by library for pair style
  double host_memory_usage() const;

  // --------------------------- TYPE DATA --------------------------

  /// lj1.x = lj1, lj1.y = lj2, lj1.z = cutsq_vdw, lj1.w = cutsq
  UCL_D_Vec<numtyp4> lj1;
  /// lj3.x = lj3, lj3.y = lj4, lj3.z = offset
  UCL_D_Vec<numtyp4> lj3;
  /// Special LJ values [0-3] and Special Coul values [4-7]
  UCL_D_Vec<numtyp> sp_lj;

  /// If atom type constants fit in shared memory, use fast kernels
  bool shared_types;

  /// Number of atom types
  int _lj_types;

  numtyp _qqrd2e;       // = spair * qqrd2e_base (pair interactions)
  numtyp _qqrd2e_self;  // = sself * qqrd2e_base (DSF self-energy)

 private:
  bool _allocated;
  numtyp _e_shift, _f_shift, _alpha, _cut_coulsq;
  int loop(const int eflag, const int vflag);
};

}

#endif

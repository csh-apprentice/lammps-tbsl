// clang-format off
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
   Contributing author: Trung Dac Nguyen (ORNL)
   References: Fennell and Gezelter, JCP 124, 234104 (2006)
------------------------------------------------------------------------- */

#include "pair_lj_cutio_coul_dsf.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "timer.h"
#include "input.h"
#include "variable.h"
#include <cmath>
#include <cstring>
#include <limits>

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

#define BIG 1e20

namespace {
// With units real, masses are g/mol and velocities are Angstrom/fs. The
// ionization thresholds supplied by TBSL inputs are kJ/mol (not kcal/mol).
double collision_energy(double mi, double mj, const double *vi, const double *vj)
{
  double relative_speed_sq = 0.0;
  for (int d = 0; d < 3; ++d) {
    const double du = vi[d] - vj[d];
    relative_speed_sq += du * du;
  }
  const double reduced_mass = mi * mj / (mi + mj);
  return 0.5e4 * reduced_mass * relative_speed_sq;
}

// Exact distance/tag ordering makes the MPI minimum reduction independent of
// visitation order, including exact distance ties.
bool nearer(double distance, int tag, double best_distance, int best_tag)
{
  return distance < best_distance ||
    (distance == best_distance && (best_tag < 0 || tag < best_tag));
}
}


/* ---------------------------------------------------------------------- */

PairLJCutIOCoulDSF::PairLJCutIOCoulDSF(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  pack_flag = 0;


}

/* ---------------------------------------------------------------------- */

PairLJCutIOCoulDSF::~PairLJCutIOCoulDSF()
{
  if (copymode) return;

  delete[] activestr;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut_lj);
    memory->destroy(cut_ljsq);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

    //EXTENSION FOR IONIZATION
    memory->destroy(iopo);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double r,rsq,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double prefactor,erfcc,erfcd,t;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = ecoul = 0.0;
  ev_init(eflag,vflag);

  //utils::logmesg(lmp,"DEBUG 1 \n");

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e; 
  double active= input->variable->compute_equal(activevar);
  if(abs(active)<1e-4) return;
  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  ionization_pipeline();
  timer->stamp(Timer::IONIZATION);
  for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      qtmp = q[i];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];
    //potential and force
    if (eflag) {
      double e_self = -(e_shift/2.0 + alpha/MY_PIS) * qtmp*qtmp*qqrd2e * sself;
      ev_tally(i,i,nlocal,0,0.0,e_self,0.0,0.0,0.0,0.0);
    }
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;

        if (rsq < cut_ljsq[itype][jtype]) {
          r6inv = r2inv*r2inv*r2inv;
          forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        } else forcelj = 0.0;

        if (rsq < cut_coulsq) {
          r = sqrt(rsq);
          prefactor = spair * qqrd2e*qtmp*q[j]/r;
          erfcd = exp(-alpha*alpha*r*r);
          t = 1.0 / (1.0 + EWALD_P*alpha*r);
          erfcc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * erfcd;
          forcecoul = prefactor * (erfcc/r + 2.0*alpha/MY_PIS * erfcd +
                                   r*f_shift) * r;
          if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
        } else forcecoul = 0.0;

        fpair = (forcecoul + factor_lj*forcelj) * r2inv;
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          if (rsq < cut_ljsq[itype][jtype]) {
            evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
                    offset[itype][jtype];
            evdwl *= factor_lj;
          } else evdwl = 0.0;

          if (rsq < cut_coulsq) {
            ecoul = prefactor * (erfcc - r*e_shift - rsq*f_shift);
            if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
          } else ecoul = 0.0;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,ecoul,fpair,delx,dely,delz);
      }
      //  utils::logmesg(lmp,"COMPUTE: step {} qi {} qj {}  tagi {} tagj {}  \n",
      //             update->ntimestep,q[i],q[j],atom->tag[i],atom->tag[j]);
    } // second j loop
  } //second i loop
  
  // pack_flag = 4;
  // comm->reverse_comm(this);

  
    //MPI_Barrier(MPI_COMM_WORLD);
    //utils::logmesg(lmp,"Instep {}, Labeli {} is {}, mindisti is {}, Labelj {} is {}, mindistj is {} \n",update->ntimestep,i_id,label[i],mindist[i],j_id,label[j],mindist[j]); 
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut_lj,n+1,n+1,"pair:cut_lj");
  memory->create(cut_ljsq,n+1,n+1,"pair:cut_ljsq");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");

  //extension for ionization
  memory->create(iopo,8,n+1,n+1,"pair:iopo");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::settings(int narg, char **arg)
{
  if (narg < 2 || narg > 4) error->all(FLERR,"Illegal pair_style command");

  alpha = utils::numeric(FLERR,arg[0],false,lmp);
  cut_lj_global = utils::numeric(FLERR,arg[1],false,lmp);
  if (narg == 2) cut_coul = cut_lj_global;
  else cut_coul = utils::numeric(FLERR,arg[2],false,lmp);
  if (narg==2 ) cut_io=cut_lj_global;
  else cut_io = utils::numeric(FLERR,arg[3],false,lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j])
          cut_lj[i][j] = cut_lj_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::coeff(int narg, char **arg)
{
  if (narg < 6)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  //right format should be pair_coeff type1 type2 num_io io1, io2, ..., io_{num} epsilon sigma (cutoff)
  num_io=utils::inumeric(FLERR, arg[2], false, lmp);
  if (num_io < 1 || num_io > 8 || (narg != num_io+7 && narg != num_io+8))
    error->all(FLERR,"Incorrect ionization ladder or args for pair coefficients");

  //extension for ionization
  double iopo_eight[8];
  for(int i=0;i<num_io;i++) {
    iopo_eight[i]=utils::numeric(FLERR, arg[3+i], false, lmp);
    if (!std::isfinite(iopo_eight[i]) || iopo_eight[i] <= 0.0)
      error->all(FLERR,"Ionization thresholds must be positive and finite (kJ/mol)");
  }

  double epsilon_one = utils::numeric(FLERR,arg[3+num_io],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[4+num_io],false,lmp);
  delete[] activestr;
  activestr=utils::strdup(arg[5+num_io]+2);
  double cut_lj_one = cut_lj_global;
  scalefactor= utils::numeric(FLERR,arg[6+num_io],false,lmp);
    // Set Coulomb scaling factors (g = scalefactor)
  spair = scalefactor;                 // keeps pair energy/forces consistent
  sself = std::pow(scalefactor, 4.0/3.0);  // keeps DSF self-energy consistent

  if (narg == 8+num_io) cut_lj_one = utils::numeric(FLERR,arg[7+num_io],false,lmp);
  //utils::logmesg(lmp,"DEBUG: 1 \n");
         

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut_lj[i][j] = cut_lj_one;
      //extension for ionization
      for(int k=0;k<num_io;k++)
        iopo[k][i][j]=iopo_eight[k];
      setflag[i][j] = 1;
      count++;
    }
  }
  //utils::logmesg(lmp,"DEBUG: 2 \n");
  variable_check();
  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::init_style()
{
  init_ionization();

  neighbor->add_request(this);

  cut_coulsq = cut_coul * cut_coul;
  double erfcc = erfc(alpha*cut_coul);
  double erfcd = exp(-alpha*alpha*cut_coul*cut_coul);
  f_shift = -(erfcc/cut_coulsq + 2.0/MY_PIS*alpha*erfcd/cut_coul);
  e_shift = erfcc/cut_coul - f_shift*cut_coul;

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutIOCoulDSF::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut_lj[i][j] = mix_distance(cut_lj[i][i],cut_lj[j][j]);
  }

  double cut = MAX(cut_lj[i][j], MAX(cut_coul,cut_io));
  cut_ljsq[i][j] = cut_lj[i][j] * cut_lj[i][j];

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);

  if (offset_flag && (cut_lj[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut_lj[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  cut_ljsq[j][i] = cut_ljsq[i][j];
  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut_lj[i][j]*cut_lj[i][j]*cut_lj[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
               sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
               sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }

  return cut;
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut_lj[i][j],sizeof(double),1,fp);
        //extension for ionization
        for (int k=0;k<num_io;k++)
          fwrite(&iopo[k][i][j], sizeof(double), 1, fp);
        }
    }
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&epsilon[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut_lj[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut_lj[i][j],1,MPI_DOUBLE,0,world);

        for(int k=0;k<num_io;k++)
        {
         if (me == 0) {
          utils::sfread(FLERR, &iopo[k][i][j], sizeof(double), 1, fp, nullptr, error);
         }
           MPI_Bcast(&iopo[k][i][j], 1, MPI_DOUBLE, 0, world);
        }
      }
    }
}

/* ----------------------------------------------------------------------
  proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::write_restart_settings(FILE *fp)
{
  fwrite(&alpha,sizeof(double),1,fp);
  fwrite(&cut_lj_global,sizeof(double),1,fp);
  fwrite(&cut_coul,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);

  fwrite(&num_io,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
  proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&alpha,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_lj_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_coul,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&tail_flag,sizeof(int),1,fp,nullptr,error);

    utils::sfread(FLERR, &num_io, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&alpha,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_lj_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_coul,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);

  MPI_Bcast(&num_io, 1, MPI_INT, 0, world);
}

/* ---------------------------------------------------------------------- */

double PairLJCutIOCoulDSF::single(int i, int j, int itype, int jtype, double rsq,
                                double factor_coul, double factor_lj,
                                double &fforce)
{
  double r2inv,r6inv,r,erfcc,erfcd,prefactor;
  double forcecoul,forcelj,phicoul,philj;

  r2inv = 1.0/rsq;
  if (rsq < cut_ljsq[itype][jtype]) {
    r6inv = r2inv*r2inv*r2inv;
    forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
  } else forcelj = 0.0;

  if (rsq < cut_coulsq) {
    r = sqrt(rsq);
    prefactor = spair* factor_coul * force->qqrd2e * atom->q[i]*atom->q[j]/r;
    erfcc = erfc(alpha*r);
    erfcd = exp(-alpha*alpha*r*r);
    forcecoul = prefactor * (erfcc/r + 2.0*alpha/MY_PIS * erfcd +
      r*f_shift) * r;
  } else forcecoul = 0.0;

  fforce = (forcecoul + factor_lj*forcelj) * r2inv;

  double eng = 0.0;
  if (rsq < cut_ljsq[itype][jtype]) {
    philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
      offset[itype][jtype];
    eng += factor_lj*philj;
  }

  if (rsq < cut_coulsq) {
    phicoul = prefactor * (erfcc - r*e_shift - rsq*f_shift);
    eng += phicoul;
  }

  return eng;
}

/* ---------------------------------------------------------------------- */

void *PairLJCutIOCoulDSF::extract(const char *str, int &dim)
{
  if (strcmp(str,"cut_coul") == 0) {
    dim = 0;
    return (void *) &cut_coul;
  }
  return nullptr;
}


void PairLJCutIOCoulDSF::init_ionization()
{
  if (!atom->q_flag || !atom->rmass_flag || atom->map_style == Atom::MAP_NONE)
    error->all(FLERR,"Ionization requires q, per-atom mass, and an atom map");
  if (strcmp(update->unit_style,"real") != 0)
    error->all(FLERR,"Pair lj/cutio/coul/dsf requires units real and thresholds in kJ/mol");

  int flag, cols;
  label_index = atom->find_custom("label", flag, cols);
  if (label_index < 0 || flag != 0 || cols != 0)
    error->all(FLERR,"Ionization requires a custom integer vector i_label");
  mindist_index = atom->find_custom("mindist", flag, cols);
  if (mindist_index < 0 || flag != 1 || cols != 0)
    error->all(FLERR,"Ionization requires a custom double vector d_mindist");

  comm_forward = 6;       // q, vx, vy, vz, mindist, label
  comm_reverse = 2;       // nearest-candidate distance and tag
  comm_reverse_off = 2;   // same reduction also works with Newton pair off
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::ionization_pipeline()
{
  auto v = atom->v;
  auto q = atom->q;
  auto label = atom->ivector[label_index];
  auto mindist = atom->dvector[mindist_index];

  // Both owners of a cross-rank pair evaluate the same pre-event snapshot.
  // Each writes only owned state; ghosts remain immutable during this phase.
  pack_flag = ION_STATE;
  comm->forward_comm(this);
  ionization(v,q,label,mindist);
  comm->forward_comm(this);

  reset(label,mindist);
  labelnext(v,q,label,mindist);

  // Neighbors may nominate candidates for ghosts, including with half lists.
  // Reduce to owners FIRST, then broadcast their final choices. This minimum
  // reduction is idempotent for Newton-off and GPU full lists as well.
  pack_flag = ION_CANDIDATES;
  comm->reverse_comm(this);
  comm->forward_comm(this);
}

/* ---------------------------------------------------------------------- */

double PairLJCutIOCoulDSF::next_ionization_energy(int i, int itype, int jtype)
{
  const double charge = atom->q[i];
  if (!std::isfinite(charge) || charge < 0.0 || charge > num_io ||
      charge != std::floor(charge))
    error->one(FLERR,"Ionization charge must be an integer between 0 and num_io");
  const int qi = static_cast<int>(charge);
  if (qi == num_io) return std::numeric_limits<double>::infinity();
  // Pair coefficients are stored in the upper triangle.
  return iopo[qi][MIN(itype,jtype)][MAX(itype,jtype)];
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::reset(int *label, double *mindist)
{
  // Clear stale labels even when their former partner has left the halo.
  for (int i = 0; i < atom->nlocal + atom->nghost; ++i) {
    label[i] = -1;
    mindist[i] = BIG;
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::ionization(double **v, double *q, int *label, double *mindist)
{
  auto x = atom->x;
  auto m = atom->rmass;
  auto tag = atom->tag;
  auto type = atom->type;
  const int nlocal = atom->nlocal;
  const double cut_iosq = cut_io * cut_io;

  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    if (label[i] <= 0) continue;
    const int j = atom->map(label[i]);
    // Labels refer to the previous timestep. Losing a partner from the ghost
    // halo is normal and cannot be treated as an invalid array access.
    if (j < 0 || j >= nlocal + atom->nghost || tag[i] == tag[j] ||
        label[j] != tag[i]) continue;

    const int itype = type[i], jtype = type[j];
    double dx = x[i][0] - x[j][0];
    double dy = x[i][1] - x[j][1];
    double dz = x[i][2] - x[j][2];
    domain->minimum_image(dx,dy,dz);
    const double rsq = dx*dx + dy*dy + dz*dz;
    if (rsq >= cutsq[itype][jtype] || rsq >= cut_iosq) continue;

    const double chi_i = next_ionization_energy(i,itype,jtype);
    const double chi_j = next_ionization_energy(j,itype,jtype);
    const int target = (chi_i < chi_j || (chi_i == chi_j && tag[i] < tag[j])) ? i : j;
    const double chi = (target == i) ? chi_i : chi_j;
    const double energy = collision_energy(m[i],m[j],v[i],v[j]);
    if (energy <= chi) continue; // Strict threshold; also handles zero relative speed.

    // Ruuth et al., PRE 66, 036310 (2002), Sec. II.G, pp. 6-7:
    // subtract chi, then retain 2/3 of the remaining COM-frame kinetic energy.
    const double scale = sqrt((2.0/3.0) * (1.0 - chi/energy));
    double vi[3], vj[3];
    const double mass_sum = m[i] + m[j];
    for (int d = 0; d < 3; ++d) {
      const double com = (m[i]/mass_sum)*v[i][d] + (m[j]/mass_sum)*v[j][d];
      vi[d] = com + scale * (v[i][d] - com);
      vj[d] = com + scale * (v[j][d] - com);
    }
    for (int d = 0; d < 3; ++d) v[i][d] = vi[d];
    if (target < nlocal) q[target] += 1.0;
    label[i] = -1;
    mindist[i] = BIG;
    if (j < nlocal) {
      for (int d = 0; d < 3; ++d) v[j][d] = vj[d];
      label[j] = -1;
      mindist[j] = BIG;
    }
    // For a ghost j, its owner performs j's update from the same snapshot.
    // Never write a ghost velocity or charge that might not reach its owner.
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::labelnext(double **v, double *q, int *label, double *mindist)
{
  auto x = atom->x;
  auto m = atom->rmass;
  auto type = atom->type;
  auto tag = atom->tag;
  const double cut_iosq = cut_io * cut_io;

  for (int ii = 0; ii < list->inum; ++ii) {
    const int i = list->ilist[ii];
    for (int jj = 0; jj < list->numneigh[i]; ++jj) {
      const int j = list->firstneigh[i][jj] & NEIGHMASK;
      if (tag[i] == tag[j]) continue;
      const double dx = x[i][0] - x[j][0];
      const double dy = x[i][1] - x[j][1];
      const double dz = x[i][2] - x[j][2];
      const double rsq = dx*dx + dy*dy + dz*dz;
      const int itype = type[i], jtype = type[j];
      if (rsq >= cutsq[itype][jtype] || rsq >= cut_iosq) continue;

      const double chi_i = next_ionization_energy(i,itype,jtype);
      const double chi_j = next_ionization_energy(j,itype,jtype);
      const double energy = collision_energy(m[i],m[j],v[i],v[j]);
      if (energy <= MIN(chi_i,chi_j)) continue;
      if (nearer(rsq,tag[j],mindist[i],label[i])) {
        label[i] = tag[j];
        mindist[i] = rsq;
      }
      if (nearer(rsq,tag[i],mindist[j],label[j])) {
        label[j] = tag[i];
        mindist[j] = rsq;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int PairLJCutIOCoulDSF::pack_forward_comm(int n, int *list, double *buf, int, int *)
{
  int k = 0;
  for (int ii = 0; ii < n; ++ii) {
    const int i = list[ii];
    if (pack_flag == ION_STATE) {
      buf[k++] = atom->q[i];
      for (int d = 0; d < 3; ++d) buf[k++] = atom->v[i][d];
    }
    buf[k++] = atom->dvector[mindist_index][i];
    buf[k++] = ubuf(atom->ivector[label_index][i]).d;
  }
  return k;
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::unpack_forward_comm(int n, int first, double *buf)
{
  int k = 0;
  for (int i = first; i < first+n; ++i) {
    if (pack_flag == ION_STATE) {
      atom->q[i] = buf[k++];
      for (int d = 0; d < 3; ++d) atom->v[i][d] = buf[k++];
    }
    atom->dvector[mindist_index][i] = buf[k++];
    atom->ivector[label_index][i] = ubuf(buf[k++]).i;
  }
}

/* ---------------------------------------------------------------------- */

int PairLJCutIOCoulDSF::pack_reverse_comm(int n, int first, double *buf)
{
  int k = 0;
  for (int i = first; i < first+n; ++i) {
    buf[k++] = atom->dvector[mindist_index][i];
    buf[k++] = ubuf(atom->ivector[label_index][i]).d;
  }
  return k;
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::unpack_reverse_comm(int n, int *list, double *buf)
{
  for (int ii = 0; ii < n; ++ii) {
    const int i = list[ii];
    const double distance = buf[2*ii];
    const int tag = ubuf(buf[2*ii+1]).i;
    auto mindist = atom->dvector[mindist_index];
    auto label = atom->ivector[label_index];
    if (tag > 0 && nearer(distance,tag,mindist[i],label[i])) {
      mindist[i] = distance;
      label[i] = tag;
    }
  }
}

void PairLJCutIOCoulDSF::variable_check()
{

  activevar=input->variable->find(activestr);
    if (activevar < 0) 
      error->all(FLERR, "Variable {} forpair lj cutio coul dsf does not exist", activestr);
    if (!input->variable->equalstyle(activevar))
      error->all(FLERR, "Variable {} for pair lj cutio coul dsf is invalid style", activestr);
    
}

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

using namespace LAMMPS_NS;
using namespace MathConst;

#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429

#define EPSILON 1e-12
#define MAXENERGY 1e12
#define BIG 1e20



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
  double **v=atom->v;
  double *m=atom->rmass;
  int *label=atom->ivector[1];
  double *mindist =atom->dvector[0];
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
  
  //utils::logmesg(lmp,"PAIR LJ CUT IO COUL DSFVIRIAL, eflag_either {}, eflag_atom {}, vflag_either {}, vflag_global {}, vflag_atom {}, newton_pair {}  \n",eflag_either,eflag_atom,vflag_either,vflag_global,vflag_atom,newton_pair) ;
  //extension for ionization
  // pack_flag = 4;
  // comm->forward_comm(this);
  // loop over neighbors of my atoms
  //if(update->ntimestep==168)
  //before started, print out all atom information
  
  ionization(v,q,label,mindist);   // change the q
  
  pack_flag = 1;
  comm->forward_comm(this);
  if(newton_pair)
  {
    pack_flag=1;
    comm->reverse_comm(this);
  }


  reset(label,mindist);   // change the label and mindist

  pack_flag = 2;
  comm->forward_comm(this);
  if(newton_pair)
  {
    pack_flag=2;
    comm->reverse_comm(this);
  }

  labelnext(v,q,label,mindist);  // change the label and mindist

  pack_flag = 3;
  comm->forward_comm(this);
  if(newton_pair)
  {
    pack_flag=3;
    comm->reverse_comm(this);
  }
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
  num_io=utils::numeric(FLERR, arg[2], false, lmp);

  //extension for ionization
  double iopo_eight[8];
  for(int i=0;i<num_io;i++)
    iopo_eight[i]=utils::numeric(FLERR, arg[3+i], false, lmp);

  double epsilon_one = utils::numeric(FLERR,arg[3+num_io],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[4+num_io],false,lmp);
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
  if (!atom->q_flag)
    error->all(FLERR,"Pair style lj/cutio/coul/dsf requires atom attribute q");

  neighbor->add_request(this);

  cut_coulsq = cut_coul * cut_coul;
  double erfcc = erfc(alpha*cut_coul);
  double erfcd = exp(-alpha*alpha*cut_coul*cut_coul);
  f_shift = -(erfcc/cut_coulsq + 2.0/MY_PIS*alpha*erfcd/cut_coul);
  e_shift = erfcc/cut_coul - f_shift*cut_coul;

  comm_forward = 3;
  comm_reverse=3;
  comm_reverse_off=3;
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


void PairLJCutIOCoulDSF::reset(int *label, double *mindist)
{
   int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double r,rsq,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double prefactor,erfcc,erfcd,t;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = ecoul = 0.0;
  //ev_init(eflag,vflag);

  //utils::logmesg(lmp,"DEBUG 1 \n");

  double **x = atom->x;
  double **f = atom->f;
  double *q = atom->q;
  double **v=atom->v;
  double *m=atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e; 

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      r=sqrt(rsq);
      jtype = type[j];
      
      //label[j]=-1; mindist[j]=BIG;
      if (rsq<cutsq[itype][jtype] && r < cut_io) {
        label[i]=-1; mindist[i]=BIG;
        label[j]=-1; mindist[j]=BIG;
      }
    }  //second j loop
  }


}

void PairLJCutIOCoulDSF::ionization(double** v, double* q, int *label, double* mindist)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double r,rsq,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double prefactor,erfcc,erfcd,t;
  int *ilist,*jlist,*numneigh,**firstneigh;
  //utils::logmesg(lmp,"DEBUG 1 \n");

  double **x = atom->x;
  double *m=atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e; 


  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    qtmp = q[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    int flag=0;  // if ionization happens
    int jlocal = -1;
    if (label[i] > 0) {
      jlocal = atom->map(label[i]);
      int nall = atom->nlocal + atom->nghost;
      if (jlocal < 0 || jlocal >= nall) {
        utils::logmesg(lmp,
          "[ION] step {}: invalid partner mapping: "
          "i_local={} tag_i={} label[i]={} jlocal={} nall={}\n",
          update->ntimestep,
          i, (atom->tag)[i], label[i], jlocal, nall);
        error->one(FLERR, "Ionization: atom->map(label[i]) out of range");
      }
    }


    // //before the loop start
    // //Label Match checking
    //for (jj = 0; jj < jnum; jj++) {
    if(jlocal>0){
      j = jlocal;
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      r=sqrt(rsq);
      
      //int llabel=label[i];
      // if(rsq <cutsq[itype][jtype])
      // utils::logmesg(lmp,"In step {}, i_id is {}, j_id is {} \n",update->ntimestep,i_id,j_id);

      if(rsq <cutsq[itype][jtype]&& r<cut_io && label[i]==(atom->tag)[j] && label[j]==(atom->tag)[i]) //label match
      {
        // recheck energy
        //double check the energy avoid shifting from last step
        double kinenergy_i=1e4*1.0/2.0*m[i]*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2]);
        double kinenergy_j=1e4*1.0/2.0*m[j]*(v[j][0]*v[j][0]+v[j][1]*v[j][1]+v[j][2]*v[j][2]);
        double kinenergy_sum=kinenergy_i+kinenergy_j;

        int i_res = num_io - int(q[i]);
        int j_res = num_io - int(q[j]);

        // --- Crash-time debug guard for q-index ---
        int qi = static_cast<int>(q[i]);
        int qj = static_cast<int>(q[j]);

        if (qi < 0 || qi >= num_io || qj < 0 || qj >= num_io) {
          utils::logmesg(lmp,
            "[ION] step {}: invalid q index in ionization: "
            "i_local={} tag_i={} qi={} j_local={} tag_j={} qj={} num_io={}\n",
            update->ntimestep,
            i, (atom->tag)[i], qi,
            j, (atom->tag)[j], qj, num_io);

          error->one(FLERR, "Ionization: q index out of range (iopo access)");
        }

        // safe access now
        double i_ionext = (i_res != 0) ? iopo[qi][itype][jtype] : MAXENERGY;
        double j_ionext = (j_res != 0) ? iopo[qj][itype][jtype] : MAXENERGY;


        int i_id=(atom->tag)[i];  int j_id=(atom->tag)[j];
        double ivscale=1.0; double jvscale=1.0;
        //double iqtmp=q[i]; double jqtmp=q[j];

        if((i_res>0&&kinenergy_sum>i_ionext)||(j_res>0&&kinenergy_sum>j_ionext))
        {
          flag=1;  //ionization happens
          int tempi=i;
          // utils::logmesg(lmp,"IODEBUG INITAL: step {} qi {} qj {} i_io {} j_io {} tagi {} tagj {} ionized id {} energy {} ifsame {}\n",
          //         update->ntimestep,q[i],q[j],i_ionext,j_ionext,i_id,j_id,(atom->tag)[tempi],kinenergy_sum,j<nlocal);
          //define cusomized symmetric ionization order here
          if (fabs(i_ionext-j_ionext)<EPSILON)
            //if have the same ionization energy, always pick the atom with a smaller global id
            tempi=(i_id<j_id)?i:j;
          else if (i_ionext>j_ionext)
            tempi=j;

          double io_next=(tempi==i)?i_ionext:j_ionext;

          //loop i
          double kinenergy_res=2.0/3.0*(kinenergy_i-io_next*kinenergy_i/kinenergy_sum);
          //kinenergy_res=(kinenergy_i-io_next*kinenergy_i/kinenergy_sum);
          double v_scale=sqrt(kinenergy_res/kinenergy_i);
          ivscale*=v_scale;
          kinenergy_i=kinenergy_res;

          //loop j
          kinenergy_res=2.0/3.0*(kinenergy_j-io_next*kinenergy_j/kinenergy_sum);
          //kinenergy_res=(kinenergy_j-io_next*kinenergy_j/kinenergy_sum);
          v_scale=sqrt(kinenergy_res/kinenergy_j);
          jvscale*=v_scale;
          kinenergy_j=kinenergy_res;

          //increament q
          //q[tempi]++;
          if(tempi==i) q[i]=q[i]+1.0;
          else q[j]=q[j]+1.0;


          i_res=num_io-int(q[i]);
          j_res=num_io-int(q[j]);
          
          // --- Crash-time guard again after q changed ---
          qi = static_cast<int>(q[i]);
          qj = static_cast<int>(q[j]);

          if (qi < 0 || qi >= num_io || qj < 0 || qj >= num_io) {
            utils::logmesg(lmp,
              "[ION] step {}: invalid q index AFTER ionization update: "
              "i_local={} tag_i={} qi={} j_local={} tag_j={} qj={} num_io={}\n",
              update->ntimestep,
              i, (atom->tag)[i], qi,
              j, (atom->tag)[j], qj, num_io);

            error->one(FLERR, "Ionization: q index out of range after update");
          }

          //update the io energy
          i_ionext=(i_res!=0)?iopo[int(q[i])][itype][jtype]:MAXENERGY;
          j_ionext=(j_res!=0)?iopo[int(q[j])][itype][jtype]:MAXENERGY;

          v[i][0]*=ivscale; v[i][1]*=ivscale; v[i][2]*=ivscale; 

          v[j][0]*=jvscale; v[j][1]*=jvscale; v[j][2]*=jvscale;

          label[i]=-1; label[j]=-1;  // avoid mutilple ionization in one round
          mindist[i]=BIG; mindist[j]=BIG;

          //kinenergy_sum=kinenergy_i+kinenergy_j;
          
          // utils::logmesg(lmp,"IODEBUG COMPUTE: step {} qi {} qj {} i_io {} j_io {} tagi {} tagj {} ionized id {} energy {} cutoff {}\n",
          //         update->ntimestep,q[i],q[j],i_ionext,j_ionext,i_id,j_id,(atom->tag)[tempi],kinenergy_sum,cutsq[itype][jtype]);
        }
        
        //break;
      }

    }  //first j loop
  } //first i loop
}

void PairLJCutIOCoulDSF::labelnext(double**v, double* q, int * label, double* mindist)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double qtmp,xtmp,ytmp,ztmp,delx,dely,delz,evdwl,ecoul,fpair;
  double r,rsq,r2inv,r6inv,forcecoul,forcelj,factor_coul,factor_lj;
  double prefactor,erfcc,erfcd,t;
  int *ilist,*jlist,*numneigh,**firstneigh;

  
  //utils::logmesg(lmp,"DEBUG 1 \n");

  double **x = atom->x;
  double *m=atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  double *special_coul = force->special_coul;
  int newton_pair = force->newton_pair;
  double qqrd2e = force->qqrd2e; 

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  
  for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      qtmp = q[i];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];
    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      r=sqrt(rsq);
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        
        if(r<cut_io)
        {
            // Extension for ionization
          double kinenergy_i=1e4*1.0/2.0*m[i]*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2]);
          int qi = static_cast<int>(q[i]);
          int i_res = num_io - qi;

          // Crash-time guard for q[i]
          if (qi < 0 || qi >= num_io) {
            utils::logmesg(lmp,
              "[LABELNEXT] step {}: invalid qi index: "
              "i_local={} tag_i={} qi={} num_io={}\n",
              update->ntimestep,
              i, (atom->tag)[i], qi, num_io);
            error->one(FLERR, "Labelnext: q[i] index out of range (iopo access)");
          }
          double i_ionext=(i_res!=0)?iopo[int(q[i])][itype][jtype]:MAXENERGY;
          int i_id=(atom->tag)[i];

          int j_id=(atom->tag)[j];
          double kinenergy_j=1e4*1.0/2.0*m[j]*(v[j][0]*v[j][0]+v[j][1]*v[j][1]+v[j][2]*v[j][2]);
          int qj = static_cast<int>(q[j]);  
          int j_res = num_io - qj;

          if (qj < 0 || qj >= num_io) {
            utils::logmesg(lmp,
              "[LABELNEXT] step {}: invalid qj index: "
              "j_local={} tag_j={} qj={} num_io={}\n",
              update->ntimestep,
              j, (atom->tag)[j], qj, num_io);
            error->one(FLERR, "Labelnext: q[j] index out of range (iopo access)");
          }
          double j_ionext=(j_res!=0)?iopo[int(q[j])][itype][jtype]:MAXENERGY;
          double kinenergy_sum=kinenergy_i+kinenergy_j;
          
          // update the possible ionization events
          if ((i_res > 0 && kinenergy_sum > i_ionext) || (j_res > 0 && kinenergy_sum > j_ionext)) {
              // utils::logmesg(lmp,"Labelnext COMPUTE: step {} qi {} qj {} i_io {} j_io {} tagi {} tagj {}  energy {} cutoff {}\n",
              //     update->ntimestep,q[i],q[j],i_ionext,j_ionext,i_id,j_id,kinenergy_sum,cutsq[itype][jtype]);
              // Process for `i`
              if (rsq <= mindist[i] - EPSILON || (rsq <= mindist[i] + EPSILON && label[i]>=j_id))  {
                  label[i] = j_id;
                  mindist[i] = rsq;
              }

              // Process for `j`
              if (rsq <= mindist[j] - EPSILON || (rsq <= mindist[j] + EPSILON && label[j]>=i_id)) {
                  label[j] = i_id;
                  mindist[j] = rsq;
              }
          }

        }
        
      }
    } // second j loop
  } //second i loop

}

/* ---------------------------------------------------------------------- */

int PairLJCutIOCoulDSF::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/) {
    if (pack_flag == 1) {
        // Communicate only q
        for (int i = 0; i < n; i++) {
            buf[i] = atom->q[list[i]]; // Pack q
        }
        return n; // Number of packed values
    } else if (pack_flag == 2) {   // reset all pair
        // Communicate only mindist
        for (int i = 0; i < n; i++) {
            buf[2 * i] = BIG; // Pack mindist
            buf[2 * i + 1] = ubuf(-1).d;; // Pack label as int using ubuf
        }
        return 2 * n; // Number of packed values
    } else if (pack_flag == 3) {
        // Communicate only mindist
        for (int i = 0; i < n; i++) {
            buf[2 * i] = atom->dvector[0][list[i]]; // Pack mindist
            buf[2 * i + 1] = ubuf(atom->ivector[1][list[i]]).d; // Pack label as int using ubuf
        }
        return 2 * n; // Number of packed values
    }  else if (pack_flag == 4) {
        for (int i = 0; i < n; i++) {
            buf[3 * i] = atom->q[list[i]];               // Pack q
            buf[3 * i + 1] = atom->dvector[0][list[i]]; // Pack mindist
            buf[3 * i + 2] = ubuf(atom->ivector[1][list[i]]).d; // Pack label as int using ubuf
        }
        return 3 * n; // Total number of packed values
    }
    return 0; // Default case (should not be reached)
}


/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::unpack_forward_comm(int n, int first, double *buf) {

  int clabel,buflabel;
  double cmindist,bufmindist;
    if (pack_flag == 1) {
        // Unpack q
        for (int i = 0; i < n; i++) {
            atom->q[first + i] = std::max(buf[i],atom->q[first+i]); // Unpack q
        }
    } else if (pack_flag == 2) {
        // Unpack mindist
        for (int i = 0; i < n; i++) {
            atom->dvector[0][first + i] = buf[2 * i];;  // Unpack mindist
            atom->ivector[1][first + i] = static_cast<int>(ubuf(buf[2 * i + 1]).i); // Unpack label as int
            
        }
    } else if (pack_flag == 3) {
        // Unpack mindist
        for (int i = 0; i < n; i++) {
            cmindist=atom->dvector[0][first + i]; clabel=atom->ivector[1][first + i];
            bufmindist=buf[2 * i]; buflabel=static_cast<int>(ubuf(buf[2 * i + 1]).i); 
            
            if(bufmindist<=cmindist-EPSILON || (bufmindist<=cmindist+EPSILON&&clabel>=buflabel))
            {
                atom->dvector[0][first + i] = bufmindist;  // Unpack mindist
                atom->ivector[1][first + i] = buflabel; // Unpack label as int
            }
        }
    }  else if (pack_flag == 4) {
        for (int i = 0; i < n; i++) {
            atom->q[first + i] = std::max(buf[3 * i],atom->q[first + i]);               // Unpack q
            cmindist=atom->dvector[0][first + i]; clabel=atom->ivector[1][first + i];
            bufmindist=buf[3 * i + 1]; buflabel=static_cast<int>(ubuf(buf[3 * i + 2]).i); 
            
            if(bufmindist<=cmindist-EPSILON || (bufmindist<=cmindist+EPSILON&&clabel>=buflabel))
            {
              atom->dvector[0][first + i] = bufmindist;  // Unpack mindist
              atom->ivector[1][first + i] = buflabel; // Unpack label as int
            }

        }
    }
}


/* ---------------------------------------------------------------------- */

int PairLJCutIOCoulDSF::pack_reverse_comm(int n, int first, double *buf) {
    if (pack_flag == 1) {
        // Communicate only q
        for (int i = 0; i < n; i++) {
            buf[i] = atom->q[first + i]; // Pack q
        }
        return n; // Number of packed values
    } else if (pack_flag == 2) {
        // Communicate only mindist
        for (int i = 0; i < n; i++) {
            buf[2 * i ] = BIG;  // Pack mindist
            buf[2 * i + 1] = ubuf(-1).d; // Pack label as int using ubuf
        }
        return 2 * n; // Total number of packed values
    } else if (pack_flag == 3) {
        // Communicate only mindist
        for (int i = 0; i < n; i++) {
            buf[2 * i ] = atom->dvector[0][first + i];  // Pack mindist
            buf[2 * i + 1] = ubuf(atom->ivector[1][first + i]).d; // Pack label as int using ubuf
        }
        return 2 * n; // Total number of packed values
    } else if (pack_flag == 4) {
        for (int i = 0; i < n; i++) {
            buf[3 * i] = atom->q[first + i];               // Pack q
            buf[3 * i + 1] = atom->dvector[0][first + i];  // Pack mindist
            buf[3 * i + 2] = ubuf(atom->ivector[1][first + i]).d; // Pack label as int using ubuf
        }
        return 3 * n; // Total number of packed values
    }
    return 0; // Default case (should not be reached)
}

/* ---------------------------------------------------------------------- */

void PairLJCutIOCoulDSF::unpack_reverse_comm(int n, int *list, double *buf) {
  // pack flag =1,2,3 is only for debug usage, should never be called in the SL simulation
    int clabel,buflabel;
    double cmindist,bufmindist;
    if (pack_flag == 1) {
        // update q
        for (int i = 0; i < n; i++) {
            atom->q[list[i]] = std::max(buf[i], atom->q[list[i]]); // update q contributions
        }
    } else if (pack_flag == 2) {
        // update mindist
        for (int i = 0; i < n; i++) {
            atom->dvector[0][list[i]] = buf[2 * i];  // Unpack mindist
            atom->ivector[1][list[i]] = static_cast<int>(ubuf(buf[2 * i + 1]).i); // Unpack label as int
        }
    } else if (pack_flag == 3) {
        // update mindist
        for (int i = 0; i < n; i++) {
            cmindist=atom->dvector[0][list[i]]; clabel=atom->ivector[1][list[i]];
            bufmindist=buf[2 * i]; buflabel=static_cast<int>(ubuf(buf[2 * i + 1]).i); 

            if(bufmindist<=cmindist-EPSILON || (bufmindist<=cmindist+EPSILON&&clabel>=buflabel))
            {
              atom->dvector[0][list[i]] = bufmindist;  // Unpack mindist
              atom->ivector[1][list[i]] = buflabel; // Unpack label as int
            }
        }
    }  else if (pack_flag == 4) {
        for (int i = 0; i < n; i++) {
            atom->q[list[i]] = std::max(atom->q[list[i]],buf[3 * i]);     // if ionization happens, then q must increase
            cmindist=atom->dvector[0][list[i]]; clabel=atom->ivector[1][list[i]];
            bufmindist=buf[3 * i + 1]; buflabel=static_cast<int>(ubuf(buf[3 * i + 2]).i); 

            if(bufmindist<=cmindist-EPSILON || (bufmindist<=cmindist+EPSILON&&clabel>=buflabel))
            {
                atom->dvector[0][list[i]] = bufmindist;  // Unpack mindist
                atom->ivector[1][list[i]] = buflabel; // Unpack label as int
            }

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
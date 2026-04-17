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
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_lj_cutio.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace MathConst;

#define EPSILON 1e-8
#define MAXENERGY 1e12

/* ---------------------------------------------------------------------- */

PairLJCutIO::PairLJCutIO(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 1;
  born_matrix_enable = 1;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairLJCutIO::~PairLJCutIO()
{
  if (copymode) return;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);

    //extension for ionization
    memory->destroy(iopo);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutIO::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int iioflag=0,jioflag=0;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  double **v=atom->v;
  double *q=atom->q;
  double *m=atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  //extension for ionization
  double kinenergy_i,kinenergy_j,kinenergy_sum,kinenergy_res;
  double i_ionext,j_ionext,io_next;
  double v_scale=0.5;
  int i_res,j_res;



  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;

        
        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }
        //v[j][1]=1.0;
        kinenergy_i=1e4*1/2*m[i]*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2]);
        kinenergy_j=1e4*1/2*m[j]*(v[j][0]*v[j][0]+v[j][1]*v[j][1]+v[j][2]*v[j][2]);
        kinenergy_sum=kinenergy_i+kinenergy_j;

        i_res=num_io-int(q[i]);
        j_res=num_io-int(q[j]);

        //output index of each atom
        i_ionext=(i_res!=0)?iopo[int(q[i])][itype][jtype]:MAXENERGY;
        j_ionext=(j_res!=0)?iopo[int(q[j])][itype][jtype]:MAXENERGY;

        int i_id=(atom->tag)[i];
        int j_id=(atom->tag)[j];

        //utils::logmesg(lmp,"IODEBUG START: step {} qi {} qj {} i_io {} j_io {} tagi {} tagj {} energy {}\n",
        //           update->ntimestep,q[i],q[j],i_ionext,j_ionext,i_id,j_id, kinenergy_sum);
        while((i_res>0&&kinenergy_sum>i_ionext)||(j_res>0&&kinenergy_sum>j_ionext))
        {
          int tempi=i;
          //define cusomized symmetric ionization order here
          if (fabs(i_ionext-j_ionext)<EPSILON)
            //if have the same ionization energy, always pick the atom with a smaller global id
            tempi=(i_id<j_id)?i:j;
          else if (i_ionext>j_ionext)
            tempi=j;

          io_next=(tempi==i)?i_ionext:j_ionext;

          //loop i
          kinenergy_res=2.0/3.0*(kinenergy_i-io_next*kinenergy_i/kinenergy_sum);
          //kinenergy_res=(kinenergy_i-io_next*kinenergy_i/kinenergy_sum);
          v_scale=sqrt(kinenergy_res/kinenergy_i);
          v[i][0]*=v_scale;
          v[i][1]*=v_scale;
          v[i][2]*=v_scale; 
          kinenergy_i=kinenergy_res;

          //loop j
          kinenergy_res=2.0/3.0*(kinenergy_j-io_next*kinenergy_j/kinenergy_sum);
          //kinenergy_res=(kinenergy_j-io_next*kinenergy_j/kinenergy_sum);
          v_scale=sqrt(kinenergy_res/kinenergy_j);
          v[j][0]*=v_scale;
          v[j][1]*=v_scale;
          v[j][2]*=v_scale; 
          kinenergy_j=kinenergy_res;

          //increament q
          q[tempi]++;
          i_res=num_io-int(q[i]);
          j_res=num_io-int(q[j]);
          

          //update the io energy
          i_ionext=(i_res!=0)?iopo[int(q[i])][itype][jtype]:MAXENERGY;
          j_ionext=(j_res!=0)?iopo[int(q[j])][itype][jtype]:MAXENERGY;

          //update the sum of the energy
          kinenergy_sum=kinenergy_i+kinenergy_j;

          // utils::logmesg(lmp,"IODEBUG COMPUTE: step {} qi {} qj {} i_io {} j_io {} tagi {} tagj {} temp {} energy {}\n",
          //         update->ntimestep,q[i],q[j],i_ionext,j_ionext,i_id,j_id,tempi,kinenergy_sum);
        }
        
        /* older version*/
        /*           
        while(i_res>0&&kinenergy_sum>iopo[int(q[i])][itype][jtype])
          {
            utils::logmesg(lmp,"IODEBUG ICompute: step {} i {} j {} nlocal {} kinenergy_sum {} qi {} qj {} kinenergy_i {} kinenergy_j {}\n",
                   update->ntimestep,i,j,nlocal,kinenergy_sum,q[i],q[j],kinenergy_i,kinenergy_j);
            kinenergy_res=2.0/3.0*(kinenergy_i-iopo[int(q[i])][itype][jtype]*kinenergy_i/kinenergy_sum);
            v_scale=sqrt(kinenergy_res/kinenergy_i);
            //utils::logmesg(lmp,"IODEBUG MIDDLE: step {} i {} j {} nlocal {} kinenergy_res {} kinenergy_i {} kinenergy_j {} io {}\n",
             //      update->ntimestep,i,j,nlocal,kinenergy_res,kinenergy_i,kinenergy_j,(kinenergy_i-iopo[int(q[i])][itype][jtype]*kinenergy_i/kinenergy_sum));
            v[i][0]*=v_scale;
            v[i][1]*=v_scale;
            v[i][2]*=v_scale;         
            //f[i][0]   
            kinenergy_i=kinenergy_res;

            // if on different procs
            kinenergy_res=2.0/3.0*(kinenergy_j-iopo[int(q[i])][itype][jtype]*kinenergy_j/kinenergy_sum);
            v_scale=sqrt(kinenergy_res/kinenergy_j);
            
            v[j][0]*=v_scale;
            v[j][1]*=v_scale;
            v[j][2]*=v_scale;
            
            kinenergy_j=kinenergy_res;
          
            i_res-=1;
            
            kinenergy_sum=kinenergy_i+kinenergy_j;

            q[i]+=1;
          }
        
        //int qj=int(q[j]);
        while(j_res>0&&kinenergy_sum>iopo[int(q[j])][itype][jtype]){
          utils::logmesg(lmp,"IODEBUG JCompute: step {} i {} j {} nlocal  {} kinenergy_sum {} qi {} qj {} kinenergy_i {} kinenergy_j {}\n",
                   update->ntimestep,i,j,nlocal,kinenergy_sum,q[i],q[j],kinenergy_i,kinenergy_j);
          kinenergy_res=2.0/3.0*(kinenergy_j-iopo[int(q[j])][itype][jtype]*kinenergy_j/kinenergy_sum);
          v_scale=sqrt(kinenergy_res/kinenergy_j);
          v[j][0]*=v_scale;
          v[j][1]*=v_scale;
          v[j][2]*=v_scale;
          kinenergy_j=kinenergy_res;
          
          kinenergy_res=2.0/3.0*(kinenergy_i-iopo[int(q[j])][itype][jtype]*kinenergy_i/kinenergy_sum);
          v_scale=sqrt(kinenergy_res/kinenergy_i);
          v[i][0]*=v_scale;
          v[i][1]*=v_scale;
          v[i][2]*=v_scale;
          kinenergy_i=kinenergy_res;

          j_res-=1;
          kinenergy_sum=kinenergy_i+kinenergy_j;

          //qj+=1;
          //if(j<nlocal)
          q[j]+=1;
        }
        
        */
        // utils::logmesg(lmp,"IODEBUG FINAL: step {} qi {} qj {} i_io {} j_io {} tagi {} tagj {} \n",
        //           update->ntimestep,q[i],q[j],i_ionext,j_ionext,i_id,j_id);
        //utils::logmesg(lmp,"COMPUTE \n");
        if (eflag) {
          evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
          evdwl *= factor_lj;
        }
        //utils::logmesg(lmp,"IODEBUG FINAL: evflag {}\n",evflag);
        //           update->ntimestep,q[i],q[j],i_ionext,j_ionext,i_id,j_id);
        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairLJCutIO::compute_inner()
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum_inner;
  ilist = list->ilist_inner;
  numneigh = list->numneigh_inner;
  firstneigh = list->firstneigh_inner;

  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];

  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on * cut_out_on;
  double cut_out_off_sq = cut_out_off * cut_out_off;

  double *q=atom->q;
  double *m=atom->rmass;
  double **v=atom->v;
  //extension for ionization
  double kinenergy_i,kinenergy_j,kinenergy_sum,kinenergy_res;
  double v_scale=0.5;
  int i_res,j_res;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_out_off_sq) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on) / cut_out_diff;
          fpair *= 1.0 - rsw * rsw * (3.0 - 2.0 * rsw);
        }

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }
        
        kinenergy_i=1e4*1/2*m[i]*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2]);
        kinenergy_j=1e4*1/2*m[j]*(v[j][0]*v[j][0]+v[j][1]*v[j][1]+v[j][2]*v[j][2]);
        kinenergy_sum=kinenergy_i+kinenergy_j;

        i_res=num_io-int(q[i]);
        j_res=num_io-int(q[j]);

          while(i_res>0&&kinenergy_sum>iopo[int(q[i])][itype][jtype])
          {
            kinenergy_res=2.0/3.0*(kinenergy_i-iopo[int(q[i])][itype][jtype]*kinenergy_i/kinenergy_sum);
            //v_scale=sqrt(kinenergy_res/kinenergy_i);
            v[i][0]*=v_scale;
            v[i][1]*=v_scale;
            v[i][2]*=v_scale;           
            kinenergy_i=kinenergy_res;

            kinenergy_res=2.0/3.0*(kinenergy_j-iopo[int(q[i])][itype][jtype]*kinenergy_j/kinenergy_sum);
            //v_scale=sqrt(kinenergy_res/kinenergy_j);
            
            v[j][0]*=v_scale;
            v[j][1]*=v_scale;
            v[j][2]*=v_scale;
            
            kinenergy_j=kinenergy_res;

            i_res-=1;
            kinenergy_sum=kinenergy_i+kinenergy_j;

            q[i]+=1;
          }

        while(j_res>0&&kinenergy_sum>iopo[int(q[j])][itype][jtype]){

          kinenergy_res=2.0/3.0*(kinenergy_j-iopo[int(q[j])][itype][jtype]*kinenergy_j/kinenergy_sum);
          //v_scale=sqrt(kinenergy_res/kinenergy_j);
          v[j][0]*=v_scale;
          v[j][1]*=v_scale;
          v[j][2]*=v_scale;
          kinenergy_j=kinenergy_res;
          
          kinenergy_res=2.0/3.0*(kinenergy_i-iopo[int(q[j])][itype][jtype]*kinenergy_i/kinenergy_sum);
          //v_scale=sqrt(kinenergy_res/kinenergy_i);
          v[i][0]*=v_scale;
          v[i][1]*=v_scale;
          v[i][2]*=v_scale;
          kinenergy_i=kinenergy_res;

          j_res-=1;
          kinenergy_sum=kinenergy_i+kinenergy_j;

          q[j]+=1;
        }
        utils::logmesg(lmp,"COMPUTE INNER \n");
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutIO::compute_middle()
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum_middle;
  ilist = list->ilist_middle;
  numneigh = list->numneigh_middle;
  firstneigh = list->firstneigh_middle;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off * cut_in_off;
  double cut_in_on_sq = cut_in_on * cut_in_on;
  double cut_out_on_sq = cut_out_on * cut_out_on;
  double cut_out_off_sq = cut_out_off * cut_out_off;


  double *q=atom->q;
  double *m=atom->rmass;
  double **v=atom->v;
  //extension for ionization
  double kinenergy_i,kinenergy_j,kinenergy_sum,kinenergy_res;
  double v_scale=0.5;

  int i_res,j_res;
  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;
        if (rsq < cut_in_on_sq) {
          rsw = (sqrt(rsq) - cut_in_off) / cut_in_diff;
          fpair *= rsw * rsw * (3.0 - 2.0 * rsw);
        }
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on) / cut_out_diff;
          fpair *= 1.0 + rsw * rsw * (2.0 * rsw - 3.0);
        }

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }


        kinenergy_i=1e4*1/2*m[i]*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2]);
        kinenergy_j=1e4*1/2*m[j]*(v[j][0]*v[j][0]+v[j][1]*v[j][1]+v[j][2]*v[j][2]);
        kinenergy_sum=kinenergy_i+kinenergy_j;

        i_res=num_io-int(q[i]);
        j_res=num_io-int(q[j]);

          while(i_res>0&&kinenergy_sum>iopo[int(q[i])][itype][jtype])
          {
            kinenergy_res=2.0/3.0*(kinenergy_i-iopo[int(q[i])][itype][jtype]*kinenergy_i/kinenergy_sum);
            //v_scale=sqrt(kinenergy_res/kinenergy_i);
            v[i][0]*=v_scale;
            v[i][1]*=v_scale;
            v[i][2]*=v_scale;
            kinenergy_i=kinenergy_res;

            kinenergy_res=2.0/3.0*(kinenergy_j-iopo[int(q[i])][itype][jtype]*kinenergy_j/kinenergy_sum);
            //v_scale=sqrt(kinenergy_res/kinenergy_j);
            
            
            v[j][0]*=v_scale;
            v[j][1]*=v_scale;
            v[j][2]*=v_scale;
            
            kinenergy_j=kinenergy_res;

            i_res-=1;
            kinenergy_sum=kinenergy_i+kinenergy_j;

            q[i]+=1;
          }

        while(j_res>0&&kinenergy_sum>iopo[int(q[j])][itype][jtype]){

          kinenergy_res=2.0/3.0*(kinenergy_j-iopo[int(q[j])][itype][jtype]*kinenergy_j/kinenergy_sum);
          //v_scale=sqrt(kinenergy_res/kinenergy_j);
          v[j][0]*=v_scale;
          v[j][1]*=v_scale;
          v[j][2]*=v_scale;
          kinenergy_j=kinenergy_res;
          
          kinenergy_res=2.0/3.0*(kinenergy_i-iopo[int(q[j])][itype][jtype]*kinenergy_i/kinenergy_sum);
         // v_scale=sqrt(kinenergy_res/kinenergy_i);
          v[i][0]*=v_scale;
          v[i][1]*=v_scale;
          v[i][2]*=v_scale;
          kinenergy_i=kinenergy_res;

          j_res-=1;
          kinenergy_sum=kinenergy_i+kinenergy_j;

          q[j]+=1;
        }
        utils::logmesg(lmp,"COMPUTE MIDDLE \n");
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCutIO::compute_outer(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off * cut_in_off;
  double cut_in_on_sq = cut_in_on * cut_in_on;

  double *q=atom->q;
  double *m=atom->rmass;
  double **v=atom->v;
  //extension for ionization
  double kinenergy_i,kinenergy_j,kinenergy_sum,kinenergy_res;
  double v_scale=0.5;

  int i_res,j_res;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        if (rsq > cut_in_off_sq) {
          r2inv = 1.0 / rsq;
          r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
          fpair = factor_lj * forcelj * r2inv;
          if (rsq < cut_in_on_sq) {
            rsw = (sqrt(rsq) - cut_in_off) / cut_in_diff;
            fpair *= rsw * rsw * (3.0 - 2.0 * rsw);
          }

          f[i][0] += delx * fpair;
          f[i][1] += dely * fpair;
          f[i][2] += delz * fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= delx * fpair;
            f[j][1] -= dely * fpair;
            f[j][2] -= delz * fpair;
          }
        }

        kinenergy_i=1e4*1/2*m[i]*(v[i][0]*v[i][0]+v[i][1]*v[i][1]+v[i][2]*v[i][2]);
        kinenergy_j=1e4*1/2*m[j]*(v[j][0]*v[j][0]+v[j][1]*v[j][1]+v[j][2]*v[j][2]);
        kinenergy_sum=kinenergy_i+kinenergy_j;

        i_res=num_io-int(q[i]);
        j_res=num_io-int(q[j]);

          while(i_res>0&&kinenergy_sum>iopo[int(q[i])][itype][jtype])
          {
            kinenergy_res=2.0/3.0*(kinenergy_i-iopo[int(q[i])][itype][jtype]*kinenergy_i/kinenergy_sum);
            //v_scale=sqrt(kinenergy_res/kinenergy_i);
            v[i][0]*=v_scale;
            v[i][1]*=v_scale;
            v[i][2]*=v_scale;
            
            kinenergy_i=kinenergy_res;

            kinenergy_res=2.0/3.0*(kinenergy_j-iopo[int(q[i])][itype][jtype]*kinenergy_j/kinenergy_sum);
            //v_scale=sqrt(kinenergy_res/kinenergy_j);
            
            v[j][0]*=v_scale;
            v[j][1]*=v_scale;
            v[j][2]*=v_scale;
            
            kinenergy_j=kinenergy_res;

            i_res-=1;
            kinenergy_sum=kinenergy_i+kinenergy_j;

            q[i]+=1;
          }

        while(j_res>0&&kinenergy_sum>iopo[int(q[j])][itype][jtype]){

          kinenergy_res=2.0/3.0*(kinenergy_j-iopo[int(q[j])][itype][jtype]*kinenergy_j/kinenergy_sum);
          //v_scale=sqrt(kinenergy_res/kinenergy_j);
          v[j][0]*=v_scale;
          v[j][1]*=v_scale;
          v[j][2]*=v_scale;          
          kinenergy_j=kinenergy_res;
          
          kinenergy_res=2.0/3.0*(kinenergy_i-iopo[int(q[j])][itype][jtype]*kinenergy_i/kinenergy_sum);
          //v_scale=sqrt(kinenergy_res/kinenergy_i);
          v[i][0]*=v_scale;
          v[i][1]*=v_scale;
          v[i][2]*=v_scale;
          kinenergy_i=kinenergy_res;

          j_res-=1;
          kinenergy_sum=kinenergy_i+kinenergy_j;

          q[j]+=1;
        }

        utils::logmesg(lmp,"COMPUTE OUTER \n");
        if (eflag) {
          r2inv = 1.0 / rsq;
          r6inv = r2inv * r2inv * r2inv;
          evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (vflag) {
          if (rsq <= cut_in_off_sq) {
            r2inv = 1.0 / rsq;
            r6inv = r2inv * r2inv * r2inv;
            forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
            fpair = factor_lj * forcelj * r2inv;
          } else if (rsq < cut_in_on_sq)
            fpair = factor_lj * forcelj * r2inv;
        }

        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCutIO::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair: ");

  memory->create(cut, n, n, "pair:cut");
  memory->create(epsilon, n, n, "pair:epsilon");
  memory->create(sigma, n, n, "pair:sigma");
  memory->create(lj1, n, n, "pair:lj1");
  memory->create(lj2, n, n, "pair:lj2");
  memory->create(lj3, n, n, "pair:lj3");
  memory->create(lj4, n, n, "pair:lj4");
  memory->create(offset, n, n, "pair:offset");

  //extension for ionization
  memory->create(iopo,8,n,n,"pair:iopo");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCutIO::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCutIO::coeff(int narg, char **arg)
{
  if (narg < 6 ) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  //right format should be pair_coeff type1 type2 num_io io1, io2, ..., io_{num} epsilon sigma (cutoff)
  num_io=utils::numeric(FLERR, arg[2], false, lmp);
  
  //extension for ionization
  double iopo_eight[8];
  for(int i=0;i<num_io;i++)
    iopo_eight[i]=utils::numeric(FLERR, arg[3+i], false, lmp);

  double epsilon_one = utils::numeric(FLERR, arg[3+num_io], false, lmp);
  double sigma_one = utils::numeric(FLERR, arg[4+num_io], false, lmp);

  double cut_one = cut_global;
  if (narg == 6+num_io) cut_one = utils::numeric(FLERR, arg[5+num_io], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      //extension for ionization
      for(int k=0;k<num_io;k++)
        iopo[k][i][j]=iopo_eight[k];

      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCutIO::init_style()
{
  // request regular or rRESPA neighbor list

  int list_style = NeighConst::REQ_DEFAULT;

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
    auto respa = dynamic_cast<Respa *>(update->integrate);
    if (respa->level_inner >= 0) list_style = NeighConst::REQ_RESPA_INOUT;
    if (respa->level_middle >= 0) list_style = NeighConst::REQ_RESPA_ALL;
  }
  neighbor->add_request(this, list_style);

  // set rRESPA cutoffs

  if (utils::strmatch(update->integrate_style, "^respa") &&
      (dynamic_cast<Respa *>(update->integrate))->level_inner >= 0)
    cut_respa = (dynamic_cast<Respa *>(update->integrate))->cutoff;
  else
    cut_respa = nullptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCutIO::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j], sigma[i][i], sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i], sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);

  if (offset_flag && (cut[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio, 12.0) - pow(ratio, 6.0));
  } else
    offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && cut[i][j] < cut_respa[3])
    error->all(FLERR, "Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2], all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count, all, 2, MPI_DOUBLE, MPI_SUM, world);

    double sig2 = sigma[i][j] * sigma[i][j];
    double sig6 = sig2 * sig2 * sig2;
    double rc3 = cut[i][j] * cut[i][j] * cut[i][j];
    double rc6 = rc3 * rc3;
    double rc9 = rc3 * rc6;
    double prefactor = 8.0 * MY_PI * all[0] * all[1] * epsilon[i][j] * sig6 / (9.0 * rc9);
    etail_ij = prefactor * (sig6 - 3.0 * rc6);
    ptail_ij = 2.0 * prefactor * (2.0 * sig6 - 3.0 * rc6);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCutIO::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j], sizeof(double), 1, fp);
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
        //extension for ionization
        for (int k=0;k<num_io;k++)
          fwrite(&iopo[k][i][j], sizeof(double), 1, fp);

      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutIO::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &epsilon[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);

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

void PairLJCutIO::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  fwrite(&tail_flag, sizeof(int), 1, fp);

  fwrite(&num_io,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCutIO::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, nullptr, error);

    utils::sfread(FLERR, &num_io, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);

  MPI_Bcast(&num_io, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJCutIO::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g %g\n", i, epsilon[i][i], sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJCutIO::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g\n", i, j, epsilon[i][j], sigma[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJCutIO::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r2inv, r6inv, forcelj, philj;

  r2inv = 1.0 / rsq;
  r6inv = r2inv * r2inv * r2inv;
  forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
  fforce = factor_lj * forcelj * r2inv;

  philj = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
  return factor_lj * philj;
}

/* ---------------------------------------------------------------------- */

void PairLJCutIO::born_matrix(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                            double /*factor_coul*/, double factor_lj, double &dupair,
                            double &du2pair)
{
  double rinv, r2inv, r6inv, du, du2;

  r2inv = 1.0 / rsq;
  rinv = sqrt(r2inv);
  r6inv = r2inv * r2inv * r2inv;

  // Reminder: lj1 = 48*e*s^12, lj2 = 24*e*s^6
  // so dupair = -forcelj/r = -fforce*r (forcelj from single method)

  du = r6inv * rinv * (lj2[itype][jtype] - lj1[itype][jtype] * r6inv);
  du2 = r6inv * r2inv * (13 * lj1[itype][jtype] * r6inv - 7 * lj2[itype][jtype]);

  dupair = factor_lj * du;
  du2pair = factor_lj * du2;
}

/* ---------------------------------------------------------------------- */

void *PairLJCutIO::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "epsilon") == 0) return (void *) epsilon;
  if (strcmp(str, "sigma") == 0) return (void *) sigma;
  return nullptr;
}

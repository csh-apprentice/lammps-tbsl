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

#include "fix_wall_mdhb.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "math_const.h"
#include "math_special.h"
#include "region.h"
#include "respa.h"
#include "input.h"
#include "update.h"
#include "pointers.h"    // IWYU pragma: export
#include "variable.h"
#include "modify.h"
#include "force.h"
#include "pair.h"
#include "kspace.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;
using MathConst::MY_PI;
using MathConst::MY_2PI;
using MathConst::MY_SQRT2;
using MathConst::KB;
using MathConst::NA;
using MathSpecial::powint;

enum { LJ93, LJ126, LJ1043, COLLOID, HARMONIC, MORSE };
enum { CONSTANT, VARIABLE, AUTO };

/* ---------------------------------------------------------------------- */

FixWallMDHeatbath::FixWallMDHeatbath(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), idregion(nullptr), region(nullptr), Tblstr(nullptr)
{
  if (narg < 8) error->all(FLERR, "Illegal fix wall/region command");

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  energy_global_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  // parse args

  region = domain->get_region_by_id(arg[3]);
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", arg[3]);
  idregion = utils::strdup(arg[3]);

  if (strcmp(arg[4], "lj93") == 0)
    style = LJ93;
  else if (strcmp(arg[4], "lj126") == 0)
    style = LJ126;
  else if (strcmp(arg[4], "lj1043") == 0)
    style = LJ1043;
  else if (strcmp(arg[4], "colloid") == 0)
    style = COLLOID;
  else if (strcmp(arg[4], "harmonic") == 0)
    style = HARMONIC;
  else if (strcmp(arg[4], "morse") == 0)
    style = MORSE;
  else
    error->all(FLERR, "Illegal fix wall/region command");

  if (style != COLLOID) dynamic_group_allow = 1;

  if (style == MORSE) {
    if (narg != 16) error->all(FLERR, "Illegal fix wall/region command");

    epsilon = utils::numeric(FLERR, arg[5], false, lmp);
    alpha = utils::numeric(FLERR, arg[6], false, lmp);
    sigma = utils::numeric(FLERR, arg[7], false, lmp);
    cutoff = utils::numeric(FLERR, arg[8], false, lmp);
    mix_coeff=utils::numeric(FLERR, arg[9], false, lmp);
    scalefactor=utils::numeric(FLERR, arg[10], false, lmp);
    density=utils::numeric(FLERR, arg[11], false, lmp);
    capacity=utils::numeric(FLERR, arg[12], false, lmp);
    Tinfty=utils::numeric(FLERR, arg[13], false, lmp)*scalefactor;
    Tblgasinit=utils::numeric(FLERR, arg[14], false, lmp)*scalefactor;
    // Tbl variable checking
    if (utils::strmatch(arg[15], "^v_")) {
    Tblstr = utils::strdup(arg[15] + 2);
    Tblstyle=VARIABLE;
    varshape=1;
    }
    else 
    {
      Tbl=utils::numeric(FLERR, arg[15], false, lmp)*scalefactor;
      Tblstyle=CONSTANT;
    }    
    // shellpressurestr=utils::strdup(arg[16] + 2);
    // shellkestressstr=utils::strdup(arg[17] + 2);

  } else {
    if (narg != 15) error->all(FLERR, "Illegal fix wall/region command");
    epsilon = utils::numeric(FLERR, arg[5], false, lmp);
    sigma = utils::numeric(FLERR, arg[6], false, lmp);
    cutoff = utils::numeric(FLERR, arg[7], false, lmp);
    mix_coeff=utils::numeric(FLERR, arg[8], false, lmp);
    scalefactor=utils::numeric(FLERR, arg[9], false, lmp);
    density=utils::numeric(FLERR, arg[10], false, lmp);
    capacity=utils::numeric(FLERR, arg[11], false, lmp);
    Tinfty=utils::numeric(FLERR, arg[12], false, lmp)*scalefactor;
    Tblgasinit=utils::numeric(FLERR, arg[13], false, lmp)*scalefactor;
    if (utils::strmatch(arg[14], "^v_")) {
    Tblstr = utils::strdup(arg[14] + 2);
    Tblstyle=VARIABLE;
    varshape=1;
    }
    else 
    {
      Tbl=utils::numeric(FLERR, arg[14], false, lmp)*scalefactor;
      //utils::logmesg(lmp,"Input Tbl is {} \n",Tbl/scalefactor);
      Tblstyle=CONSTANT;
    }
    // shellpressurestr=utils::strdup(arg[15] + 2);
    // shellkestressstr=utils::strdup(arg[16] + 2);
  }

  if (cutoff <= 0.0) error->all(FLERR, "Fix wall/region cutoff <= 0.0");

  eflag = 0;
  ewall[0] = ewall[1] = ewall[2] = ewall[3] = 0.0;
  
  
  // variable_check();
  if (varshape)
  {
    variable_check();
    temp_update();
  }
}

/* ---------------------------------------------------------------------- */

FixWallMDHeatbath::~FixWallMDHeatbath()
{
  delete[] idregion;
  delete[] Tblstr;
}

/* ---------------------------------------------------------------------- */

int FixWallMDHeatbath::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixWallMDHeatbath::init()
{
  // set index and check validity of region
  //ev_setup(1,1);
  region = domain->get_region_by_id(idregion);
  if (!region) error->all(FLERR, "Region {} for fix wall/region does not exist", idregion);

  // error checks for style COLLOID
  // ensure all particles in group are extended particles

  if (style == COLLOID) {
    if (!atom->sphere_flag) error->all(FLERR, "Fix wall/region colloid requires atom style sphere");

    double *radius = atom->radius;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;

    int flag = 0;
    for (int i = 0; i < nlocal; i++)
      if (mask[i] & groupbit)
        if (radius[i] == 0.0) flag = 1;

    int flagall;
    MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);
    if (flagall) error->all(FLERR, "Fix wall/region colloid requires extended particles");
  }

  // setup coefficients for each style

  if (style == LJ93) {
    coeff1 = 6.0 / 5.0 * epsilon * powint(sigma, 9);
    coeff2 = 3.0 * epsilon * powint(sigma, 3);
    coeff3 = 2.0 / 15.0 * epsilon * powint(sigma, 9);
    coeff4 = epsilon * powint(sigma, 3);
    double rinv = 1.0 / cutoff;
    double r2inv = rinv * rinv;
    double r4inv = r2inv * r2inv;
    offset = coeff3 * r4inv * r4inv * rinv - coeff4 * r2inv * rinv;
  } else if (style == LJ126) {
    coeff1 = 48.0 * epsilon * powint(sigma, 12);
    coeff2 = 24.0 * epsilon * powint(sigma, 6);
    coeff3 = 4.0 * epsilon * powint(sigma, 12);
    coeff4 = 4.0 * epsilon * powint(sigma, 6);
    double r2inv = 1.0 / (cutoff * cutoff);
    double r6inv = r2inv * r2inv * r2inv;
    offset = r6inv * (coeff3 * r6inv - coeff4);
  } else if (style == LJ1043) {
    coeff1 = MY_2PI * 2.0 / 5.0 * epsilon * powint(sigma, 10);
    coeff2 = MY_2PI * epsilon * powint(sigma, 4);
    coeff3 = MY_2PI * MY_SQRT2 / 3.0 * epsilon * powint(sigma, 3);
    coeff4 = 0.61 / MY_SQRT2 * sigma;
    coeff5 = coeff1 * 10.0;
    coeff6 = coeff2 * 4.0;
    coeff7 = coeff3 * 3.0;
    double rinv = 1.0 / cutoff;
    double r2inv = rinv * rinv;
    double r4inv = r2inv * r2inv;
    offset = coeff1 * r4inv * r4inv * r2inv - coeff2 * r4inv - coeff3 * powint(cutoff + coeff4, -3);
  } else if (style == MORSE) {
    coeff1 = 2 * epsilon * alpha;
    double alpha_dr = -alpha * (cutoff - sigma);
    offset = epsilon * (exp(2.0 * alpha_dr) - 2.0 * exp(alpha_dr));
  } else if (style == COLLOID) {
    coeff1 = -4.0 / 315.0 * epsilon * powint(sigma, 6);
    coeff2 = -2.0 / 3.0 * epsilon;
    coeff3 = epsilon * powint(sigma, 6) / 7560.0;
    coeff4 = epsilon / 6.0;
    double rinv = 1.0 / cutoff;
    double r2inv = rinv * rinv;
    double r4inv = r2inv * r2inv;
    offset = coeff3 * r4inv * r4inv * rinv - coeff4 * r2inv * rinv;
  }

  if (utils::strmatch(update->integrate_style, "^respa")) {
    ilevel_respa = (dynamic_cast<Respa *>(update->integrate))->nlevels - 1;
    if (respa_level >= 0) ilevel_respa = MIN(respa_level, ilevel_respa);
  }
  liquidenergy=cal_liquid_energy(region->SL_lastradius,region->SL_deltaold);
  region->SL_Tblgas=Tblgasinit;
  region->SL_Tblgasold=Tblgasinit;
  region->SL_Tblliquid=Tbl;
  region->SL_Tblliquidold=Tbl;

}

/* ---------------------------------------------------------------------- */

void FixWallMDHeatbath::setup(int vflag)
{
  //  utils::logmesg(lmp,"RESTART FLAG IS {}",region->SL_restart);
  if (utils::strmatch(update->integrate_style, "^respa")) {
    //utils::logmesg(lmp,"Respa mode!");
    auto respa = dynamic_cast<Respa *>(update->integrate);
    respa->copy_flevel_f(ilevel_respa);
    if(!region->SL_restart) post_force_respa(vflag, ilevel_respa, 0);
    respa->copy_f_flevel(ilevel_respa);
  } else {
    if(!region->SL_restart) post_force(vflag);
    else modify->addstep_compute_all(update->ntimestep+1);
    // modify->addstep_compute_all(0);
    // modify->addstep_compute_all(1);
    // modify->addstep_compute_all(2);
    // modify->addstep_compute_all(3);
    // modify->addstep_compute_all(4);
    // modify->addstep_compute_all(5);
    //post_force(vflag);
  }
}

/* ---------------------------------------------------------------------- */

void FixWallMDHeatbath::min_setup(int vflag)
{
  post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallMDHeatbath::post_force(int vflag)
{
  int i, m, n;
  double rinv, fx, fy, fz, tooclose;
  double delx, dely, delz, v[6];
  double rvx,rvy,rvz;
  double rx,ry,rz,rr,invrr;

  double **x = atom->x;
  double **f = atom->f;
  double **av=atom->v;
  double *am=atom->rmass;
  double *radius = atom->radius;
  double **pair_vatom = force->pair->vatom;
  double *pair_virial_total = force->pair->virial;
    // flag Kspace contribution separately, since not summed across procs
  double **kspace_vatom=nullptr;
  if (force->kspace) kspaceflag=1; else kspaceflag=0;
  if(kspaceflag) kspace_vatom= force->kspace->vatom;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *tlast=atom->ivector[0];
  
  double vscale;

  double old_kienergy;
  double new_kienergy;
  double sum_kienergy=0.0;
  int numatoms=0;

  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  double mvv2e = force->mvv2e;
  double nktv2p = -force->nktv2p;
  double fix_stress_sum=0.0;
  double pair_stress_sum=0.0;
  double kspace_stress_sum=0.0;
  double ke_stress_sum=0.0;
  double energy_loss=0.0;

  int numlost=0; // lost particles in each thread
  int nghost=0;
  if (force->newton) nghost= atom->nghost;

  pair_virial[0]=pair_virial[1]=pair_virial[2]=pair_virial[3]=pair_virial[4]=pair_virial[5]=0.0;
  kspace_virial[0]=kspace_virial[1]=kspace_virial[2]=kspace_virial[3]=kspace_virial[4]=kspace_virial[5]=0.0;
  ke_virial[0]=ke_virial[1]=ke_virial[2]=ke_virial[3]=ke_virial[4]=ke_virial[5]=0.0;
  //int numatoms=0;  //number of atoms in the shell

  //if(update->ntimestep%1000==0)
  //  utils::logmesg(lmp,"varshape {} dynamic {}\n",varshape,dynamic);
  // region->prematch();
  // modify->addstep_compute(update->ntimestep+1);
  // modify->addstep_compute(update->ntimestep);
  update->vflag_atom=update->ntimestep;
  if (varshape) 
  {
    temp_update();
  }
  int onflag = 0;

  // virial setup

  v_init(vflag);
  // region->match() ensures particle is in region or on surface, else error
  // if returned contact dist r = 0, is on surface, also an error
  // in COLLOID case, r <= radius is an error
  // initilize ewall after region->prematch(),
  //   so a dynamic region can access last timestep values

  ewall[0] = ewall[1] = ewall[2] = ewall[3] = 0.0;

  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (!region->match(x[i][0], x[i][1], x[i][2])) {
        numlost++;
        onflag = 1;
        continue;
      }
      if (style == COLLOID)
        tooclose = radius[i];
      else
        tooclose = 0.0;
      //utils::logmesg(lmp,"Index {} with x {}, y {}, z {} ", (atom->tag)[i],x[i][0],x[i][1],x[i][2]);
      n = region->surface(x[i][0], x[i][1], x[i][2], cutoff);

      for (m = 0; m < n; m++) {
        if (region->contact[m].r <= tooclose) {
          onflag = 1;
          continue;
        } else
          rinv = 1.0 / region->contact[m].r;

        if (style == LJ93)
          lj93(region->contact[m].r);
        else if (style == LJ126)
          lj126(region->contact[m].r);
        else if (style == LJ1043)
          lj1043(region->contact[m].r);
        else if (style == MORSE)
          morse(region->contact[m].r);
        else if (style == COLLOID)
          colloid(region->contact[m].r, radius[i]);
        else
          harmonic(region->contact[m].r);

        delx = region->contact[m].delx;
        dely = region->contact[m].dely;
        delz = region->contact[m].delz;
        fx = fwall * delx * rinv;
        fy = fwall * dely * rinv;
        fz = fwall * delz * rinv;
        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        ewall[1] -= fx;
        ewall[2] -= fy;
        ewall[3] -= fz;
        ewall[0] += eng;

         //scale the velocity to match the B.C.
         // for restart, always skip the first step
        if(tlast[i]!=update->ntimestep-1)
        {
        // scaling the temperature in the first time of collision  
        rx=x[i][0]-delx;
        ry=x[i][1]-dely;
        rz=x[i][2]-delz;
        rr=sqrt(rx*rx+ry*ry+rz*rz);
        invrr=1.0/rr;

        old_kienergy=1.0/2.0*am[i]*(av[i][0]*av[i][0]+av[i][1]*av[i][1]+av[i][2]*av[i][2]); 
        new_kienergy=3.0/2.0*KB*Tbl*NA/1e7;
      
        //utils::logmesg(lmp,"OLD FIX WALL HB DEBUG: avx {} avy {} avz {} old energy {} new energy {}\n",av[i][0],av[i][1],av[i][2],old_kienergy,new_kienergy);
        //utils::logmesg(lmp,"FIX WALL HB DEBUG: ry {} rz {} dely {} delz {} rvy {} rvz {}\n",ry,rz,dely,delz,rvy,rvz);
        
        
        //utils::logmesg(lmp,"Current Temperature is {} \n",Tbl);
        vscale=sqrt((1-mix_coeff+mix_coeff*new_kienergy/old_kienergy));
        energy_loss+=(mix_coeff*(old_kienergy-new_kienergy))*1e7/NA;  // use the standard unit
        //sum_kienergy+=mix_coeff*new_kienergy+(1-mix_coeff)*old_kienergy;
        // utils::logmesg(lmp,"Current time is {}, Index {} tlast is {} \n", update->ntimestep, (atom->tag)[i], tlast[i]);
        av[i][0]*=vscale;
        av[i][1]*=vscale;
        av[i][2]*=vscale;
        //utils::logmesg(lmp,"NEW FIX WALL HB DEBUG: vscale {} avx {} avy {} avz {}\n",vscale,av[i][0],av[i][1],av[i][2]);
        // TODO: This part may need to be modified
        }
        //evflag=1;
        //vflag_global=1;
        // sjip the first step
        if (evflag) {
          v[0] = fx * delx;
          v[1] = fy * dely;
          v[2] = fz * delz;
          v[3] = fx * dely;
          v[4] = fx * delz;
          v[5] = fy * delz;
          //if(!region->SL_restart) v_tally(i, v);
          //else v_tally_novel(i,v);
          //utils::logmesg(lmp,"FIX WALL HB DEBUG: Step {} Index {}, x {}, y {}, z {} avx {} avy {} avz {}, fix virial is {}, fx is {}, delx {}, rinv {}, fwall {}\n",update->ntimestep,(atom->tag)[i],x[i][0],x[i][1],x[i][2], av[i][0],av[i][1],av[i][2],v[0], fx, delx, rinv, fwall);
          v_tally(i, v);
          //utils::logmesg(lmp,"FIX WALL MDHB, THE SHELL VIRIAL IN STEP {} is {}, current v0 equals {}\n",update->ntimestep,virial[i][0],v[0]);
        }
        tlast[i]=update->ntimestep;
        //num_tally(1);

        //adding the pair stress of the group
        pair_virial[0]+=pair_vatom[i][0];
        pair_virial[1]+=pair_vatom[i][1];
        pair_virial[2]+=pair_vatom[i][2];
        pair_virial[3]+=pair_vatom[i][3];
        pair_virial[4]+=pair_vatom[i][4];
        pair_virial[5]+=pair_vatom[i][5];

        //adding the ksapce stress of the group
          // flag Kspace contribution separately, since not summed across procs
        if (kspaceflag){
        kspace_virial[0]+=kspace_vatom[i][0];
        kspace_virial[1]+=kspace_vatom[i][1];
        kspace_virial[2]+=kspace_vatom[i][2];
        kspace_virial[3]+=kspace_vatom[i][3];
        kspace_virial[4]+=kspace_vatom[i][4];
        kspace_virial[5]+=kspace_vatom[i][5];
        }
        
        //utils::logmesg(lmp,"FIX WALL MDHB, IN STEP {}, current pair_vatom {} equals {}\n",update->ntimestep,i,pair_vatom[i][0]);
        // if ((atom->tag)[i]<1000)
        // utils::logmesg(lmp," FIX WALL HB DEBUG: Step {} Index {}, x {}, y {}, z {} avx {} avy {} avz {}, pair virial is {}, fx is {}, pair virial is {}\n",update->ntimestep,(atom->tag)[i],x[i][0],x[i][1],x[i][2], av[i][0],av[i][1],av[i][2],pair_virial[0], f[i][0],pair_virial_total[0]);
        // //adding the ke stress of the group 
        ke_virial[0]+=mvv2e*rmass[i]*av[i][0]*av[i][0];
        ke_virial[1]+=mvv2e*rmass[i]*av[i][1]*av[i][1];
        ke_virial[2]+=mvv2e*rmass[i]*av[i][2]*av[i][2];
        ke_virial[3]+=mvv2e*rmass[i]*av[i][0]*av[i][1];
        ke_virial[4]+=mvv2e*rmass[i]*av[i][0]*av[i][2];
        ke_virial[5]+=mvv2e*rmass[i]*av[i][1]*av[i][2];

        numatoms++;
      }
    }
    //utils::logmesg(lmp," GHOST Number is {}\n",nghost);
   for (i = nlocal; i < nlocal+nghost; i++)
   
    if (groupbit) {
      
      if (style == COLLOID)
        tooclose = radius[i];
      else
        tooclose = 0.0;

      n = region->surface(x[i][0], x[i][1], x[i][2], cutoff);

      for (m = 0; m < n; m++) {
        if (region->contact[m].r <= tooclose) {
          onflag = 1;
          continue;
        } else
          rinv = 1.0 / region->contact[m].r;
        //adding the pair stress of the group
        pair_virial[0]+=pair_vatom[i][0];
        pair_virial[1]+=pair_vatom[i][1];
        pair_virial[2]+=pair_vatom[i][2];
        pair_virial[3]+=pair_vatom[i][3];
        pair_virial[4]+=pair_vatom[i][4];
        pair_virial[5]+=pair_vatom[i][5];
        // if ((atom->tag)[i]<1000)
        // utils::logmesg(lmp," GHOST FIX WALL HB DEBUG: Step {} Index {}, x {}, y {}, z {} avx {} avy {} avz {}, pair virial is {}, fx is {}, pair virial is {}\n",update->ntimestep,(atom->tag)[i],x[i][0],x[i][1],x[i][2], av[i][0],av[i][1],av[i][2],pair_virial[0], f[i][0],pair_virial_total[0]);
      }
    }
    
  //if (onflag) error->one(FLERR, "Particle outside surface of region used in fix wall/region");  // it's okay to lost particles
  MPI_Allreduce(&numlost, &sumlostparticles, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&numatoms, &sumnumatoms, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&energy_loss,&sumenergyloss,1,MPI_DOUBLE,MPI_SUM,world);
  MPI_Allreduce(virial, fix_stress, 6, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(pair_virial, pair_stress, 6, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(kspace_virial, kspace_stress, 6, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(ke_virial,ke_stress,6,MPI_DOUBLE,MPI_SUM,world);
  // utils::logmesg(lmp,"In step {}, sumenergyloss is {}, Tblgas before update is {}\n",update->ntimestep,sumenergyloss,region->SL_Tblgas);
  fix_stress_sum=(fix_stress[0]+fix_stress[1]+fix_stress[2])*nktv2p;
  pair_stress_sum=(pair_stress[0]+pair_stress[1]+pair_stress[2])*nktv2p;
  kspace_stress_sum=(kspace_stress[0]+kspace_stress[1]+kspace_stress[2])*nktv2p;
  ke_stress_sum=(ke_stress[0]+ke_stress[1]+ke_stress[2])*nktv2p;
  gasshellenergy=(ke_stress[0]+ke_stress[1]+ke_stress[2])*1.0/(2.0*mvv2e);
  //calculate the pressure
  
  // double shellstress= input->variable->compute_equal(shellpressurevar);
  // double readgasshellenergy= input->variable->compute_equal(shellkestressvar)/nktv2p*1.0/(2.0*mvv2e);
  stress_all=fix_stress_sum+pair_stress_sum+kspace_stress_sum+ke_stress_sum;
  // utils::logmesg(lmp,"In step {}, read pair stress is {}, calculated pair stress sum is {}, read gas shell energy is {}, calculated gas shell energyis {}\n",update->ntimestep,shellstress, pair_stress_sum,readgasshellenergy,gasshellenergy);
  region->lost_partilces=sumlostparticles;
  region->stress=stress_all;
  region->numatoms=sumnumatoms;
  region->SL_Tblgasold=region->SL_Tblgas;
  region->SL_Tblgas=gasshellenergy*1e7/(3.0/2.0*KB*NA*sumnumatoms);

  
  //utils::logmesg(lmp,"In step {}, sumenergyloss is {}, Tblgas after update is {}\n",update->ntimestep,sumenergyloss,region->SL_Tblgas);
    //utils::logmesg(lmp,"liquid energy is {}, sumenergyloss is {}\n",liquidenergy,sumenergyloss);
  // utils::logmesg(lmp,"In step {}, fix_stress_sum is {}, kspace_stress_sum is {}, ke_stress_sum is {}, SL_Tblgas is {}, SL_Tblgasold is {} \n",update->ntimestep, fix_stress_sum, pair_stress_sum, ke_stress_sum,region->SL_Tblgas,region->SL_Tblgasold);
  Tbl_update(region->SL_radius,region->SL_lastradius,region->SL_delta,region->SL_deltaold);
  region->SL_Tblliquidold=Tblold;
  region->SL_Tblliquid=Tbl;
  region->prematch();
  if(update->ntimestep>=1000)
  {
    region->sumdiffTbl+=(Tbl-Tblold);
    region->stepnum++;
  }
  region->SL_restart=false;
  
}
/* ---------------------------------------------------------------------- */

void FixWallMDHeatbath::post_force_respa(int vflag, int ilevel, int /* iloop */)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ---------------------------------------------------------------------- */

void FixWallMDHeatbath::min_post_force(int vflag)
{
  post_force(vflag);
}

/* ----------------------------------------------------------------------
   energy of wall interaction
------------------------------------------------------------------------- */

double FixWallMDHeatbath::compute_scalar()
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall, ewall_all, 4, MPI_DOUBLE, MPI_SUM, world);
    eflag = 1;
  }
  return ewall_all[0];
}

/* ----------------------------------------------------------------------
   components of force on wall
------------------------------------------------------------------------- */

double FixWallMDHeatbath::compute_vector(int n)
{
  // only sum across procs one time

  if (eflag == 0) {
    MPI_Allreduce(ewall, ewall_all, 4, MPI_DOUBLE, MPI_SUM, world);
    eflag = 1;
  }
  return ewall_all[n + 1];
}

/* ----------------------------------------------------------------------
   LJ 9/3 interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallMDHeatbath::lj93(double r)
{
  double rinv = 1.0 / r;
  double r2inv = rinv * rinv;
  double r4inv = r2inv * r2inv;
  double r10inv = r4inv * r4inv * r2inv;
  fwall = coeff1 * r10inv - coeff2 * r4inv;
  eng = coeff3 * r4inv * r4inv * rinv - coeff4 * r2inv * rinv - offset;
}

/* ----------------------------------------------------------------------
   LJ 12/6 interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallMDHeatbath::lj126(double r)
{
  double rinv = 1.0 / r;
  double r2inv = rinv * rinv;
  double r6inv = r2inv * r2inv * r2inv;
  fwall = r6inv * (coeff1 * r6inv - coeff2) * rinv;
  eng = r6inv * (coeff3 * r6inv - coeff4) - offset;
}

/* ----------------------------------------------------------------------
   LJ 10/4/3 interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallMDHeatbath::lj1043(double r)
{
  double rinv = 1.0 / r;
  double r2inv = rinv * rinv;
  double r4inv = r2inv * r2inv;
  double r10inv = r4inv * r4inv * r2inv;
  fwall = coeff5 * r10inv * rinv - coeff6 * r4inv * rinv - coeff7 * powint(r + coeff4, -4);
  eng = coeff1 * r10inv - coeff2 * r4inv - coeff3 * powint(r + coeff4, -3) - offset;
}

/* ----------------------------------------------------------------------
   Morse interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallMDHeatbath::morse(double r)
{
  double dr = r - sigma;
  double dexp = exp(-alpha * dr);
  fwall = coeff1 * (dexp * dexp - dexp);
  eng = epsilon * (dexp * dexp - 2.0 * dexp) - offset;
}

/* ----------------------------------------------------------------------
   colloid interaction for finite-size particle of rad with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallMDHeatbath::colloid(double r, double rad)
{
  double new_coeff2 = coeff2 * rad * rad * rad;
  double diam = 2.0 * rad;

  double rad2 = rad * rad;
  double rad4 = rad2 * rad2;
  double rad8 = rad4 * rad4;
  double delta2 = rad2 - r * r;
  double rinv = 1.0 / delta2;
  double r2inv = rinv * rinv;
  double r4inv = r2inv * r2inv;
  double r8inv = r4inv * r4inv;
  fwall = coeff1 *
          (rad8 * rad + 27.0 * rad4 * rad2 * rad * r * r + 63.0 * rad4 * rad * powint(r, 4) +
           21.0 * rad2 * rad * powint(r, 6)) *
          r8inv -
      new_coeff2 * r2inv;

  double r2 = 0.5 * diam - r;
  double rinv2 = 1.0 / r2;
  double r2inv2 = rinv2 * rinv2;
  double r4inv2 = r2inv2 * r2inv2;
  double r3 = r + 0.5 * diam;
  double rinv3 = 1.0 / r3;
  double r2inv3 = rinv3 * rinv3;
  double r4inv3 = r2inv3 * r2inv3;
  eng = coeff3 *
          ((-3.5 * diam + r) * r4inv2 * r2inv2 * rinv2 +
           (3.5 * diam + r) * r4inv3 * r2inv3 * rinv3) -
      coeff4 * ((-diam * r + r2 * r3 * (log(-r2) - log(r3))) * (-rinv2) * rinv3) - offset;
}

/* ----------------------------------------------------------------------
   harmonic interaction for particle with wall
   compute eng and fwall = magnitude of wall force
------------------------------------------------------------------------- */

void FixWallMDHeatbath::harmonic(double r)
{
  double dr = cutoff - r;
  fwall = 2.0 * epsilon * dr;
  eng = epsilon * dr * dr;
}


void FixWallMDHeatbath::variable_check()
{
  if (Tblstyle == VARIABLE) {
    Tblvar=input->variable->find(Tblstr);
    if (Tblvar < 0) error->all(FLERR, "Variable {} for heat bath wall does not exist", Tblstr);
    if (!input->variable->equalstyle(Tblvar))
      error->all(FLERR, "Variable {} for  heat bath wall is invalid style", Tblstr);
  }  
  // shellpressurevar=input->variable->find(shellpressurestr);
  // if (shellpressurevar < 0) error->all(FLERR, "Variable {} for heat bath wall does not exist", shellpressurestr);

  // shellkestressvar=input->variable->find(shellkestressstr);
  // if (shellkestressvar < 0) error->all(FLERR, "Variable {} for heat bath wall does not exist", shellpressurestr);
}

void FixWallMDHeatbath::temp_update()
{
  if (Tblstyle == VARIABLE ) 
  {
    Tblold=Tbl;
    Tbl= scalefactor*input->variable->compute_equal(Tblvar);
    //utils::logmesg(lmp,"Current Temperature is {} \n",Tbl);
  }
  if (Tbl < 0.0) error->one(FLERR, "Variable evaluation in heat bath boundary gave bad value");
}


void FixWallMDHeatbath::Tbl_update(const double R, const double oldR, const double delta, const double olddelta )
{
  //all calulation based on standard units

  // oldoutshell=R+delta+dR+ddelta;
  //double newoutshell=R+delta;
  //double oldinshell= R+dR;
  //double newinshell= R;
  double stan_R=R*1e-10;
  double stan_delta=delta*1e-10;
  double stan_oldR=oldR*1e-10;
  double stan_olddelta=olddelta*1e-10;

  double oldvolume=4.0*MY_PI*(powint(stan_oldR+stan_olddelta,3)-powint(stan_oldR,3))/3.0;
  double newvolume=4.0*MY_PI*(powint(stan_R+stan_delta,3)-powint(stan_R,3))/3.0;
  //E_qold+(V-Vold)*Tinfty=E_qnew+(V-Vnew)*Tinfty  ->  E_qnew=E_qold+(VnewV-old)*Tinfty
  double shellvolume=newvolume-oldvolume;
  //double shellvolume=4.0*MY_PI*(powint(stan_R,2)*stan_dR-powint(stan_R+stan_delta,2)*(stan_dR+stan_ddelta)); //positive in the start of collapse
  double energyshell=shellvolume*density*capacity*Tinfty;  
  //energyshell=0.0;
  //double energyshell= 4.0*MY_PI*(R+delta)*(R+delta)*ddelta*density*capacity*Tinfty*1e-15;
  //energyshell-=4.0*MY_PI*R*R*dR*density*capacity*Tinfty*1e-15;

  //double benchmarkenergy=cal_liquid_energy(R);
  //utils::logmesg(lmp,"Energy Loss is {} \n",sumenergyloss);
  double old_liquidenergy=liquidenergy;
  //liquidenergy=liquidenergy+energyshell; // subtract the loss of the energy
  liquidenergy=liquidenergy+sumenergyloss+energyshell+region->acousticenergy; // subtract the loss of the energy

 
  double Tblcoeff=density*capacity*(2.0*stan_delta*MY_PI*(10.0*powint(stan_R,2) + powint(stan_delta,2) +5*stan_R*stan_delta))/15.0;
  double constcoeff=density*capacity*(2.0*stan_delta*MY_PI*(20.0*powint(stan_R,2)*Tinfty+ 9.0*Tinfty*powint(stan_delta,2)+ 25.0*stan_R*Tinfty*stan_delta ))/15.0;
  Tblold=Tbl;

  //utils::logmesg(lmp,"In step {}, update temperature! \n",update->ntimestep);
  Tbl=(liquidenergy-constcoeff)/Tblcoeff;
  double benchmarkenergy=cal_liquid_energy(R,delta);
  //if(update->ntimestep%100000==0 || update->ntimestep%100000==1 || update->ntimestep%100000==2)
  {
    //utils::logmesg(lmp,"In step {}, Input Temperature is {} Current Temperature is {}, R is {}, oldR is {}, delta is {}, old delta is {} \n",update->ntimestep,Tblold/scalefactor,Tbl/scalefactor,R,oldR,delta,olddelta);
    //utils::logmesg(lmp, "Update Tbl is {}, old_liquidenergy is {}, liquidenergy is {}, benchmarkenergy is {}, sumenergyloss is {}, energyshell is {} \n",Tbl/scalefactor,old_liquidenergy,liquidenergy,benchmarkenergy,sumenergyloss,energyshell);
    //utils::logmesg(lmp,"ddelta is {}, vradius is {} \n",ddelta,dR);
    //utils::logmesg(lmp,"Liquid Energy is {} \n",liquidenergy);
    //utils::logmesg(lmp,"Benchmark energy is {} \n",benchmarkenergy);
    //utils::logmesg(lmp,"Energy Loss is {} \n",sumenergyloss);
    //utils::logmesg(lmp,"Current energyshell is {} \n",energyshell);
  }
  
  //Tbl=liquidenergy/(newvolume*density*capacity);

  // if(update->ntimestep%1000==0)
  // utils::logmesg(lmp,"Tbl Temperature is {}, liquid energy is {}, sumenergyloss is {}, acoustic energy is {}\n",Tbl/scalefactor,liquidenergy,sumenergyloss,region->acousticenergy);
  
}

double FixWallMDHeatbath::cal_liquid_energy(const double R, const double delta)
{
  //all calulation based on standard units

 double stan_R=R*1e-10;
 double stan_delta=delta*1e-10;
 double Tblcoeff=density*capacity*(2.0*stan_delta*MY_PI*(10.0*powint(stan_R,2) + powint(stan_delta,2) +5.0*stan_R*stan_delta))/15.0;
 double constcoeff=density*capacity*(2.0*stan_delta*MY_PI*(20.0*powint(stan_R,2)*Tinfty+ 9.0*Tinfty*powint(stan_delta,2)+ 25.0*stan_R*Tinfty*stan_delta ))/15.0;
 //utils::logmesg(lmp,"Tblcoeff is {}, constcoeff is {}\n",Tblcoeff,constcoeff);
 //double constenergy=Tbl*density*capacity*4.0*MY_PI*(powint(stan_R+stan_delta,3)-powint(stan_R,3))/3.0;

 //utils::logmesg(lmp,"CAL LIQUID ENERGY: In step {}, currrent R is {}, current delta is {}, Tbl is {}, liquid energy is {} \n",update->ntimestep,R,delta,Tbl/scalefactor,Tblcoeff*Tbl+constcoeff);
 return Tblcoeff*Tbl+constcoeff;
 //return constenergy;
}
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

#include "region_kmsphere.h"

#include "error.h"
#include "input.h"
#include "update.h"
#include "variable.h"
#include "compute.h"
#include "modify.h"
#include "math_const.h"
#include "force.h"
#include "math_special.h"

#include <cmath>

using namespace LAMMPS_NS;
using MathConst::MY_PI;
using MathSpecial::powint;

enum { CONSTANT,VARIABLE, AUTO, MIXING };

/* ---------------------------------------------------------------------- */

RegKMSphere::RegKMSphere(LAMMPS *lmp, int narg, char **arg) :
    Region(lmp, narg, arg), deltastr(nullptr)
{
  options(narg - 30, &arg[30]);
  xc = xscale * utils::numeric(FLERR, arg[2], false, lmp);
  yc = yscale * utils::numeric(FLERR, arg[3], false, lmp);
  zc = zscale * utils::numeric(FLERR, arg[4], false, lmp);
  radius = xscale * utils::numeric(FLERR, arg[5], false, lmp);
  radiusold =xscale * utils::numeric(FLERR, arg[6], false, lmp);
  vradius = xscale * utils::numeric(FLERR, arg[7], false, lmp);
  CB= xscale*utils::numeric(FLERR, arg[8], false, lmp);
  rho=utils::numeric(FLERR, arg[9], false, lmp);
  Pinfty=utils::numeric(FLERR, arg[10], false, lmp);
  sigma=utils::numeric(FLERR, arg[11], false, lmp);
  miu=utils::numeric(FLERR, arg[12], false, lmp);
  cutoff=utils::numeric(FLERR, arg[13], false, lmp);
  radius_in=radius-cutoff;
  freq=utils::numeric(FLERR, arg[14], false, lmp);
  w=2.0*MY_PI*freq;
  PA=utils::numeric(FLERR, arg[15], false, lmp)*101325.0;
  tstart=utils::numeric(FLERR, arg[16], false, lmp)*1e-10;
  scalefactor=utils::numeric(FLERR, arg[17], false, lmp);
  Tinfty=utils::numeric(FLERR, arg[18], false, lmp)*scalefactor;
  alpha1=utils::numeric(FLERR, arg[19], false, lmp);
  SL_vdelta=utils::numeric(FLERR, arg[20], false, lmp);

  th_radius=utils::numeric(FLERR, arg[23], false, lmp);
  th_vradius=utils::numeric(FLERR, arg[24], false, lmp);
  th_aradius=utils::numeric(FLERR, arg[25], false, lmp);
  th_Tb0=utils::numeric(FLERR, arg[26], false, lmp);
  th_delta=utils::numeric(FLERR, arg[27], false, lmp);

  simulate_time=utils::numeric(FLERR, arg[28], false, lmp);
  pB_old_init=utils::numeric(FLERR, arg[29], false, lmp);

  // delta variable checking
  if (utils::strmatch(arg[21], "^v_")) {
    deltastr = utils::strdup(arg[21] + 2);
    SL_delta=0.0;
    SL_deltaold=0.0;
    deltastyle=VARIABLE;
    varshape=1;
    }
  else if (utils::strmatch(arg[21], "^m_"))
  {
    deltastr = utils::strdup(arg[21] + 2);
    SL_delta=0.0;
    SL_deltaold=0.0;
    deltastyle=MIXING;
    varshape=1;
  }
  else {
      SL_delta=utils::numeric(FLERR, arg[21], false, lmp);
      SL_deltaold=SL_delta;
      deltastyle=AUTO;
    }
    //utils::logmesg(lmp,"REGION KMSPHERE. deltasytle is {} \n",deltastyle);
  SL_deltaold=utils::numeric(FLERR, arg[22], false, lmp);
  varshape=1; // enable the shape_update()
  if (varshape)
  {
    variable_check();
    if (deltastyle == VARIABLE || deltastyle==MIXING)
      delta_update();
  }


  //
  // error check

  if (radius < 0.0) error->all(FLERR, "Illegal region sphere radius: {} \n", radius);

  // extent of sphere
  // for variable radius, uses initial radius and origin for variable center

  if (interior) {
    bboxflag = 1;
    extent_xlo = xc - radius;
    extent_xhi = xc + radius;
    extent_ylo = yc - radius;
    extent_yhi = yc + radius;
    extent_zlo = zc - radius;
    extent_zhi = zc + radius;
  } else
    bboxflag = 0;

  cmax = 1;
  contact = new Contact[cmax];
  tmax = 1;

  //utils::logmesg(lmp,"REGION KMSPHERE, Initial radius is {}\n",radius);
}

/* ---------------------------------------------------------------------- */

RegKMSphere::~RegKMSphere()
{
  delete[] deltastr;
  delete[] contact;
}

/* ---------------------------------------------------------------------- */

void RegKMSphere::init()
{
  
  Region::init();
  SL_radius=radius;
  SL_lastradius=radiusold;
  SL_vradius=vradius;
  SL_debug=0.0;
  pB=0.0;
  pB_old=0.0;
  stepnum=0;
  sumdiffTbl=0.0;
  //simulate_time=0.0;
  SL_steps=simulate_time*1e15;
  //utils::logmesg(lmp,"Shouldn't calll it ,multi times! \n");
  SL_restart=SL_steps>1e-6;
  // SL_restart=true;

  // Theoty initialization
  // th_radius=3.086969205287614e-06;
  // th_vradius=-3.162987025445613e+02;
  // th_aradius=-4.094560456322810e+10;
  // th_Tb0=2.530107262855020e+03;
  // th_delta=2.520848698181851e-07;
  update_t=tstart+simulate_time;    // standard unit

  th_Tinfty=300.0;
  th_k1=0.598;
  th_mygamma=5.0/3.0;
  th_A=2.682e-5;
  th_B=1.346e-2;
  th_initialR=4.5e-6;
  th_NBC=1.316;
  th_Pinfty=101325;
  th_initialrhog=1.603;
  double gas_mass=(4.0/3.0) * MY_PI * powint(th_initialR,3) * th_initialrhog;
  th_a=(4.0/3.0) * MY_PI * powint(th_initialR,3) * th_initialrhog * 5.0 / (4.0 * MY_PI) * (1.0 - th_NBC);
  th_c=(gas_mass-(4.0*MY_PI/5.0)*th_a)*(3.0/(4.0*MY_PI));
  acousticenergy=cal_acousticenergy(update_t,SL_radius,SL_vradius);
  modify->addstep_compute_all(update->ntimestep);

  // hard coding the input no
  if (varshape) variable_check();
  
}

/* ----------------------------------------------------------------------
   inside = 1 if x,y,z is inside or on surface
   inside = 0 if x,y,z is outside and not on surface
------------------------------------------------------------------------- */

int RegKMSphere::inside(double x, double y, double z)
{
  double delx = x - xc;
  double dely = y - yc;
  double delz = z - zc;
  double r = sqrt(delx * delx + dely * dely + delz * delz);

  if (r <= radius) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from inner surface of sphere
   no contact if outside (possible if called from union/intersect)
   delxyz = vector from nearest point on sphere to x
   special case: no contact if x is at center of sphere
------------------------------------------------------------------------- */

int RegKMSphere::surface_interior(double *x, double cutoff)
{
  double delx = x[0] - xc;
  double dely = x[1] - yc;
  double delz = x[2] - zc;
  double r = sqrt(delx * delx + dely * dely + delz * delz);
  if (r > radius || r == 0.0) return 0;
  
  double delta = radius - r;
  if (delta < cutoff) {
    contact[0].r = delta;
    contact[0].delx = delx * (1.0 - radius / r);
    contact[0].dely = dely * (1.0 - radius / r);
    contact[0].delz = delz * (1.0 - radius / r);
    contact[0].radius = -radius;
    contact[0].iwall = 0;
    contact[0].varflag = 1;
    //utils::logmesg(lmp,"Surface Interior DEBUG: Step {}, x {}, y {}, z {}, delx {}, radius {}, r {}\n",update->ntimestep,x[0],x[1],x[2],delx,radius,r);
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from outer surface of sphere
   no contact if inside (possible if called from union/intersect)
   delxyz = vector from nearest point on sphere to x
------------------------------------------------------------------------- */

int RegKMSphere::surface_exterior(double *x, double cutoff)
{
  double delx = x[0] - xc;
  double dely = x[1] - yc;
  double delz = x[2] - zc;
  double r = sqrt(delx * delx + dely * dely + delz * delz);
  if (r < radius) return 0;

  double delta = r - radius;
  if (delta < cutoff) {
    contact[0].r = delta;
    contact[0].delx = delx * (1.0 - radius / r);
    contact[0].dely = dely * (1.0 - radius / r);
    contact[0].delz = delz * (1.0 - radius / r);
    contact[0].radius = radius;
    contact[0].iwall = 0;
    contact[0].varflag = 1;
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   change region shape via variable evaluation
------------------------------------------------------------------------- */

void RegKMSphere::shape_update()
{
  //radius=radius+vradius*update->dt;
  modify->addstep_compute_all(update->ntimestep+1);
  radius_in=radius-cutoff;
  volume=4.0*MY_PI/3.0*(radius*radius*radius-radius_in*radius_in*radius_in);
  if(SL_restart)
  {
    pb=-stress/(3.0*volume)*101325;
    pB_old=pB_old_init;
  }
  else
  {
    pB_old=pB;
    pb=-stress/(3.0*volume)*101325;
  }
  pB=pb-(2*sigma/(radius*1e-10)+4.0*miu*(vradius*1e5)/(radius*1e-10));

  // // log down the old pressure
  // if(abs(stress)>1e-6) 
  // {
  //   if(abs(pB)>1e-6 || update->ntimestep==0) 
  //   {
  //     pB_old=pB;
  //     pb=-stress/(3.0*volume)*101325;
  //   }
  //   else 
  //   {
  //     pb=pb_init;
  //     pB_old=pB_old_init;
  //   }  
  //   pB=pb-(2*sigma/(radius*1e-10)+4.0*miu*(vradius*1e5)/(radius*1e-10));

  // }
  // else{
  //   pb=pb_init;
  //   pB_old=pB_old_init;
  //   pB=pb-(2*sigma/(radius*1e-10)+4.0*miu*(vradius*1e5)/(radius*1e-10));
  // }


  // invoke the RK4 update and log down the old pressure
  SL_lastradius=radius;
  //utils::logmesg(lmp, "[shape update], In step {}, pressure is {}, pressureold is {}, SL restart is {}\n", update->ntimestep, pB, pB_old,SL_restart);
  // These can ensure the consistency in restart but may cause two steps latency
  if (abs(pB_old)>1e-6)
  {
      //update the delta and radius
    switch (deltastyle) {
    case VARIABLE:
      delta_update();
      // changing the seqennce seems to give distcint results when restart?
      SLRK4(pB, pB_old);
      // SLRK4_debug(pressure, pressure_old, SL_Tblliquid, SL_Tblliquidold);
      theroy_SLRK4();
      break;

    case AUTO:
      delta_update();
      theroy_SLRK4();
      SLRK4(pB, pB_old);      
      break;

    case MIXING:
      if (update->ntimestep < 1000) {
        delta_update();
        SLRK4(pB, pB_old);
      } else {
        SL_deltaold = SL_delta;
        SL_delta = SL_delta + SL_vdelta * update->dt;

        if (stepnum >= 2) {
          double avgTbl = SL_Tblliquid;
          double avfdiffTbl = sumdiffTbl / double(stepnum);
          double avgTblold = avgTbl - avfdiffTbl;
          SLRK4(pB, pB_old, avgTbl, avgTblold);
          // utils::logmesg(lmp, "REGION KMSPHERE, In step {}, avgTbl is {}, avgTblold is {}\n", update->ntimestep, avgTbl, avgTblold);
          stepnum = 0;  // reset the stepnum
          sumdiffTbl = 0.0;  // reset the sum of diffTbl
        }
        // SL_delta = SL_delta + SL_vdelta * update->dt;
      }
      break;

    default:
      // Handle unexpected deltastyle values if necessary
      break;
    }

  }
  // else  // fix twp steps latency
  // {
  //   delta_update();
  //   theroy_SLRK4();
  //   double h=(update->dt*1e-15/force->femtosecond);
  //   // we use the old velocity to update
  //   radius-=vradius*(update->dt/force->femtosecond);
  //   simulate_time+=h;
  //   SL_steps+=update->dt;
  // }
  

  SL_radius=radius;
  SL_vradius=vradius;

  acousticenergy=cal_acousticenergy(update_t,SL_radius,SL_vradius);

}

// void RegKMSphere::shape_update()
// {
//   //radius=radius+vradius*update->dt;
//   radius_in=radius-cutoff;
//   volume=4.0*MY_PI/3.0*(radius*radius*radius-radius_in*radius_in*radius_in);
//   pressure=-stress/(3.0*volume)*101325;
//   pb=pressure;
//   pressure-=2*sigma/(radius*1e-10)+4.0*miu*(vradius*1e5)/(radius*1e-10);

//   // invoke the RK4 update and log down the old pressure
//   SL_lastradius=radius;

//   if (update->ntimestep>0)
//   {
//       //update the delta and radius
//     switch (deltastyle) {
//     case VARIABLE:
//       delta_update();
//       SLRK4(pressure, pressure_old);
//       // SLRK4_debug(pressure, pressure_old, SL_Tblliquid, SL_Tblliquidold);
//       theroy_SLRK4();
//       break;

//     case AUTO:
//       delta_update();
//       SLRK4(pressure, pressure_old);
//       theroy_SLRK4();
//       break;

//     case MIXING:
//       if (update->ntimestep < 1000) {
//         delta_update();
//         SLRK4(pressure, pressure_old);
//       } else {
//         SL_deltaold = SL_delta;
//         SL_delta = SL_delta + SL_vdelta * update->dt;

//         if (stepnum >= 2) {
//           double avgTbl = SL_Tblliquid;
//           double avfdiffTbl = sumdiffTbl / double(stepnum);
//           double avgTblold = avgTbl - avfdiffTbl;
//           SLRK4(pressure, pressure_old, avgTbl, avgTblold);
//           // utils::logmesg(lmp, "REGION KMSPHERE, In step {}, avgTbl is {}, avgTblold is {}\n", update->ntimestep, avgTbl, avgTblold);
//           stepnum = 0;  // reset the stepnum
//           sumdiffTbl = 0.0;  // reset the sum of diffTbl
//         }
//         // SL_delta = SL_delta + SL_vdelta * update->dt;
//       }
//       break;

//     default:
//       // Handle unexpected deltastyle values if necessary
//       break;
//     }

//   }
  
//   // log down the old pressure
//   pressure_old=pressure;
//   SL_radius=radius;
//   SL_vradius=vradius;

// }

/* ----------------------------------------------------------------------
   error check on existence of variable
------------------------------------------------------------------------- */

void RegKMSphere::variable_check()
{
  if (deltastyle == VARIABLE || deltastyle==MIXING) {
    deltavar=input->variable->find(deltastr);
    if (deltavar < 0) error->all(FLERR, "Variable {} for heat bath wall does not exist", deltastr);
    if (!input->variable->equalstyle(deltavar))
      error->all(FLERR, "Variable {} for  heat bath wall is invalid style", deltastr);
  }

}

/* ----------------------------------------------------------------------
   Set values needed to calculate velocity due to shape changes.
   These values do not depend on the contact, so this function is
   called once per timestep by fix/wall/gran/region.

------------------------------------------------------------------------- */

void RegKMSphere::set_velocity_shape()
{
  xcenter[0] = xc;
  xcenter[1] = yc;
  xcenter[2] = zc;
  forward_transform(xcenter[0], xcenter[1], xcenter[2]);
  if (update->ntimestep > 0)
    rprev = prev[4];
  else
    rprev = radius;
  prev[4] = radius;
}

/* ----------------------------------------------------------------------
   add velocity due to shape change to wall velocity
------------------------------------------------------------------------- */

void RegKMSphere::velocity_contact_shape(double *vwall, double *xc)
{
  double delx, dely, delz;    // Displacement of contact point in x,y,z

  delx = (xc[0] - xcenter[0]) * (1 - rprev / radius);
  dely = (xc[1] - xcenter[1]) * (1 - rprev / radius);
  delz = (xc[2] - xcenter[2]) * (1 - rprev / radius);

  vwall[0] += delx / update->dt;
  vwall[1] += dely / update->dt;
  vwall[2] += delz / update->dt;
}


void RegKMSphere::SLRK4(const double pressure,const double pressure_old)
{
  // y1: position
  // y2: velocity
  // dy1/dt=y2
  // dy2/dt=update_aradius
  
  double h=(update->dt*1e-15/force->femtosecond);
  double current_t=simulate_time+tstart;
  double diffpress=(pressure-pressure_old)/h;
  // if(update->ntimestep%100000==0)
  //utils::logmesg(lmp,"[SLRK4] ,In step {}, SLRK4, Input radius is {}, vradius is {}, pressure is {}, pressure_old is {}, diffpress is {}\n",update->ntimestep,radius,vradius,pressure, pressure_old,diffpress);
  double k1y1=vradius*1e5;   // standard unit
  double k1y2=calf2(current_t,radius*1e-10,vradius*1e5,pressure,pressure_old);

  // h*k1y2 -> standard unit
  double k2y1=vradius*1e5+(h*k1y2/2.0);
  double k2y2=calf2(current_t+h/2.0,radius*1e-10+h/2.0*k1y1,vradius*1e5+h/2.0*k1y2,pressure,pressure_old);

  double k3y1=vradius*1e5+(h*k2y2/2.0);
  double k3y2=calf2(current_t+h/2.0,radius*1e-10+h/2.0*k2y1,vradius*1e5+h/2.0*k2y2,pressure,pressure_old);

  double k4y1=vradius*1e5+(h*k3y1);
  double k4y2=calf2(current_t+h,radius*1e-10+h*k3y1,vradius*1e5+h*k3y2,pressure,pressure_old);
  
  radius=radius+h/6.0*(k1y1+2.0*k2y1+2.0*k3y1+k4y1)*1e10;
  vradius=vradius+h/6.0*(k1y2+2.0*k2y2+2.0*k3y2+k4y2)*1e-5;
  simulate_time+=h;
  SL_steps+=update->dt;
  //utils::logmesg(lmp,"In step {}, SLRK4, OUTPUT radius is {}, vradius is {}\n",update->ntimestep , radius,vradius);
  //double debugvradius=1.0/6.0*(k1y1+2.0*k2y1+2.0*k3y1+k4y1)*1e-5;
  //utils::logmesg(lmp,"debugvradius is {},calculated vradius is {} \n",debugvradius,vradius);
  //utils::logmesg(lmp,"k1y1 is {}, k2y1 is {}, k3y1 is {}, k4y1 is {}, calculated vradius is {} \n",k1y1,k2y1,k3y1,k4y1,vradius);
  //utils::logmesg(lmp,"k1y2 is {}, k2y2 is {}, k3y2 is {}, k4y2 is {} \n",k1y2,k2y2,k3y2,k4y2);

}

double RegKMSphere::calf2(double update_t, double update_radius, double update_vradius, double pressure, double pressure_old)
{
  // the units of the input should be standard
  double h=(update->dt*1e-15/force->femtosecond);
  double diffpress=(pressure-pressure_old)/h;
  double f2=-((3.0*powint(update_vradius,2)*(update_vradius/(3.0*CB) - 1.0))/2.0 - ((update_vradius/CB + 1)*(Pinfty - pressure - PA*sin(w*(update_t + update_radius/CB)) \
  + (2.0*sigma)/update_radius + (4.0*update_vradius*miu)/update_radius) - (update_radius*(diffpress + (2.0*update_vradius*sigma)/powint(update_radius,2) + (4.0*powint(update_vradius,2)*miu)/powint(update_radius,2) \
  + PA*w*cos(w*(update_t + update_radius/CB))*(update_vradius/CB + 1)))/CB)/rho)\
  /(update_radius*(update_vradius/CB - 1.0) - (4.0*miu)/(CB*rho));

  //return standard output
  return f2;
}

void RegKMSphere::SLRK4(const double pressure,const double pressure_old, const double Tbl, const double Tbl_old)
{
  // y1: position
  // y2: velocity
  // dy1/dt=y2
  // dy2/dt=update_aradius

  double h=(update->dt*1e-15/force->femtosecond);
  double current_t=simulate_time+tstart;
  double diffpress=(pressure-pressure_old)/h;
  double diffTbl=(Tbl-Tbl_old)/h;
  double k1y1=vradius*1e5;   // standard unit
  double k1y2=calf2(current_t,radius*1e-10,vradius*1e5,pressure,pressure_old);
  double k1y3=calf3(current_t,radius*1e-10,vradius*1e5,SL_delta*1e-10,Tbl,Tbl_old);
  
  // h*k1y2 -> standard unit
  double k2y1=vradius*1e5+(h*k1y2/2.0);
  double k2y2=calf2(current_t+h/2.0,radius*1e-10+h/2.0*k1y1,vradius*1e5+h/2.0*k1y2,pressure,pressure_old);
  double k2y3=calf3(current_t+h/2.0,radius*1e-10+h/2.0*k1y1,vradius*1e5+h/2.0*k1y2,SL_delta*1e-10+h/2.0*k1y3,Tbl,Tbl_old);

  double k3y1=vradius*1e5+(h*k2y2/2.0);
  double k3y2=calf2(current_t+h/2.0,radius*1e-10+h/2.0*k2y1,vradius*1e5+h/2.0*k2y2,pressure,pressure_old);
  double k3y3=calf3(current_t+h/2.0,radius*1e-10+h/2.0*k2y1,vradius*1e5+h/2.0*k2y2,SL_delta*1e-10+h/2.0*k2y3,Tbl,Tbl_old);

  double k4y1=vradius*1e5+(h*k3y1);
  double k4y2=calf2(current_t+h,radius*1e-10+h*k3y1,vradius*1e5+h*k3y2,pressure,pressure_old);
  double k4y3=calf3(current_t+h,radius*1e-10+h*k3y1,vradius*1e5+h*k3y2,SL_delta*1e-10+h*k3y3,Tbl,Tbl_old);
  
  radius=radius+h/6.0*(k1y1+2.0*k2y1+2.0*k3y1+k4y1)*1e10;
  vradius=vradius+h/6.0*(k1y2+2.0*k2y2+2.0*k3y2+k4y2)*1e-5;
  // SL_vdelta=std::min(h/6.0*(k1y3+2.0*k2y3+2.0*k3y3+k4y3)*1e10,SL_vdelta);
  SL_vdelta=h/6.0*(k1y3+2.0*k2y3+2.0*k3y3+k4y3)*1e10;
  simulate_time+=h;
  SL_steps+=update->dt;
  //double debugvradius=1.0/6.0*(k1y1+2.0*k2y1+2.0*k3y1+k4y1)*1e-5;
  //utils::logmesg(lmp,"debugvradius is {},calculated vradius is {} \n",debugvradius,vradius);
  //utils::logmesg(lmp,"k1y1 is {}, k2y1 is {}, k3y1 is {}, k4y1 is {}, calculated vradius is {} \n",k1y1,k2y1,k3y1,k4y1,vradius);
  //utils::logmesg(lmp,"k1y2 is {}, k2y2 is {}, k3y2 is {}, k4y2 is {} \n",k1y2,k2y2,k3y2,k4y2);
  //if(update->ntimestep%1000==0)
  //utils::logmesg(lmp,"k1y3 is {}, k2y3 is {}, k3y3 is {}, k4y3 is {}, Tbl is {}, Tblold is {}, delta is {} \n",k1y3,k2y3,k3y3,k4y3,Tbl,Tbl_old,SL_delta);

}

double RegKMSphere::calf3(double update_t, double update_radius, double update_vradius, const double update_delta, const double Tbl, const double Tbl_old)
{
  // the units of the input should be standard
  double h=(update->dt*1e-15/force->femtosecond);
  double diffpress=(pB-pB_old)/h;
  double diffTbl=(Tbl-Tbl_old)/h;
  double denominator=1.0+update_delta/update_radius+3.0/10.0*powint(update_delta/update_radius,2);
  double numerator=6*alpha1/update_delta-(2.0*update_delta/update_radius+1.0/2.0*powint((update_delta/update_radius),2))*update_vradius-update_delta*(1.0+update_delta/(2.0*update_radius)+1.0/10.0*powint((update_delta/update_radius),2))*1.0/(Tbl-Tinfty)*diffTbl;
  double f3=numerator/denominator;
  //if(update->ntimestep%1000==0)
  //utils::logmesg(lmp,"numerator is {}, denominator is {}, Tbl is {}, Tblold is {}, delta is {}, deltaold is {} \n",numerator,denominator,Tbl,Tbl_old,SL_delta,SL_deltaold);
  //return standard output
  return f3;
}

void RegKMSphere::delta_update()
{
  //utils::logmesg(lmp,"Deltastr is {}\n",deltastr);
  //utils::logmesg(lmp,"Delta update begins! Delta style is Constant {} Variable {} Auto {}\n",deltastyle==CONSTANT,deltastyle==VARIABLE,deltastyle==AUTO);
  if (deltastyle == VARIABLE || deltastyle==MIXING) {
    SL_deltaold=SL_delta;
    SL_delta= input->variable->compute_equal(deltavar);
    //SL_vdelta=SL_delta-SL_deltaold;
    //utils::logmesg(lmp,"In step {}, Current delta is {} \n",update->ntimestep,SL_delta);
  }
  else{
     // In Auto Mode, delta is borrowed from the thery_SLRK4()
     SL_deltaold=SL_delta;
     SL_delta=th_delta*1e10;
     SL_vdelta=(SL_delta-SL_deltaold)/update->dt;
     //if(update->ntimestep%1==0)
      //utils::logmesg(lmp,"In step {}, Current delta is {} \n",update->ntimestep,SL_delta);

  }
  if (SL_delta < 0.0) error->one(FLERR, "Variable delta evaluation in heat bath boundary gave bad value");
  
}

void RegKMSphere::SLRK4_debug(const double pressure,const double pressure_old, const double Tbl, const double Tbl_old)
{
  // y1: position
  // y2: velocity
  // dy1/dt=y2
  // dy2/dt=update_aradius

  double h=(update->dt*1e-15/force->femtosecond);
  double current_t=tstart+(update->ntimestep*update->dt)*1e-15/force->femtosecond;
  double diffpress=(pressure-pressure_old)/h;
  double diffTbl=(Tbl-Tbl_old)/h;
  double k1y1=vradius*1e5;   // standard unit
  double k1y2=calf2(current_t,radius*1e-10,vradius*1e5,pressure,pressure_old);
  double k1y3=calf3(current_t,radius*1e-10,vradius*1e5,SL_delta*1e-10,Tbl,Tbl_old);
  
  // h*k1y2 -> standard unit
  double k2y1=vradius*1e5+(h*k1y2/2.0);
  double k2y2=calf2(current_t+h/2.0,radius*1e-10+h/2.0*k1y1,vradius*1e5+h/2.0*k1y2,pressure,pressure_old);
  double k2y3=calf3(current_t+h/2.0,radius*1e-10+h/2.0*k1y1,vradius*1e5+h/2.0*k1y2,SL_delta*1e-10+h/2.0*k1y3,Tbl,Tbl_old);

  double k3y1=vradius*1e5+(h*k2y2/2.0);
  double k3y2=calf2(current_t+h/2.0,radius*1e-10+h/2.0*k2y1,vradius*1e5+h/2.0*k2y2,pressure,pressure_old);
  double k3y3=calf3(current_t+h/2.0,radius*1e-10+h/2.0*k2y1,vradius*1e5+h/2.0*k2y2,SL_delta*1e-10+h/2.0*k2y3,Tbl,Tbl_old);

  double k4y1=vradius*1e5+(h*k3y1);
  double k4y2=calf2(current_t+h,radius*1e-10+h*k3y1,vradius*1e5+h*k3y2,pressure,pressure_old);
  double k4y3=calf3(current_t+h,radius*1e-10+h*k3y1,vradius*1e5+h*k3y2,SL_delta*1e-10+h*k3y3,Tbl,Tbl_old);
  
  SL_debug=h/6.0*(k1y3+2.0*k2y3+2.0*k3y3+k4y3)*1e10;
  //double debugvradius=1.0/6.0*(k1y1+2.0*k2y1+2.0*k3y1+k4y1)*1e-5;
  //utils::logmesg(lmp,"debugvradius is {},calculated vradius is {} \n",debugvradius,vradius);
  //utils::logmesg(lmp,"k1y1 is {}, k2y1 is {}, k3y1 is {}, k4y1 is {}, calculated vradius is {} \n",k1y1,k2y1,k3y1,k4y1,vradius);
  //utils::logmesg(lmp,"k1y2 is {}, k2y2 is {}, k3y2 is {}, k4y2 is {} \n",k1y2,k2y2,k3y2,k4y2);
  //if(update->ntimestep%1000==0)
  //utils::logmesg(lmp,"k1y3 is {}, k2y3 is {}, k3y3 is {}, k4y3 is {}, Tbl is {}, Tblold is {}, delta is {} \n",k1y3,k2y3,k3y3,k4y3,Tbl,Tbl_old,SL_delta);

}

double RegKMSphere::calTb0(double update_t, double update_radius,double update_vradius, double update_Tb0, double update_delta)
{
  //double temp=(powint((th_k1*update_radius)/(th_B*update_delta) + 1.0,2) + (2.0*th_A*(update_Tb0 + (th_A*powint(th_Tb0,2))/(2.0*th_B) + (th_Tinfty*th_k1*update_radius)/(th_B*update_delta)))/th_B);
  //return (th_Tinfty*th_k1*powint(update_radius,2)*(6.0*th_mygamma - 6.0)*(th_Tinfty + (th_B*((th_k1*update_radius)/(th_B*update_delta) + 1.0))/th_A - (th_B*sqrt(temp))/th_A))/(th_NBC*Pinfty*powint(th_initialR,3)*update_delta) - (update_vradius*update_Tb0*(3.0*th_mygamma - 3.0))/update_radius;
  return (th_Tinfty*th_k1*powint(update_radius,2)*(6.0*th_mygamma - 6.0)*(th_Tinfty + (th_B*((th_k1*update_radius)/(th_B*update_delta) + 1.0))/th_A - (th_B*sqrt(powint(((th_k1*update_radius)/(th_B*update_delta) + 1.0),2) + (2.0*th_A*(update_Tb0 + (th_A*powint(update_Tb0,2))/(2.0*th_B) + (th_Tinfty*th_k1*update_radius)/(th_B*update_delta)))/th_B))/th_A))/(th_NBC*th_Pinfty*powint(th_initialR,3)*update_delta) - (update_vradius*update_Tb0*(3.0*th_mygamma - 3.0))/update_radius;
}

double RegKMSphere::calaR(double update_t, double update_radius,double update_vradius, double update_aradius,double update_Tb0, double update_delta, double update_dTb0)
{
  //return (CB*rho*((3.0*powint(update_vradius,2)*(update_radius/(3.0*CB) - 1.0))/2.0 - ((update_radius/CB + 1.0)*(th_Pinfty - PA*sin(w*(update_t + update_radius/CB)) + (2.0*sigma)/update_radius + (4.0*update_vradius*miu)/update_radius + update_aradius*update_radius*(th_a/(4.0*powint(update_radius,3)) + th_c/(2.0*powint(update_radius,3))) - (th_NBC*th_Pinfty*powint(th_initialR,3)*update_Tb0)/(th_Tinfty*powint(update_radius,3))) - (update_radius*((2.0*update_vradius*sigma)/powint(update_radius,2) - (4.0*update_aradius*miu)/update_radius - update_vradius*update_aradius*(th_a/(4.0*powint(update_radius,3)) + th_c/(2.0*powint(update_radius,3))) + update_aradius*update_radius*((3.0*th_a*update_vradius)/(4.0*powint(update_radius,4)) + (3.0*th_c*update_vradius)/(2.0*powint(update_radius,4))) + (4.0*powint(update_vradius,2)*miu)/powint(update_radius,2) + PA*w*cos(w*(update_t + update_radius/CB))*(update_vradius/CB + 1.0) + (th_NBC*Pinfty*powint(th_initialR,3)*update_dTb0)/(th_Tinfty*powint(update_radius,3)) - (3.0*th_NBC*Pinfty*update_vradius*powint(th_initialR,3)*update_Tb0)/(th_Tinfty*powint(update_radius,4))))/CB)/rho + update_aradius*update_radius*(update_vradius/CB - 1.0)))/(powint(update_radius,2)*(th_a/(4.0*powint(update_radius,3)) + th_c/(2.0*powint(update_radius,3))));
  return (CB*rho*((3.0*powint(update_vradius,2)*(update_vradius/(3.0*CB) - 1.0))/2.0 - ((update_vradius/CB + 1.0)*(th_Pinfty - PA*sin(w*(update_t + update_radius/CB)) + (2.0*sigma)/update_radius + (4.0*update_vradius*miu)/update_radius + update_aradius*update_radius*(th_a/(4.0*powint(update_radius,3)) + th_c/(2.0*powint(update_radius,3))) - (th_NBC*th_Pinfty*powint(th_initialR,3)*update_Tb0)/(th_Tinfty*powint(update_radius,3))) - (update_radius*((2.0*update_vradius*sigma)/powint(update_radius,2) - (4.0*update_aradius*miu)/update_radius - update_vradius*update_aradius*(th_a/(4.0*powint(update_radius,3)) + th_c/(2.0*powint(update_radius,3))) + update_aradius*update_radius*((3.0*th_a*update_vradius)/(4.0*powint(update_radius,4)) + (3.0*th_c*update_vradius)/(2.0*powint(update_radius,4))) + (4.0*powint(update_vradius,2)*miu)/powint(update_radius,2) + PA*w*cos(w*(update_t + update_radius/CB))*(update_vradius/CB + 1.0) + (th_NBC*th_Pinfty*powint(th_initialR,3)*update_dTb0)/(th_Tinfty*powint(update_radius,3)) - (3.0*th_NBC*th_Pinfty*update_vradius*powint(th_initialR,3)*update_Tb0)/(th_Tinfty*powint(update_radius,4))))/CB)/rho + update_aradius*update_radius*(update_vradius/CB - 1.0)))/(powint(update_radius,2)*(th_a/(4.0*powint(update_radius,3)) + th_c/(2.0*powint(update_radius,3))));
}

double RegKMSphere::caldelta(double update_t, double update_radius,double update_vradius,double update_aradius, double update_Tb0, double update_delta,double update_dTb0, double update_eta, double update_Tbll)
{
 return -(update_vradius*((2.0*update_delta)/update_radius + powint(update_delta,2)/(2.0*powint(update_radius,2))) - (6.0*alpha1)/update_delta + (update_delta*((update_vradius*th_k1)/(th_A*update_delta) - (th_B*((2.0*th_A*(update_dTb0 + (th_A*update_dTb0*update_Tb0)/th_B + (th_Tinfty*update_vradius*th_k1)/(th_B*update_delta)))/th_B + (2.0*update_vradius*th_k1*((th_k1*update_radius)/(th_B*update_delta) + 1.0))/(th_B*update_delta)))/(2.0*th_A*sqrt(powint(((th_k1*update_radius)/(th_B*update_delta) + 1.0),2) + (2.0*th_A*(update_Tb0 + (th_A*powint(update_Tb0,2))/(2.0*th_B) + (th_Tinfty*th_k1*update_radius)/(th_B*update_delta)))/th_B)))*(update_delta/(2.0*update_radius) + powint(update_delta,2)/(10.0*powint(update_radius,2)) + 1.0))/(th_Tinfty + (th_B*((th_k1*update_radius)/(th_B*update_delta) + 1.0))/th_A - (th_B*sqrt(powint((th_k1*update_radius)/(th_B*update_delta) + 1.0,2) + (2.0*th_A*(update_Tb0 + (th_A*powint(update_Tb0,2))/(2.0*th_B) + (th_Tinfty*th_k1*update_radius)/(th_B*update_delta)))/th_B))/th_A))/(update_delta/update_radius + (3.0*powint(update_delta,2))/(10.0*powint(update_radius,2)) + (update_delta*((th_B*((2.0*th_k1*update_radius*((th_k1*update_radius)/(th_B*update_delta) + 1.0))/(th_B*powint(update_delta,2)) + (2.0*th_A*th_Tinfty*th_k1*update_radius)/(powint(th_B,2)*powint(update_delta,2))))/(2.0*th_A*sqrt(powint((th_k1*update_radius)/(th_B*update_delta) + 1.0,2) + (2.0*th_A*(update_Tb0 + (th_A*powint(update_Tb0,2))/(2.0*th_B) + (th_Tinfty*th_k1*update_radius)/(th_B*update_delta)))/th_B)) - (th_k1*update_radius)/(th_A*powint(update_delta,2)))*(update_delta/(2.0*update_radius) + powint(update_delta,2)/(10.0*powint(update_radius,2)) + 1.0))/(th_Tinfty + (th_B*((th_k1*update_radius)/(th_B*update_delta) + 1.0))/th_A - (th_B*sqrt(powint((th_k1*update_radius)/(th_B*update_delta) + 1.0,2) + (2.0*th_A*(update_Tb0 + (th_A*powint(update_Tb0,2))/(2.0*th_B) + (th_Tinfty*th_k1*update_radius)/(th_B*update_delta)))/th_B))/th_A) + 1.0);
}

void RegKMSphere::theroy_SLRK4()
{
  double h=(update->dt*1e-15/force->femtosecond);
  //double current_t=(tstart+update->ntimestep*update->dt)*1e-15/force->femtosecond;
  double update_eta=0.0;
  double update_Tbll=0.0;
  
  double k1y0=th_radius;
  double k1y1=th_vradius;
  double k1y2=th_aradius;
  double k1y4=calTb0(update_t,th_radius,th_vradius,th_Tb0,th_delta);
  double k1y3=calaR(update_t,th_radius,th_vradius,th_aradius,th_Tb0,th_delta,k1y4);
  update_eta=(th_radius/th_delta)*(th_k1/th_B);
  update_Tbll = -th_B/th_A*(1.0 + update_eta) + th_B/th_A*sqrt(powint((1.0 + update_eta),2) + 2.0*th_A/th_B*(th_Tb0 + th_A/(2.0*th_B)*powint(th_Tb0,2) + update_eta*th_Tinfty));
  double k1y5=caldelta(update_t,th_radius,th_vradius,th_aradius,th_Tb0,th_delta,k1y4,update_eta,update_Tbll);
  
  double k2y0=th_radius+h/2.0*k1y1;
  double k2y1=th_vradius+h/2.0*k1y2;
  double k2y2=th_aradius+h/2.0*k1y3;
  double k2y4=calTb0(update_t+h/2.0,k2y0,k2y1,th_Tb0+h/2.0*k1y4,th_delta+h/2.0*k1y5);
  double k2y3=calaR(update_t+h/2.0,k2y0,k2y1,k2y2,th_Tb0+h/2.0*k1y4,th_delta+h/2.0*k1y5,k2y4);
  update_eta=(k2y0/(th_delta+h/2.0*k1y5))*(th_k1/th_B);
  update_Tbll = -th_B/th_A*(1.0 + update_eta) + th_B/th_A*sqrt(powint((1.0 + update_eta),2) + 2.0*th_A/th_B*(th_Tb0+h/2.0*k1y4 + th_A/(2.0*th_B)*powint(th_Tb0+h/2.0*k1y4,2) + update_eta*th_Tinfty));
  double k2y5=caldelta(update_t+h/2.0,k2y0,k2y1,k2y2,th_Tb0+h/2.0*k1y4,th_delta+h/2.0*k1y5,k2y4,update_eta,update_Tbll);

  double k3y0=th_radius+h/2.0*k2y1;
  double k3y1=th_vradius+h/2.0*k2y2;
  double k3y2=th_aradius+h/2.0*k2y3;
  double k3y4=calTb0(update_t+h/2.0,k3y0,k3y1,th_Tb0+h/2.0*k2y4,th_delta+h/2.0*k2y5);
  double k3y3=calaR(update_t+h/2.0,k3y0,k3y1,k3y2,th_Tb0+h/2.0*k2y4,th_delta+h/2.0*k2y5,k3y4);
  update_eta=(k3y0/(th_delta+h/2.0*k2y5))*(th_k1/th_B);
  update_Tbll = -th_B/th_A*(1.0 + update_eta) + th_B/th_A*sqrt(powint((1.0 + update_eta),2) + 2.0*th_A/th_B*(th_Tb0+h/2.0*k2y4 + th_A/(2.0*th_B)*powint(th_Tb0+h/2.0*k2y4,2) + update_eta*th_Tinfty));
  double k3y5=caldelta(update_t+h/2.0,k3y0,k3y1,k3y2,th_Tb0+h/2.0*k2y4,th_delta+h/2.0*k2y5,k3y4,update_eta,update_Tbll);

  double k4y0=th_radius+h*k3y1;
  double k4y1=th_vradius+h*k3y2;
  double k4y2=th_aradius+h*k3y3;
  double k4y4=calTb0(update_t+h,k4y0,k4y1,th_Tb0+h*k3y4,th_delta+h*k3y5);
  double k4y3=calaR(update_t+h,k4y0,k4y1,k4y2,th_Tb0+h*k3y4,th_delta+h*k3y5,k4y4);
  update_eta=(k4y0/(th_delta+h*k3y5))*(th_k1/th_B);
  update_Tbll = -th_B/th_A*(1.0 + update_eta) + th_B/th_A*sqrt(powint((1.0 + update_eta),2) + 2.0*th_A/th_B*(th_Tb0+h*k3y4 + th_A/(2.0*th_B)*powint(th_Tb0+h*k3y4,2) + update_eta*th_Tinfty));
  double k4y5=caldelta(update_t+h,k4y0,k4y1,k4y2,th_Tb0+h*k3y4,th_delta+h*k3y5,k4y4,update_eta,update_Tbll);

  update_t+=h;
  th_radius+=h/6.0*(k1y1+2.0*k2y1+2.0*k3y1+k4y1);
  th_vradius+=h/6.0*(k1y2+2.0*k2y2+2.0*k3y2+k4y2);
  th_aradius+=h/6.0*(k1y3+2.0*k2y3+2.0*k3y3+k4y3);
  th_Tb0+=h/6.0*(k1y4+2.0*k2y4+2.0*k3y4+k4y4);
  th_delta+=h/6.0*(k1y5+2.0*k2y5+2.0*k3y5+k4y5);

  SL_debug=th_radius*1e10;

  //if(update->ntimestep%1==0)
    //utils::logmesg(lmp,"In step {}, k1y3 is {}, k2y3 is {}, k3y3 is {}, k4y3 is {} \n",update->ntimestep,k1y3,k2y3,k3y3,k4y3);

  //calaR ?
}

double RegKMSphere::cal_acousticenergy(double current_t, double radius,double vradius)
{

  double Pa= -PA*sin(w*current_t);
  double dVdt = 4.0 * M_PI * radius * radius * vradius *1e-15;
  double h=(update->dt*1e-15/force->femtosecond);
  double E_acoustic = Pa * dVdt * h;
  return E_acoustic;

}

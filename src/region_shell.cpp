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

#include "region_shell.h"

#include "error.h"
#include "input.h"
#include "update.h"
#include "variable.h"
#include "compute.h"
#include "modify.h"

#include <cmath>

using namespace LAMMPS_NS;

enum { CONSTANT, VARIABLE };

/* ---------------------------------------------------------------------- */

RegShell::RegShell(LAMMPS *lmp, int narg, char **arg) :
    Region(lmp, narg, arg), xstr(nullptr), ystr(nullptr), zstr(nullptr), sstr(nullptr)
{
  options(narg - 14, &arg[14]);
  xc = xscale * utils::numeric(FLERR, arg[2], false, lmp);
  yc = yscale * utils::numeric(FLERR, arg[3], false, lmp);
  zc = zscale * utils::numeric(FLERR, arg[4], false, lmp);
  radius_out = xscale * utils::numeric(FLERR, arg[5], false, lmp);
  vradius = xscale * utils::numeric(FLERR, arg[6], false, lmp);
  CB= xscale*utils::numeric(FLERR, arg[7], false, lmp);
  rho=utils::numeric(FLERR, arg[8], false, lmp);
  Pinfty=utils::numeric(FLERR, arg[9], false, lmp);
  sigma=utils::numeric(FLERR, arg[10], false, lmp);
  miu=utils::numeric(FLERR, arg[11], false, lmp);
  cutoff=utils::numeric(FLERR, arg[12], false, lmp);
  radius_in=radius_out-cutoff;

  varshape = 1; //debug use, always turn on the varshape
  if (utils::strmatch(arg[13], "^v_")) {
    sstr=utils::strdup(arg[13] + 2);
    rstyle = VARIABLE;
    varshape = 1;
  } else {
    stresssum=utils::numeric(FLERR, arg[13], false, lmp);
    rstyle=CONSTANT;
  }

  if (varshape) {
    modify->addstep_compute(update->ntimestep);
    update->vflag_atom=update->ntimestep;
    //variable_check();
    //RegKMSphere::shape_update();
  }

  // error check

  if (radius_in < 0.0) error->all(FLERR, "Illegal region sphere radius: {}", radius_in);

  // extent of sphere
  // for variable radius, uses initial radius and origin for variable center

  if (interior) {
    bboxflag = 1;
    extent_xlo = xc - radius_out;
    extent_xhi = xc + radius_out;
    extent_ylo = yc - radius_out;
    extent_yhi = yc + radius_out;
    extent_zlo = zc - radius_out;
    extent_zhi = zc + radius_out;
  } else
    bboxflag = 0;
  if (interior) cmax=2;
  else cmax = 1;
  contact = new Contact[cmax];
  tmax = 1;

  utils::logmesg(lmp,"REGION SHELL, Initial radius is {} \n",radius_in);
}

/* ---------------------------------------------------------------------- */

RegShell::~RegShell()
{
  delete[] xstr;
  delete[] ystr;
  delete[] zstr;
  delete[] sstr;
  delete[] contact;
}

/* ---------------------------------------------------------------------- */

void RegShell::init()
{
  Region::init();
  //if (varshape ) variable_check();
  
}

/* ----------------------------------------------------------------------
   inside = 1 if x,y,z is inside or on surface
   inside = 0 if x,y,z is outside and not on surface
------------------------------------------------------------------------- */

int RegShell::inside(double x, double y, double z)
{
  double delx = x - xc;
  double dely = y - yc;
  double delz = z - zc;
  double r = sqrt(delx * delx + dely * dely + delz * delz);

  if (r <= radius_out && r>= radius_in) return 1;
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from inner surface of sphere
   no contact if outside (possible if called from union/intersect)
   delxyz = vector from nearest point on sphere to x
   special case: no contact if x is at center of sphere
------------------------------------------------------------------------- */

int RegShell::surface_interior(double *x, double cutoff)
{
  double delx = x[0] - xc;
  double dely = x[1] - yc;
  double delz = x[2] - zc;
  double r = sqrt(delx * delx + dely * dely + delz * delz);
  if (r >radius_out || r< radius_in) return 0;

  double delta_in = r-radius_in;
  double delta_out=radius_out-r;
  if (delta_in < cutoff) {
    contact[0].r = delta_in;
    contact[0].delx = delx * (1.0 - radius_in / r);
    contact[0].dely = dely * (1.0 - radius_in / r);
    contact[0].delz = delz * (1.0 - radius_in / r); 
    contact[0].radius = radius_in;
    contact[0].iwall = 0;
    contact[0].varflag = 1;
    return 1;
  }
  if (delta_out < cutoff) {
    contact[1].r = delta_out;
    contact[1].delx = delx * (1.0 - radius_out / r);
    contact[1].dely = dely * (1.0 - radius_out / r);
    contact[1].delz = delz * (1.0 - radius_out / r); 
    contact[1].radius = -radius_out;
    contact[1].iwall = 0;
    contact[1].varflag = 1;
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   one contact if 0 <= x < cutoff from outer surface of sphere
   no contact if inside (possible if called from union/intersect)
   delxyz = vector from nearest point on sphere to x
------------------------------------------------------------------------- */

int RegShell::surface_exterior(double *x, double cutoff)
{
  double delx = x[0] - xc;
  double dely = x[1] - yc;
  double delz = x[2] - zc;
  double r = sqrt(delx * delx + dely * dely + delz * delz);
  if (r <radius_out && r> radius_in) return 0;

  double delta_in = radius_in-r;
  double delta_out= r-radius_out;
  if (delta_in < cutoff && delta_in>0) {
    contact[0].r = delta_in;
    contact[0].delx = delx * (1.0 - radius_in / r);
    contact[0].dely = dely * (1.0 - radius_in / r);
    contact[0].delz = delz * (1.0 - radius_in / r);
    contact[0].radius = -radius_in;
    contact[0].iwall = 0;
    contact[0].varflag = 1;
    return 1;
  }
  if(delta_out<cutoff && delta_out>0)
  {
    contact[0].r = delta_out;
    contact[0].delx = delx * (1.0 - radius_out / r);
    contact[0].dely = dely * (1.0 - radius_out / r);
    contact[0].delz = delz * (1.0 - radius_out / r);
    contact[0].radius = radius_out;
    contact[0].iwall = 0;
    contact[0].varflag = 1;
    return 1;
  }
  return 0;
}

/* ----------------------------------------------------------------------
   change region shape via variable evaluation
------------------------------------------------------------------------- */

void RegShell::shape_update()
{
  
  if (update->ntimestep>10)
  {
     //variable_check();
  if (rstyle == VARIABLE) {
    modify->addstep_compute(update->ntimestep);
    update->vflag_atom=update->ntimestep;
    stresssum = input->variable->compute_equal_circular(svar);
    if (radius_in < 0.0) error->one(FLERR, "Variable evaluation in region gave bad value");
  }
  //if(update->ntimestep%1000==0)
    //utils::logmesg(lmp,"REGION KMSPHERE, STEP {} \n",update->ntimestep);
  radius_out=radius_out+vradius*update->dt;
  radius_in=radius_in+vradius*update->dt;
    if(update->ntimestep%1==0)
    {
      //utils::logmesg(lmp,"REGION SHELL, THE SHELL Stress IN STEP {} IS {} \n",update->ntimestep,stresssum);
      //utils::logmesg(lmp,"REGION SHELL, Step {}, Radius {}\n",update->ntimestep,radius_in);
    }
  }
 
  //auto compute_press=modify->get_compute_by_id(utils::strdup(press_id + 2));
  //if (!compute_press)
  //        error->all(FLERR, "Could not find compute ID in KMSPHERE: {}",press_id);
  //press=compute_press->scalar;
  //if(update->ntimestep%1000==0 || update->ntimestep<2)
  //radius_out=radius_out+vradius*update->dt;
  //radius_in=radius_in+vradius*update->dt;

  
}

/* ----------------------------------------------------------------------
   error check on existence of variable
------------------------------------------------------------------------- */

void RegShell::variable_check()
{

  if (rstyle == VARIABLE) {
    svar = input->variable->find(sstr);
    if (svar < 0) error->all(FLERR, "Variable {} for region sphere does not exist", sstr);
    if (!input->variable->equalstyle(svar))
      error->all(FLERR, "Variable {} for region KMsphere is invalid style", sstr);
  }
}

/* ----------------------------------------------------------------------
   Set values needed to calculate velocity due to shape changes.
   These values do not depend on the contact, so this function is
   called once per timestep by fix/wall/gran/region.

------------------------------------------------------------------------- */

void RegShell::set_velocity_shape()
{
  xcenter[0] = xc;
  xcenter[1] = yc;
  xcenter[2] = zc;
  forward_transform(xcenter[0], xcenter[1], xcenter[2]);
  if (update->ntimestep > 0)
    rprev = prev[4];
  else
    rprev = radius_out;
  prev[4] = radius_out;
}

/* ----------------------------------------------------------------------
   add velocity due to shape change to wall velocity
------------------------------------------------------------------------- */

void RegShell::velocity_contact_shape(double *vwall, double *xc)
{
  utils::logmesg(lmp,"REGION SHELL, Velocity Contact shape invoke! radius {}\n",radius_in);
  double delx, dely, delz;    // Displacement of contact point in x,y,z

  delx = (xc[0] - xcenter[0]) * (1 - rprev / radius_out);
  dely = (xc[1] - xcenter[1]) * (1 - rprev / radius_out);
  delz = (xc[2] - xcenter[2]) * (1 - rprev / radius_out);

  vwall[0] += delx / update->dt;
  vwall[1] += dely / update->dt;
  vwall[2] += delz / update->dt;
}

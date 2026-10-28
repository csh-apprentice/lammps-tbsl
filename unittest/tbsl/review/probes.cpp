// Temporary review reproductions. Reuse the isolated fixture without changing
// production code or the existing baseline test registration.
#define main original_ionization_suite_main
#include "../test_ionization_pair.cpp"
#undef main
#include <cstdio>
#include <fstream>
#include "error.h"

namespace {
class PairProbe : public PairLJCutIOCoulDSF {
 public:
  explicit PairProbe(LAMMPS *lmp) : PairLJCutIOCoulDSF(lmp) {}
  void sentinel() { cut_io=-91; scalefactor=-92; spair=-93; sself=-94; }
  double ion_cutoff() const { return cut_io; }
  double scale() const { return scalefactor; }
  double pair_scale() const { return spair; }
  double self_scale() const { return sself; }
  const char *active_name() const { return activestr; }
};

void charge_eight()
{
  for(int variant=0; variant<3; ++variant) {
    auto p=pair_at_boundary();
    p[0].q = variant==0 ? 7 : 8;
    p[1].q = variant==2 ? 8 : 7;
    Fixture f(p,true,{1000,2000,3000,4000,5000,6000,7000,8000});
    f.evaluate(); f.evaluate();
    const auto s=f.state();
    near(s[0].q,8,"first endpoint reaches/stays at q=8");
    near(s[1].q,variant==0 ? 7 : 8,"second endpoint cannot exceed q=8");
  }
}

void integrated_owner_updates(bool migrate)
{
  auto p=pair_at_boundary({{3,-2,1}});
  if(migrate) p[0].x[0]=5.9999;
  Fixture f(p);
  int owner_before=0, owner_after=0, local_owner=0;
  for(int i=0;i<f.lmp->atom->nlocal;++i)
    if(f.lmp->atom->tag[i]==1) local_owner=rank+1;
  MPI_Allreduce(&local_owner,&owner_before,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  f.command("timestep 0.0001");
  f.command("neigh_modify every 1 delay 0 check no");
  f.command("fix integrator all nve");
  f.command("run 3");
  const auto s=f.state();
  near(s[0].q,1,"integration updates the ionized owner");
  near(s[1].q,0,"partner owner retains charge");
  const double a=std::sqrt(1.0/3.0);
  near(s[0].v[0],3+a,"migrating owner has post-collision velocity");
  near(s[1].v[0],3-a,"other owner has post-collision velocity");
  for(int d=0; d<3; ++d)
    near(s[0].v[d]+s[1].v[d],p[0].v[d]+p[1].v[d],"integrated pair momentum");
  auto atom=f.lmp->atom;
  local_owner=0;
  for(int i=0;i<atom->nlocal;++i) if(atom->tag[i]==1)local_owner=rank+1;
  MPI_Allreduce(&local_owner,&owner_after,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if(nranks>1)check((owner_before!=owner_after)==migrate,"fixture has intended owner migration");
  int local_ghosts=0,total_ghosts=0;
  for(int i=atom->nlocal;i<atom->nlocal+atom->nghost;++i) {
    ++local_ghosts;
    int k=atom->tag[i]-1;
    near(atom->q[i],s[k].q,"ghost charge matches owner after integration");
    for(int d=0; d<3; ++d) near(atom->v[i][d],s[k].v[d],"ghost velocity matches owner");
  }
  MPI_Allreduce(&local_ghosts,&total_ghosts,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if(nranks>1)check(total_ghosts>0,"ghost-state assertions are exercised");
}

void restart_parameters()
{
  Fixture f(pair_at_boundary(),true,{1e7,1e12},1000);
  FILE *file=rank==0 ? std::tmpfile() : nullptr;
  if(rank==0) { f.pair->write_restart(file); std::rewind(file); }
  PairProbe restored(f.lmp.get());
  restored.sentinel();
  restored.read_restart(file);
  near(restored.ion_cutoff(),2,"binary restart restores reactive cutoff");
  near(restored.scale(),1000,"binary restart restores EP scale");
  near(restored.pair_scale(),1000,"binary restart restores Coulomb pair factor");
  near(restored.self_scale(),10000,"binary restart restores DSF self factor");
  check(restored.active_name()!=nullptr && std::string(restored.active_name())=="active",
        "binary restart restores active-variable name");
  if(rank==0) std::fclose(file);
}

void split_run()
{
  const std::vector<double> thresholds={1000,2000,3000,4000,5000,6000,7000,8000};
  std::vector<Particle> continuous_state;
  {
    Fixture f(pair_at_boundary(),true,thresholds);
    f.command("timestep 0.00000001");
    f.command("fix integrator all nve");
    f.command("run 2");
    continuous_state=f.state();
  }
  Fixture split(pair_at_boundary(),true,thresholds);
  split.command("timestep 0.00000001");
  split.command("fix integrator all nve");
  split.command("run 1");
  split.command("run 1");
  const auto split_state=split.state();
  for(int k=0;k<2;++k) {
    near(split_state[k].q,continuous_state[k].q,"splitting a run must not add ionization events");
    for(int d=0;d<3;++d)
      near(split_state[k].v[d],continuous_state[k].v[d],"split run retains the same velocity");
  }
}

void three_settings()
{
  Fixture f(pair_at_boundary());
  PairProbe probe(f.lmp.get());
  // Allocate an extra sentinel argument to observe the out-of-contract read
  // safely instead of crashing on the actual input parser's null terminator.
  char a[]="1.5", b[]="2", c[]="2", sentinel[]="917";
  char *args[]={a,b,c,sentinel};
  bool rejected=false;
  try { probe.settings(3,args); }
  catch(const std::exception &) { rejected=true; }
  check(rejected,"three arguments rejected before reading a nonexistent fourth");
  if(!rejected) near(probe.ion_cutoff(),2,"reactive cutoff must not depend on memory beyond narg");
}

void loader_properties(bool reordered)
{
  const char *args[]={"review","-screen","none","-log","none","-nocite"};
  LAMMPS lmp(6,const_cast<char **>(args),MPI_COMM_WORLD);
  auto cmd=[&](const std::string &s){lmp.input->one(s);};
  cmd("units real"); cmd("atom_style charge");
  cmd("atom_modify map array sort 0 0.0"); cmd("boundary f f f");
  cmd("processors "+std::to_string(nranks)+" 1 1");
  cmd("region box block 0 12 0 12 0 12");cmd("create_box 1 box");
  cmd("create_atoms 1 single 5.75 6 6 units box");
  cmd("create_atoms 1 single 6.25 6 6 units box"); cmd("mass 1 2");
  cmd("fix mass all property/atom rmass ghost yes");cmd("set atom * mass 2");
  if(reordered) {
    cmd("fix label all property/atom i_label ghost yes");
    cmd("fix tlast all property/atom i_tlast");
    cmd("fix spare all property/atom d_spare ghost yes");
  } else {
    cmd("fix tlast all property/atom i_tlast");
    cmd("fix label all property/atom i_label ghost yes");
  }
  cmd("fix distance all property/atom d_mindist ghost yes");
  cmd("set atom * i_tlast 37");cmd("set atom 1 i_label 2");cmd("set atom 2 i_label 1");
  cmd("set atom * d_mindist 0.25");
  if(reordered) cmd("set atom * d_spare 123");
  const std::string path="/tmp/tbsl-review-dump-"+std::to_string(nranks)+(reordered?"-permuted":"-standard")+".dump";
  cmd("write_dump all custom "+path+" id type x y z vx vy vz q i_tlast i_label d_mindist modify sort id format float %.17g");
  cmd("set atom * i_tlast 0");cmd("set atom * i_label -1");cmd("set atom * d_mindist 999");
  cmd("read_sldump "+path+" 0 x y z vx vy vz q i_tlast i2_label d_mindist label i2_label i_label");
  int flag,cols;
  int ti=lmp.atom->find_custom("tlast",flag,cols);
  int li=lmp.atom->find_custom("label",flag,cols);
  int di=lmp.atom->find_custom("mindist",flag,cols);
  for(int i=0;i<lmp.atom->nlocal;++i) {
    near(lmp.atom->ivector[ti][i],37,"dump reload restores tlast by its name");
    near(lmp.atom->ivector[li][i],lmp.atom->tag[i]==1 ? 2 : 1,"dump reload restores partner label by name");
    near(lmp.atom->dvector[di][i],.25,"dump reload restores minimum distance by name");
    if(reordered) {
      int si=lmp.atom->find_custom("spare",flag,cols);
      near(lmp.atom->dvector[si][i],123,"dump reload must not overwrite unrelated property");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(rank==0) std::remove(path.c_str());
}
}

int main(int argc,char **argv)
{
 MPI_Init(&argc,&argv);
 MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nranks);
 const std::string filter=argc>1 ? argv[1] : "";
 auto run=[&](const std::string &name,void(*test)()) {
  if(!filter.empty() && name.find(filter)==std::string::npos)return;
  current=name;int before=failures;
  try {test();} catch(const std::exception &e) {check(false,std::string("exception: ")+e.what());}
  int total;MPI_Allreduce(&failures,&total,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  if(rank==0)std::cout<<(total==before?"PASS ":"FAIL ")<<name<<std::endl;
 };
 run("charge_q8",charge_eight);
 run("integrated_cross_rank",[]{integrated_owner_updates(false);});
 run("integrated_migration",[]{integrated_owner_updates(true);});
 run("binary_restart_parameters",restart_parameters);
 run("split_run",split_run);
 run("three_settings",three_settings);
 run("dump_standard",[]{loader_properties(false);});
 run("dump_reordered",[]{loader_properties(true);});
 int total,total_checks;MPI_Allreduce(&failures,&total,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
 MPI_Allreduce(&checks,&total_checks,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
 if(rank==0)std::cout<<total_checks<<" total rank-local checks, "<<total<<" max failed checks per rank on "<<nranks<<" ranks\n";
 MPI_Finalize();return total?1:0;
}

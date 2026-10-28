// Validate the author's actual read_sldump + restartverlet + activation gate.
#define main original_ionization_suite_main
#include "../../test_ionization_pair.cpp"
#undef main
#include "update.h"
#include <cstdio>
#include <fstream>

namespace {
using Record = std::array<double,15>;
std::vector<Record> snapshot(Fixture &f)
{
  std::vector<double> local(15*f.size,0),global(local.size());
  int flag,cols;
  auto atom=f.lmp->atom;
  int ti=atom->find_custom("tlast",flag,cols);
  int li=atom->find_custom("label",flag,cols);
  int di=atom->find_custom("mindist",flag,cols);
  for(int i=0;i<atom->nlocal;++i) {
    int k=atom->tag[i]-1;
    for(int d=0;d<3;++d) {
      local[15*k+d]=atom->x[i][d]; local[15*k+3+d]=atom->v[i][d];
      local[15*k+6+d]=atom->f[i][d];
    }
    local[15*k+9]=atom->q[i];local[15*k+10]=atom->ivector[ti][i];
    local[15*k+11]=atom->ivector[li][i];local[15*k+12]=atom->dvector[di][i];
    local[15*k+13]=atom->rmass[i];local[15*k+14]=1;
  }
  MPI_Allreduce(local.data(),global.data(),global.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
  std::vector<Record> result(f.size);
  for(int k=0;k<f.size;++k)for(int j=0;j<15;++j)result[k][j]=global[15*k+j];
  return result;
}
void save_state(const std::string &path,const std::vector<Record> &s)
{
  if(rank)return;
  std::ofstream out(path);out<<std::setprecision(17);
  out<<"id,x,y,z,vx,vy,vz,fx,fy,fz,q,tlast,label,mindist,rmass,owners\n";
  for(int k=0;k<s.size();++k) {out<<k+1;for(double v:s[k])out<<','<<v;out<<'\n';}
}
}
int main(int argc,char **argv)
{
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&nranks);
 if(argc!=2){if(!rank)std::cerr<<"supply existing output directory\n";MPI_Finalize();return 2;}
 const std::string directory=argv[1];
 const std::vector<double> thresholds={1000,2000,3000,4000,5000,6000,7000,8000};
 std::vector<Record> reference;
 {
  Fixture f(pair_at_boundary(),true,thresholds);
  f.command("timestep 0.00000001");f.command("fix integrate all nve");
  f.command("dump checkpoint all custom 2 "+directory+"/checkpoint_%.*.dump id type x y z vx vy vz fx fy fz q i_tlast i2_label d_mindist");
  f.command("dump_modify checkpoint format float %.17g");
  f.command("run 4");reference=snapshot(f);save_state(directory+"/reference.csv",reference);
 }
 for(bool gate : {true,false}) {
  const std::string mode=gate?"guarded":"unguarded";current=mode;
  Fixture f(pair_at_boundary(),true,thresholds);
  f.command("read_sldump "+directory+"/checkpoint_%.2.dump 2 x y z vx vy vz fx fy fz q i_tlast i2_label d_mindist nfile "+std::to_string(nranks));
  save_state(directory+"/"+mode+"_loaded.csv",snapshot(f));
  f.command("reset_timestep 2 time 0.00000002");f.command("timestep 0.00000001");
  f.command("fix integrate all nve");
  f.command(std::string("variable active equal ")+(gate?"step>2":"1"));
  f.command("run_style restartverlet");
  f.command("dump resumed all custom 2 "+directory+"/"+mode+"_%.*.dump id type x y z vx vy vz fx fy fz q i_tlast i2_label d_mindist");
  f.command("dump_modify resumed format float %.17g");
  f.command("run 2");const auto result=snapshot(f);save_state(directory+"/"+mode+"_final.csv",result);
  if(gate)for(int k=0;k<f.size;++k)for(int j=0;j<15;++j)
    near(result[k][j],reference[k][j],"author's guarded custom restart matches reference field "+std::to_string(j));
 }
 int failed;MPI_Allreduce(&failures,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
 if(!rank)std::cout<<checks<<" guarded-protocol checks, "<<failed<<" failures on "<<nranks<<" ranks\n";
 MPI_Finalize();return failed?1:0;
}

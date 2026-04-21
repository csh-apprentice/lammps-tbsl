#!/bin/bash -l
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --job-name=sl_1e4_alpha1_shortio
#SBATCH --output=/anvil/projects/x-phy250136/lammps-tbsl/test_1e4/run_1e4_alpha1_shortio.lammps
#SBATCH --error=/anvil/projects/x-phy250136/lammps-tbsl/test_1e4/run_1e4_alpha1_shortio.err
#SBATCH --time=1:00:00

set -x

cd /anvil/projects/x-phy250136/lammps-tbsl/test_1e4

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NNODES=${SLURM_NNODES}"

mpirun -np 16 /anvil/projects/x-phy250136/lammps-tbsl/src/lmp_mpi -in in.simulation_1e4_alpha_1_shortio

#!/bin/bash -l
#SBATCH --exclusive
#SBATCH --partition=wholenode
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --job-name=sl_1e6_01_13_alpha_0_shortio
#SBATCH --output=run_1e6_01_13_alpha_0_shortio.lammps
#SBATCH --error=run_1e6_01_13_alpha_0_shortio.err
#SBATCH --time=1-00:00:00

set -x  # debug: show commands

cd /anvil/projects/x-phy250136/Time_Based_SL/build/lammps_sequence_input

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NNODES=${SLURM_NNODES}"

# srun -n ${SLURM_NTASKS} ./lmp -in in.simulation
# Use your custom binary explicitly if it's not in PATH
# e.g. /anvil/projects/x-phy250136/Time_Based_SL/build/lmp
mpirun -np ${SLURM_NTASKS} ./lmp -in in.simulation_1e6_alpha_0_shortio
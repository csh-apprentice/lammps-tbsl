#!/bin/bash -l
#SBATCH --exclusive
#SBATCH --partition=wholenode
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --job-name=sl_1e4_16_nodes_16c
#SBATCH --output=run_1e4_16_nodes_16c.lammps
#SBATCH --error=run_1e4_16_nodes_16c.err
#SBATCH --time=1-00:00:00

set -x  # debug: show commands

cd /anvil/projects/x-phy250136/Time_Based_SL/build/lammps_sequence_input

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NNODES=${SLURM_NNODES}"

# srun -n ${SLURM_NTASKS} ./lmp -in in.simulation
# Use your custom binary explicitly if it's not in PATH
# e.g. /anvil/projects/x-phy250136/Time_Based_SL/build/lmp
mpirun -np ${SLURM_NTASKS} ./lmp -in in.simulation
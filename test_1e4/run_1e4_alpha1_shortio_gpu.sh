#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=sl_1e4_gpu
#SBATCH --output=/anvil/projects/x-phy250136/lammps-tbsl/test_1e4/run_1e4_alpha1_shortio_gpu.lammps
#SBATCH --error=/anvil/projects/x-phy250136/lammps-tbsl/test_1e4/run_1e4_alpha1_shortio_gpu.err
#SBATCH --time=4:00:00

set -x

cd /anvil/projects/x-phy250136/lammps-tbsl/test_1e4

echo "SLURM_NTASKS=${SLURM_NTASKS}"
echo "SLURM_NNODES=${SLURM_NNODES}"

# GPU binary built with -DPKG_GPU=on -DGPU_API=cuda -DGPU_ARCH=sm_86
# 'neigh no' is mandatory: ionization pipeline uses the CPU neighbor list
/anvil/projects/x-phy250136/lammps-tbsl/build-gpu/lmp_gpu -in in.simulation_1e4_alpha_1_shortio_gpu

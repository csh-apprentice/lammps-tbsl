#!/bin/bash -l
#SBATCH --job-name=ar_simplify_1e6
#SBATCH --output=intialize_1e6_lattice.log
#SBATCH --error=intialize_1e6_latticd.err
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --time=24:00:00



# Move to the directory containing this script
module load anaconda
cd "$(dirname "$0")"
conda activate TBSL

# Run LAMMPS with 16 MPI processes
#python generate_particles_fcc_warp_single.py --exact_N
#python generate_particles_chunked.py
# python generate_particles_satellite.py --avoid_boundary
# python genreate_particles_lattice.py \
#   --N=1000000000 \
#   --parallel \
#   --block=4 \
#   --avoid_boundary

# python generate_particles_lattice.py \
#   --N 1000000000 \
#   --parallel --cpus 128 \
#   --block 8 --tile 64 \
#   --pilot_factor 2.0 --tau_slack 1.02 \
#   --avoid_boundary \
#   --output initialize_lattice_1e9.lammpsdata

# python generate_particles_lattice.py \
#   --N 100000000 \
#   --parallel --cpus 128 \
#   --block 8 --tile 64 \
#   --pilot_factor 2.0 --tau_slack 1.02 \
#   --avoid_boundary \
#   --output initialize_lattice_1e8.lammpsdata

# python generate_particles_lattice.py \
#   --N 10000 \
#   --parallel --cpus 128 \
#   --block 8 --tile 64 \
#   --pilot_factor 2.0 --tau_slack 1.02 \
#   --avoid_boundary \
#   --output initialize_lattice_1e4.lammpsdata


# python generate_particles_lattice.py \
#   --N 10000000 \
#   --parallel --cpus 128 \
#   --block 8 --tile 64 \
#   --pilot_factor 2.0 --tau_slack 1.02 \
#   --avoid_boundary \
#   --output initialize_lattice_1e7.lammpsdata


python generate_particles_lattice.py \
  --N 1000000 \
  --parallel --cpus 128 \
  --block 8 --tile 64 \
  --pilot_factor 2.0 --tau_slack 1.02 \
  --avoid_boundary \
  --output initialize_lattice_1e6.lammpsdata

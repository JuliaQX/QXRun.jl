#!/bin/bash

#SBATCH -A tra21_hackathon
#SBATCH --time 1:00:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --ntasks-per-node=8 # 8 tasks out of 128
#SBATCH --gres=gpu:1        # 1 gpus per node out of 4
#SBATCH --mem=16000          # memory per node out of 246000MB
#SBATCH --job-name=TestProfile
#SBATCH --reservation=s_tra_hackathon21

module load profile/advanced
module load autoload julia/1.7--gnu--8.4.0
module load hpc-sdk/2021--binary

#nsys launch -o 0_timings_report_%h_%p julia --project=../../ 0_timings_prof.jl
nsys launch julia --project=../../ 0_timings_prof.jl
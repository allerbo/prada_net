#!/usr/bin/env bash
#SBATCH -A C3SE508-19-3
#SBATCH -p chair
#SBATCH -t 2-00:00:00
#SBATCH -o out_lbda_sweep.txt
#SBATCH -e err_lbda_sweep.txt
#SBATCH --gres=gpu:1

echo `date`
ml GCC/7.3.0-2.30 OpenMPI/3.1.1
ml CUDA/9.2.88
ml TensorFlow/1.10.1-Python-2.7.15
bash lbda_sweep.sh
echo `date`

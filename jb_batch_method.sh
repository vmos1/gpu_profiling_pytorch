#!/bin/bash
#################
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --output=slurm-%x-%j.out
#SBATCH --constraint=gpu
#SBATCH --account=m3363
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8 
#SBATCH --gpus-per-task=1
#SBATCH --time=0:20:00
#SBATCH --job-name=cnn_pytorch

echo "--start date" `date` `date +%s`
echo '--hostname ' $HOSTNAME
### Initial setup
# conda activate v3
module load pytorch/1.7.0-gpu
# module load nsight-systems
srun -N1 python pytorch_cnn-jb_method.py -b 256 -e 10

echo "--end date" `date` `date +%s`

#!/bin/bash
#BSUB -o /cluster/work/cvl/nipopovic/experiments/composite_tasking/euler_logs  # path to output file
#BSUB -W 23:59 # HH:MM runtime
#BSUB -n 32 # number of cpu cores
#BSUB -R "rusage[mem=4096]" # MB per CPU core
#BSUB -R "rusage[ngpus_excl_p=4]" # number of GPU cores
#BSUB -R "select[gpu_mtotal0>=10240]" # MB per GPU core
#BSUB -J "composite_tasking"

# Activate python environment
source /cluster/home/nipopovic/python_envs/composite_tasking/bin/activate

# Access to internet to download torch models
module load eth_proxy
# For parallel data unzipping
module load pigz

# Print number of GPU/CPU resources available
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Number of CPU threads/core: $(nproc --all)"

# Transfer ImageNet to scratch
echo "Transfering data to cluster scratch..."
tar -I pigz -xf /cluster/work/cvl/nipopovic/data/PASCAL_MT/pascal_mt.tar.gz -C ${TMPDIR}/
echo "Transfered data to cluster scratch"

# Set project paths
PROJECT_ROOT_DIR=/cluster/project/cvl/nipopovic/code/composite-tasking
export PYTHONPATH=${PYTHONPATH}:${PROJECT_ROOT_DIR}
cd ${PROJECT_ROOT_DIR}
pwd

# call the training script
python3 -u run_experiments/train.py \
--config=${PROJECT_ROOT_DIR}/run_experiments/configs/debug_config_euler.yaml \
--data_root_dir=${TMPDIR}/ILSVRC2012 \
--code_root_dir=${PROJECT_ROOT_DIR} \
--exp_root_dir=/cluster/work/cvl/nipopovic/experiments/composite_tasking

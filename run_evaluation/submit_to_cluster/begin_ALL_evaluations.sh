#!/bin/bash
#BSUB -o /cluster/work/cvl/nipopovic/experiments/composite_tasking/euler_logs  # path to output file
#BSUB -W 23:59 # HH:MM runtime
#BSUB -n 8 # number of cpu cores
#BSUB -R "rusage[mem=4096]" # MB per CPU core
#BSUB -R "rusage[ngpus_excl_p=1]" # number of GPU cores
#BSUB -R "select[gpu_mtotal0>=20240]" # MB per GPU core
#BSUB -J "CT_tr"

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

CONFIGS="ST_edges.yaml ST_normals.yaml ST_parts.yaml ST_saliency.yaml ST_seg.yaml R2.yaml R3.yaml MT.yaml Rrnd.yaml R1r.yaml"
for cfg in $CONFIGS; do 
    # call the evaluation script
    python3 -u run_evaluation/evaluate.py \
    --config_file_path=${PROJECT_ROOT_DIR}/run_evaluation/configs/composite_tasking_paper/$cfg \
    --checkpoint_path=PATH \
    --data_root_dir=${TMPDIR}/PASCAL_MT
done
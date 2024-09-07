#!/bin/bash
#SBATCH --job-name=openvla-lora-finetune     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=8      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=256G               # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of allocated gpus per node
#SBATCH --time=04:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=interactive  # grizzly, interactive or polar 
#SBATCH --output=/lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/slurm_outputs/multinode_%j.out
#SBATCH --err=/lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/slurm_outputs/logs/%x-%j.err
#SBATCH --account=nvr_srl_simpler
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@15       # tells the controller
                                 # to send SIGTERM to the job 15 secs
                                 # before its time ends to give it a
                                 # chance for better cleanup.


export MASTER_PORT="$((${SLURM_JOB_ID} % 10000 + 10000))"
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

trap 'scontrol requeue ${SLURM_JOB_ID}; exit 15' SIGTERM

#timeout 140s srun \
srun \
  --exclusive \
  --export=MASTER_ADDR,MASTER_PORT,SLURM_GPUS_ON_NODE,WORLD_SIZE,SLURM_PROCID,PYTHONDONTWRITEBYTECODE=1,PYTHONUNBUFFERED=1,SLURM_LAUNCH_NODE_IPADDR \
  /bin/bash -c "cd /lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/openvla && \
    source ~/.bashrc && \
    conda activate openvla && \
    torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
                                --data_root_dir ~/tensorflow_datasets/ \
                                --dataset_name robomimic_lift_dataset \
                                --run_root_dir /lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/openvla/logs \
                                --batch_size 16 \
                                --grad_accumulation_steps 1 \
                                --save_steps 200 \
                                --learning_rate 5e-4 \
                                --max_steps 10000" &
wait

#if [[ $? == 124 ]]; then
#  echo "Requeueing job "$SLURM_JOB_ID
#  scontrol requeue $SLURM_JOB_ID
#fi

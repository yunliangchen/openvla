cd /lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/openvla && \
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "prism-dinosiglip-224px+mx-robomimic" \
  --data_root_dir ~/tensorflow_datasets/ \
  --run_root_dir /lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/openvla/logs \
  --save_interval 100 \
  --pretrained_checkpoint "openvla/openvla-7b" \
  --is_resume false
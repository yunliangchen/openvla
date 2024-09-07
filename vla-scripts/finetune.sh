cd /lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/openvla && \
sleep 5 && \
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune.py \
                                --data_root_dir ~/tensorflow_datasets/ \
                                --dataset_name icrl_vla_tfds \
                                --run_root_dir /lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/openvla/logs \
                                --batch_size 8 \
                                --grad_accumulation_steps 4 \
                                --save_steps 10 \
                                --learning_rate 5e-4 \
                                --max_steps 10000 \
                                --vla_path /lustre/fsw/portfolios/nvr/users/lawchen/project/openvla/openvla/logs/openvla-7b+icrl_vla_tfds+b32+lr-0.0005+lora-r32+dropout-0.0+wrist+raw_action+16step
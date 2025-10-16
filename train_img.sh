#!/bin/bash
export MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES=1,2,3 # Only use the last three GPUs
echo "MASTER_ADDR is $MASTER_ADDR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
torchrun --master_addr=$MASTER_ADDR --nproc_per_node=2 train.py \
    --model DiT-XS/2 \
    --epochs 10 \
    --batch_size 4 \
    --num_workers 8 \
    --num_classes 1 \
    --ckpt_every 5000 \
    --sat_files_path /ssd/yghu/Data/dataset/noradar \
    --radar_files_path /ssd/yghu/Data/dataset/radar/71 \
    --start_date 20210201 \
    --end_date 20210430 \
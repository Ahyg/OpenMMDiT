#!/bin/bash
export MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES=0,1,2,3 # Only use these GPUs
echo "MASTER_ADDR is $MASTER_ADDR"
echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
torchrun --master_addr=$MASTER_ADDR --nproc_per_node=4 train.py \
    --model DiT-B/2 \
    --epochs 100 \
    --batch_size 4 \
    --num_workers 0 \
    --num_classes 1 \
    --ckpt_every 1000 \
    --sat_files_path /mnt/ssd_1/yghu/Data/dataset/noradar \
    --radar_files_path /mnt/ssd_1/yghu/Data/dataset/radar/71 \
    --split_ratio "(0.7,0.2,0.1)" \
    --block_size 100 \
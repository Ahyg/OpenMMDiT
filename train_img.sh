#!/bin/bash
export MASTER_ADDR=127.0.0.1
echo "MASTER_ADDR is $MASTER_ADDR"
torchrun --master_addr=$MASTER_ADDR --nproc_per_node=4 train.py \
    --model DiT-B/2 \
    --epochs 400 \
    --batch_size 4 \
    --num_workers 0 \
    --num_classes 0 \
    --ckpt_every 5000 \
    --sat_files_path /ssd/yghu/Data/dataset/noradar \
    --radar_files_path /ssd/yghu/Data/dataset/radar/71 \
    --start_date 20210201 \
    --end_date 20210430 \
    --history_frames 0 \
    --future_frame 0 \
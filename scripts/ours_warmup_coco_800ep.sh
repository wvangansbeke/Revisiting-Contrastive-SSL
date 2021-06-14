#!/bin/sh

python main_ours_warmup.py \
       -a resnet50 \
       --epochs 800 \
       --lr 0.3 \
       --workers 16 \
       --output_dir '/path/to/output_dir' \
       --batch-size 256 \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
       /path/to/MS-COCO-2017/train2017 \
       --mlp --moco-t 0.2 --cos --moco-m 0.995 \
       --size_crops 160 96 \
       --min_scale_crops 0.2 0.05 \
       --max_scale_crops 1.0 0.14 \
       --num_crops 2 4 \
       --constrained-cropping \
       --auto-augment 0 2 3 4 5 \
       --aux-topk 20 --aux-weight 0.4 --warmup-epochs 20 

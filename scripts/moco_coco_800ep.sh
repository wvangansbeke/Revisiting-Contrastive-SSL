#!/bin/sh

python main.py \
       -a resnet50 \
       --epochs 800 \
       --lr 0.3 \
       --workers 16 \
       --output_dir '/path/to/output_dir' \
       --batch-size 256 \
       --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
       /path/to/MS-COCO-2017/train2017 \
       --mlp --moco-t 0.2 --cos \
       --size_crops 224 \
       --min_scale_crops 0.2 \
       --max_scale_crops 1.0 \
       --num_crops 2

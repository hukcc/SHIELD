#!/bin/bash

gt_file="experiments/data/POPE/coco/coco_pope_random.json"

gen_files_dir="opera-pope-coco/"

for gen_file in "$gen_files_dir"*.json; do
    echo "Gen file: $gen_file"
    
    python3 experiments/eval/eval_pope.py --gt_files "$gt_file" --gen_files "$gen_file"
    
    echo "--------------------------------------------"
done

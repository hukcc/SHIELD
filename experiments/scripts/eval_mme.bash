#!/bin/bash

gt_file="experiments/data/MME/hal.json"

gen_files_dir="output_mme/"

for gen_file in "$gen_files_dir"*.jsonl; do
    echo "Gen file: $gen_file"
    
    python3 experiments/eval/eval_mme.py --gt_files "$gt_file" --gen_files "$gen_file"
    
    echo "--------------------------------------------"
done

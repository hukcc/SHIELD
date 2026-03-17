#!/bin/bash

gt_file="experiments/data/MME/hal.json"

gen_files_dir="opera_mme_subset_results/"

for gen_file in "${gen_files_dir}"*.json; do
    echo "Gen file: $gen_file"
    
    gt_file="${gen_file/$gen_files_dir/experiments/data/MME/mme_json_files/}"

    echo "Gt file: $gt_file"
    
    python3 experiments/eval/eval_mme.py --gt_files "$gt_file" --gen_files "$gen_file"
    
    echo "--------------------------------------------"
done


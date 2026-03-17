#!/bin/bash

# 20241221: S=36.6, I=10.3. Only alpha, beta, and `the` were tuned.
export CUDA_VISIBLE_DEVICES=0
# ============================ # 
seed=${1:-42} 
dataset_name=${2:-"chair"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
# ============================ # 
cd_alpha=${5:-2.0}
cd_beta=${6:-0.35}
noise_step=${7:-500}
# ============================ # 
the=${8:-0.002} 
gamma_gain=${9:-3.0}
gain_per=${11:-0.55}
gamma_reduce=${10:-3.0}
reduce_per=${12:-0.0}
# ============================ # 
bias_weight=${13:-0.01}
bias_sample_num=${14:-32}
# ============================ # 
cw_epsilon=${15:-0.14}
cw_num_steps=${16:-30}
cw_c=${17:-12}
cw_lr=${18:-0.02}


answers_file="./output/llava15_${dataset_name}_answers_bias_weight${bias_weight}_bias_sample_num${bias_sample_num}_alpha${cd_alpha}_beta${cd_beta}_the${the}_gamma${gamma_gain}_per${gain_per}_cw_epsilon${cw_epsilon}_cw_c${cw_c}_cw_lr${cw_lr}_seed${seed}.jsonl"

caption_file="./experiments/first_cap/llava15_chair_first_caption.jsonl"

python experiments/eval/chair-llava.py \
      --model-path ${model_path} \
      --question-file ./experiments/data/CHAIR/questions.jsonl \
      --image-folder ./experiments/data/coco/val2014 \
      --caption-file ${caption_file} \
      --answers-file ${answers_file} \
      --use_cd \
      --cd_alpha $cd_alpha \
      --cd_beta $cd_beta \
      --noise_step $noise_step \
      --the ${the} \
      --gamma_gain ${gamma_gain} \
      --gamma_reduce ${gamma_reduce} \
      --gain_per ${gain_per} \
      --reduce_per ${reduce_per} \
      --bias_weight ${bias_weight} \
      --bias_sample_num ${bias_sample_num} \
      --cw_epsilon ${cw_epsilon} \
      --cw_num_steps ${cw_num_steps} \
      --cw_c ${cw_c} \
      --cw_lr ${cw_lr} \
      --seed ${seed}

# ===== CHAIR Evaluation =====
echo ""
echo "========================================"
echo "Running CHAIR evaluation..."
echo "========================================"
python experiments/eval/chair_eval.py \
      --cap_file ${answers_file} \
      --image_id_key image_id \
      --caption_key caption \
      --cache experiments/eval/chair.pkl
echo "========================================"

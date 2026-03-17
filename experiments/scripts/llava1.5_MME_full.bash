#!/bin/bash

# Set the GPU device.
export CUDA_VISIBLE_DEVICES=1

# ============================ # 
seed=${1:-42} 
dataset_name=${2:-"mme"}
type=${3:-"full"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
# ============================ # 
cd_alpha=${5:-0.5}
cd_beta=${6:-0.5}
noise_step=${7:-500}
# ============================ # 
the=${8:-0.004}
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
# ============================ # 
temperature=${19:-1.0}
top_p=${20:-1.0}
top_k=${21:-3}

question_files=(
  "./experiments/data/MME/mme_json_files/artwork.json"
  "./experiments/data/MME/mme_json_files/code_reasoning.json"
  "./experiments/data/MME/mme_json_files/commonsense_reasoning.json"
  "./experiments/data/MME/mme_json_files/existence.json"
  "./experiments/data/MME/mme_json_files/numerical_calculation.json"
  "./experiments/data/MME/mme_json_files/position.json"
  "./experiments/data/MME/mme_json_files/scene.json"
  "./experiments/data/MME/mme_json_files/celebrity.json"
  "./experiments/data/MME/mme_json_files/color.json"
  "./experiments/data/MME/mme_json_files/count.json"
  "./experiments/data/MME/mme_json_files/landmark.json"
  "./experiments/data/MME/mme_json_files/OCR.json"
  "./experiments/data/MME/mme_json_files/posters.json"
  "./experiments/data/MME/mme_json_files/text_translation.json"
)

for question_file in "${question_files[@]}"; do
  echo "Running evaluation for question file: $question_file"

  caption_file="./experiments/first_cap/llava15_mme_full_first_caption.jsonl"

  python experiments/eval/mme_llava.py \
          --model-path ${model_path} \
          --question-file ${question_file} \
          --image-folder ./experiments/data/MME/MME_Benchmark_release_version/MME_Benchmark \
          --caption-file ${caption_file} \
          --answers-file ./output/llava15_${dataset_name}_${type}_answers_bias_weight${bias_weight}_bias_sample_num${bias_sample_num}_alpha${cd_alpha}_beta${cd_beta}_the${the}_gamma${gamma_gain}_per${gain_per}_cw_epsilon${cw_epsilon}_cw_c${cw_c}_cw_lr${cw_lr}_seed${seed}.jsonl \
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
          --seed ${seed} \
          --temperature ${temperature} \
          --top_p ${top_p} \
          --top_k ${top_k}
done

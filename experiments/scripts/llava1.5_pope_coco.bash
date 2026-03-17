#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
seed=${1:-42} 
# coco aokvqa gqa
dataset_name=${2:-"coco"}
# random popular adversarial
type=${3:-"random"}
model_path=${4:-"liuhaotian/llava-v1.5-7b"}
# ============================ # 
cd_alpha=${5:-2.0} #1.64
cd_beta=${6:-0.35}
noise_step=${7:-999}
# ============================ # 
the=${8:-0.011}
gamma_gain=${9:-3.0}
gain_per=${11:-0.5}
gamma_reduce=${10:-3.0}
reduce_per=${12:-0.0}
# ============================ # 
bias_weight=${13:-0.1} #0.15 / 0.1
bias_sample_num=${14:-32}
# ============================ # 
cw_epsilon=${15:-0.14}
cw_num_steps=${16:-30}
cw_c=${17:-12}
cw_lr=${18:-0.14}


if [[ $dataset_name == 'coco' || $dataset_name == 'aokvqa' ]]; then
  image_folder=./experiments/data/coco/val2014
else
  image_folder=./data/gqa/images
fi

caption_file="./experiments/first_cap/llava15_${dataset_name}_pope_first_caption.jsonl"

python experiments/eval/object_hallucination_vqa_llava.py \
      --model-path ${model_path} \
      --question-file ./experiments/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json \
      --image-folder ${image_folder} \
      --caption-file ${caption_file} \
      --answers-file ./output/llava15_${dataset_name}_pope_${type}_answers_bias_weight${bias_weight}_bias_sample_num${bias_sample_num}_alpha${cd_alpha}_beta${cd_beta}_the${the}_gamma${gamma_gain}_per${gain_per}_cw_epsilon${cw_epsilon}_cw_c${cw_c}_cw_lr${cw_lr}_seed${seed}.jsonl \
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



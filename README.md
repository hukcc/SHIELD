<p align="center">
    <img src="figs/teasor.png">
</p>

# SHIELD

**[ICLR 2026🔥] SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense**

This is the official implementation for [SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](https://arxiv.org/abs/2510.16596)

by [Yiyang Huang](https://hukcc.github.io/), Liang Shi, Yitian Zhang, Yi Xu, [Yun Fu](https://www1.ece.neu.edu/~yunfu/).

[[Paper (arXiv)]](https://arxiv.org/abs/2510.16596) [[OpenReview]](https://openreview.net/forum?id=yk7FFLoNcP)

<p align="center">
    <img src="figs/pipeline.png">
</p>

SHIELD is a **training-free** framework that mitigates hallucinations in Large Vision-Language Models (LVLMs) by tracing the issue back to **visual encoders** and addressing three factors: **statistical bias**, **inherent bias**, and **vulnerability**. It achieves strong performance across multiple hallucination benchmarks (CHAIR, POPE, MME, AMBER).

## Table of Contents

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Inference and Evaluation](#inference-and-evaluation)
  - [POPE](#pope)
  - [CHAIR](#chair)
  - [MME](#mme)
- [Project Structure](#project-structure)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
- [License](#license)

## Getting Started

### Installation

1. Create a new conda environment and activate it:
```bash
conda create -n shield python=3.10
conda activate shield
```

2. Install PyTorch (tested with PyTorch 2.0.1 / CUDA 11.8):
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

3. Install the remaining requirements:
```bash
pip install -r requirements.txt
```

### Data Preparation

The repository already includes all text-based metadata (question files, question lists, evaluation scripts, etc.). You only need to download the **images** and **COCO annotations** separately.

#### LLaVA Model

SHIELD uses [LLaVA-1.5](https://github.com/haotian-liu/LLaVA) as the base model. Model weights are automatically downloaded from Hugging Face when first used (`liuhaotian/llava-v1.5-7b`).

#### COCO Images & Annotations

COCO val2014 images are shared across POPE (COCO), CHAIR, and LLaVA-Bench evaluations.

1. Download [COCO val2014 images](http://images.cocodataset.org/zips/val2014.zip) and extract to `experiments/data/coco/val2014/`.
2. Download [COCO 2014 annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) and extract to `experiments/data/coco/annotations/`.

```bash
cd experiments/data/coco
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2014.zip -d .
mv annotations_trainval2014/annotations .
```

> The COCO annotations are needed to build the CHAIR evaluator. A pre-built cache (`experiments/eval/chair.pkl`) is included so you can skip this step if you only want to run the CHAIR metric.

#### POPE

POPE question files for COCO, A-OKVQA, and GQA are already included under `experiments/data/POPE/`. No extra download needed.

For **GQA** images (only needed for POPE-GQA evaluation), download from the [GQA dataset](https://cs.stanford.edu/people/doersch/gqa/images.zip) and extract to `experiments/data/gqa/images/`.

#### CHAIR

CHAIR questions are included at `experiments/data/CHAIR/questions.jsonl`. Images come from COCO val2014 (see above).

#### MME

1. Download the [MME Benchmark](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) images and extract to `experiments/data/MME/MME_Benchmark_release_version/`.
2. MME question lists and evaluation tools are already included.

#### LLaVA-Bench

LLaVA-Bench data (images + questions) is fully included in `experiments/data/llava-bench/`. No extra download needed.

#### Caption Files

Pre-generated first-round captions for all benchmarks are provided under `experiments/first_cap/`.

#### Expected Directory Structure

After downloading, the data directory should look like:

```
experiments/data/
├── POPE/                          # (included) question files
│   ├── coco/
│   │   ├── coco_pope_random.json
│   │   ├── coco_pope_popular.json
│   │   └── coco_pope_adversarial.json
│   ├── aokvqa/                    # A-OKVQA POPE splits
│   └── gqa/                       # GQA POPE splits
├── CHAIR/
│   └── questions.jsonl            # (included)
├── MME/
│   ├── full.json                  # (included) question list
│   ├── hal.json                   # (included) question list
│   ├── MME_Benchmark_release_version/  # (download) images + QA text
│   └── mme_json_files/            # (included)
├── llava-bench/                   # (fully included)
│   ├── images/
│   └── questions.jsonl
├── coco/
│   ├── val2014/                   # (download) COCO val2014 images
│   └── annotations/              # (download) COCO 2014 annotations
└── gqa/
    └── images/                    # (download) GQA images
```

## Quick Start

SHIELD works as a **non-invasive wrapper** around LLaVA models. No source code modification of LLaVA is needed.

We provide ready-to-run scripts under `experiments/scripts/` for all supported benchmarks. To reproduce the results from our paper, simply run:

```bash
# POPE (COCO)
bash experiments/scripts/llava1.5_pope_coco.bash

# CHAIR
bash experiments/scripts/llava1.5_chair.bash

# MME (full / hallucination subset)
bash experiments/scripts/llava1.5_MME_full.bash
bash experiments/scripts/llava1.5_MME_hal.bash

# GPT-4o aided evaluation (LLaVA-Bench)
bash experiments/scripts/llava1.5_gpt4o_eval.bash
```

### Python API

For custom integration, SHIELD can be used as a Python API:

```python
import shield
from llava.model.builder import load_pretrained_model

# 1. Load the model as usual
tokenizer, model, image_processor, _ = load_pretrained_model(
    "liuhaotian/llava-v1.5-7b", None, "llava-v1.5-7b"
)

# 2. Wrap the model with SHIELD (one-time setup)
shield.wrap(model, tokenizer,
    caption_file="experiments/first_cap/llava15_coco_pope_first_caption.jsonl",
    cd_alpha=2.0,
    cd_beta=0.35,
)

# 3. For each image, prepare SHIELD inputs and generate
image = Image.open("path/to/image.jpg")
image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

shield_kw = model.shield_prepare(image, image_tensor, "image.jpg", use_cd=True)

output_ids = model.generate(
    input_ids,
    **shield_kw,
    do_sample=True,
    max_new_tokens=1024,
    use_cache=True,
)
```

## Inference and Evaluation

### POPE

Run inference on POPE (COCO, random split):

```bash
bash experiments/scripts/llava1.5_pope_coco.bash
```

Evaluate the results:

```bash
python experiments/eval/eval_pope.py \
    --gt_file experiments/data/POPE/coco/coco_pope_random.json \
    --gen_file output/llava15_coco_pope_random_answers_*.jsonl
```

Other POPE splits (popular, adversarial) and datasets (A-OKVQA, GQA) can be run by passing arguments to the script. See `experiments/scripts/llava1.5_pope_coco.bash` for details.

### CHAIR

Run inference and evaluation (evaluation runs automatically after inference):

```bash
bash experiments/scripts/llava1.5_chair.bash
```

The CHAIR evaluation script (`experiments/eval/chair_eval.py`) computes CHAIRs, CHAIRi, and Recall metrics. It requires the `pattern` library for NLP singularization:

```bash
pip install git+https://github.com/clips/pattern.git
```

### MME

```bash
bash experiments/scripts/llava1.5_MME_full.bash
bash experiments/scripts/llava1.5_MME_hal.bash
```

## Project Structure

```
SHIELD/
├── shield/                      # Core SHIELD library (pip-installable)
│   ├── __init__.py              # Public API
│   ├── wrapper.py               # shield.wrap() -- non-invasive model patching
│   ├── attack.py                # CW and PGD adversarial attacks in CLIP space
│   ├── caption.py               # Caption loading and preprocessing
│   ├── clip_utils.py            # CLIP model loading and text features
│   ├── feature.py               # Feature weighting, bias computation
│   ├── noise.py                 # Diffusion noise injection (VCD-compatible)
│   └── sampling.py              # Custom contrastive decoding sampler
├── experiments/
│   ├── eval/                    # Evaluation scripts
│   │   ├── object_hallucination_vqa_llava.py   # POPE inference
│   │   ├── chair-llava.py                      # CHAIR inference
│   │   ├── chair_eval.py                       # CHAIR metric computation
│   │   ├── eval_pope.py                        # POPE metric computation
│   │   ├── mme_llava.py                        # MME inference
│   │   └── eval_mme.py                         # MME metric computation
│   ├── scripts/                 # Bash scripts for running experiments
│   ├── data/                    # Evaluation datasets (POPE, CHAIR, MME)
│   ├── first_cap/               # Pre-generated first-round captions
│   └── llava/                   # LLaVA model code (vendored)
├── logs/                        # SOTA results for LLaVA, InstructBLIP, Qwen-VL
├── figs/                        # Paper figures
├── requirements.txt
├── CITATION.bib
├── LICENSE
└── README.md
```

## Acknowledgement

We extend our gratitude to the following projects:

- [LLaVA](https://github.com/haotian-liu/LLaVA) -- Large Language and Vision Assistant
- [VCD](https://github.com/DAMO-NLP-SG/VCD) -- Visual Contrastive Decoding
- [OPERA](https://github.com/shikiw/OPERA) -- Alleviating Hallucination in Multi-Modal LLMs via Over-Trust Penalty and Retrospection-Allocation
- [CHAIR](https://github.com/LisaAnne/Hallucination) -- Object Hallucination evaluation metric
- [Qwen-VL](https://github.com/QwenLM/Qwen-VL) -- Qwen Vision-Language model

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{
huang2026shield,
title={{SHIELD}: Suppressing Hallucinations In {LVLM} Encoders via Bias and Vulnerability Defense},
author={Yiyang Huang and Liang Shi and Yitian Zhang and Yi Xu and Yun Fu},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=yk7FFLoNcP}
}
```

arXiv version:

```bibtex
@article{huang2025shield,
  title={SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense},
  author={Huang, Yiyang and Shi, Liang and Zhang, Yitian and Xu, Yi and Fu, Yun},
  journal={arXiv preprint arXiv:2510.16596},
  year={2025}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).

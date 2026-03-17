import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--gt_files", type=str, default="experiments/data/MME/full.json")
parser.add_argument("--gen_files", type=str, default="output_unify_params/llava15_mme_full_answers_seed42.jsonl")
args = parser.parse_args()

# open ground truth answers
gt_files = [json.loads(q) for q in open(os.path.expanduser(args.gt_files), "r")]

# open generated answers
gen_files = [json.loads(q) for q in open(os.path.expanduser(args.gen_files), "r")]

# Initialize metrics
true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
unknown = 0
total_questions = len(gt_files)
yes_answers = 0

# Dictionary to track per-setting and per-image accuracy
setting_acc_tracker = defaultdict(lambda: defaultdict(list))

# Compare answers
for index, line in enumerate(gt_files):
    idx = line["question_id"]
    gt_answer = line["label"]
    assert idx == gen_files[index]["question_id"] 
    gen_answer = gen_files[index]["text"]
    image_file = gen_files[index]["image"]

    # Extract setting from the image path
    setting = os.path.dirname(image_file)
    
    # Convert to lowercase and strip
    gt_answer = gt_answer.lower().strip()
    gen_answer = gen_answer.lower().strip()
    
    # pos = 'yes', neg = 'no'
    is_correct = False
    if gt_answer == 'yes':
        if 'yes' in gen_answer:
            true_pos += 1
            yes_answers += 1
            is_correct = True
        else:
            false_neg += 1
    elif gt_answer == 'no':
        if 'no' in gen_answer:
            true_neg += 1
            is_correct = True
        else:
            yes_answers += 1
            false_pos += 1
    else:
        print(f'Warning: unknown gt_answer: {gt_answer}')
        unknown += 1
    
    # Track setting and image-level correctness
    setting_acc_tracker[setting][image_file].append(is_correct)

# Calculate global accuracy
precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (true_pos + true_neg) / total_questions if total_questions > 0 else 0
yes_proportion = yes_answers / total_questions if total_questions > 0 else 0
unknown_prop = unknown / total_questions if total_questions > 0 else 0

# Calculate per-setting accuracy and acc+
fianl_score = 0
setting_results = {}
for setting, images in setting_acc_tracker.items():
    total_images = len(images)
    correct_images = 0
    total_questions = sum(len(correctness_list) for correctness_list in images.values())
    correct_questions = sum(sum(correctness_list) for correctness_list in images.values())
    
    for correctness_list in images.values():
        if all(correctness_list):
            correct_images += 1
    
    acc = correct_questions / total_questions if total_questions > 0 else 0
    acc_plus = correct_images / total_images if total_images > 0 else 0
    acc = acc * 100
    acc_plus = acc_plus * 100
    total_score = acc+acc_plus
    fianl_score += total_score
    setting_results[setting] = {
        "acc": acc,
        "acc_plus": acc_plus,
        "total_score": total_score,
    }

# Report global results
print(f'fianl_score: {fianl_score:.4f}')

# Report per-setting results
print("\nPer-Setting Results:")
for setting, metrics in setting_results.items():
    print(f"Setting: {setting}" + f", total_score: {metrics['total_score']:.4f}")
    # print(f"  Accuracy (Acc): {metrics['acc']:.4f}")
    # print(f"  Image-level Accuracy (Acc+): {metrics['acc_plus']:.4f}")
    # print(f"  total_score: {metrics['total_score']:.4f}")

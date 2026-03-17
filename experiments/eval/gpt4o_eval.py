import base64
import requests
import os
import json

GPT_JUDGE_PROMPT = '''
You are required to score the performance of four AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinations should be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not count as necessary details.
Please output the scores for each criterion, containing only four values indicating the scores for Assistant 1, 2, 3 and 4, respectively. The four scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

[Assistant 3]
{}
[End of Assistant 3]

[Assistant 4]
{}
[End of Assistant 4]

Output format:
Accuracy: <Scores of the four answers>
Reason:

Detailedness: <Scores of the four answers>
Reason: 
'''


API_KEY = os.environ.get("OPENAI_API_KEY", "")


def call_api(prompt, image_path):
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
    "model": "chatgpt-4o-latest",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response)
    print(response.json().keys())
    return response.json()


def get_gpt4v_answer(prompt, image_path):
    while 1:
        try:
            res = call_api(prompt, image_path)
            if "choices" in res.keys():
                return res["choices"][0]["message"]["content"]
            else:
                assert False
        except Exception as e:
            print("retry")


base_path = "output"

json_files = [
                "/path/to/SHIELD/SOTA-results/QwenVL/gpt4o-eval-aid/output_logs/baseline.jsonl",
                "/path/to/SHIELD/SOTA-results/QwenVL/gpt4o-eval-aid/output_logs/opera.jsonl",
                "/path/to/SHIELD/SOTA-results/QwenVL/gpt4o-eval-aid/output_logs/contrastive_decoding.jsonl",
                "/path/to/SHIELD/SOTA-results/QwenVL/gpt4o-eval-aid/output_logs/ours.jsonl"
                ]

image_folder_path = "experiments/data/llava-bench/images"

model_dir = base_path
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

gpt_answer_records = {}
avg_hal_score_1 = 0
avg_hal_score_2 = 0
avg_hal_score_3 = 0
avg_hal_score_4 = 0
avg_det_score_1 = 0
avg_det_score_2 = 0
avg_det_score_3 = 0
avg_det_score_4 = 0
num_count = 0


data_list = []

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        file_data = []
        for line in f:
            file_data.append(json.loads(line.strip()))
        data_list.append(file_data)

num_entries = len(data_list[0])

for i in range(num_entries):
    captions = [data_list[0][i]["caption"], data_list[1][i]["caption"], data_list[2][i]["caption"], data_list[3][i]["caption"]]

    prompt = GPT_JUDGE_PROMPT.format(captions[0], captions[1], captions[2], captions[3])

    image_path = image_folder_path + '/' + f"{(i+1):03}" + '.jpg'

    gpt_answer = get_gpt4v_answer(prompt, image_path)
    print(gpt_answer)

    gpt_answer_records[f"image_{i}"] = gpt_answer

    success = False
    for attempt in range(10):
        try:
            hal_score_1, hal_score_2, hal_score_3, hal_score_4 = gpt_answer.split("Accuracy: ")[-1].split("\n")[0].strip().split(" ")
            det_score_1, det_score_2, det_score_3, det_score_4 = gpt_answer.split("Detailedness: ")[-1].split("\n")[0].strip().split(" ")
            
            num_count += 1
            success = True
            break
        except ValueError:
            print(f"Error processing scores for entry {i}. Retrying...")
            if attempt < 9:
                gpt_answer = get_gpt4v_answer(prompt, image_path)
                print(f"Retry {attempt + 1}: {gpt_answer}")
                gpt_answer_records[f"image_{i}_retry_{attempt + 1}"] = gpt_answer
        
    if success:
        avg_hal_score_1 += int(hal_score_1)
        avg_hal_score_2 += int(hal_score_2)
        avg_hal_score_3 += int(hal_score_3.replace("**", ""))
        avg_hal_score_4 += int(hal_score_4.replace("**", ""))
        avg_det_score_1 += int(det_score_1)
        avg_det_score_2 += int(det_score_2)
        avg_det_score_3 += int(det_score_3.replace("**", ""))
        avg_det_score_4 += int(det_score_4.replace("**", ""))
    else:
        print(f"Failed to process scores for entry {i} after multiple attempts. Skipping...")
        continue

    print("=========================================")


if num_count > 0:
    avg_hal_score_1 /= num_count
    avg_hal_score_2 /= num_count
    avg_hal_score_3 /= num_count
    avg_hal_score_4 /= num_count
    avg_det_score_1 /= num_count
    avg_det_score_2 /= num_count
    avg_det_score_3 /= num_count
    avg_det_score_4 /= num_count

print(f"The avg hal score for Assistant 1 and Assistant 2 and Assistant 3 and Assistant 4: {avg_hal_score_1}; {avg_hal_score_2}; {avg_hal_score_3}; {avg_hal_score_4}")
print(f"The avg det score for Assistant 1 and Assistant 2 and Assistant 3 and Assistant 4: {avg_det_score_1}; {avg_det_score_2}; {avg_det_score_3}; {avg_det_score_4}")

from datasets import load_from_disk, Dataset
import json
from tqdm import tqdm
import os

mcq_path = r"/projects/XXXX-1/dora/mcq_dataset/mcq_dataset.json"
dora_path = r"/projects/XXXX-1/dora/grpo_dataset_updatedv2/"

# mcq_dataset = load_from_disk(mcq_path)
with open(mcq_path, 'r') as f:
    mcq_data = json.load(f)
dora_dataset = load_from_disk(dora_path)

transcript_dict = {}
print("Building index...")
exp_count = 10
i = 0
if os.path.exists("temp/transcript.json"):
    with open("temp/transcript.json", 'r') as f:
        transcript_dict = json.load(f)
else:
    for item in tqdm(dora_dataset):
        context = item['transcript']
        transcript_dict[item['question']] = context
        # if i > exp_count:
        #     break
        # i += 1
        
    with open("temp/transcript.json", 'w') as f:
        json.dump(transcript_dict, f)

print("Updating context...")
i = 0
for idx in tqdm(range(len(mcq_data))):
    item = mcq_data[idx]
    find_id = item['question']
    if find_id not in transcript_dict.keys():
        print("key not found: ", find_id)
        continue
    new_context = transcript_dict[find_id]
    mcq_data[idx]['transcript'] = new_context

print("Saving data...")
with open("mcq_dataset_updated.json", 'w', encoding='utf-8') as f:
    json.dump(mcq_data, f, indent=2, ensure_ascii=False)
    



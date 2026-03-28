import re
import os
import csv
import json
import time
import logging
import requests
from tqdm import tqdm
from datetime import datetime

api_key = 'sk-fhnynvqbkfaiiqiyyuyraygksgaujwjrlkpgacnviaukbqmp'
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

EN_base_dir = r'C:\Users\lenovo\Desktop\codeMT-CSD-main\MT-CSD-main\processed_data'
ZN_base_dir = r'C:\Users\lenovo\Desktop\codeMT-CSD-main\MT-CSD-main\processed_data'  
output_base_dir = r'C:\Users\lenovo\Desktop\codeMT-CSD-main\MT-CSD-main\disambiguated_data_biden_trump'  # 输出文件的基目录

os.makedirs(output_base_dir, exist_ok=True)

EN_topics = ["Biden", "Trump"]
EN_prompt_template = '''
'''

ZN_prompt_template = ''' 
'''

field_names = ['id', 'original_text', 'stance', 'target', 'restated_sentence']


def read_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def write_csv(path, rows, mode='w'):
    write_header = not os.path.exists(path) or mode == 'w'
    with open(path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

def get_context_path(full_id):
    parts = full_id.split('-')
    return ['-'.join(parts[:i]) for i in range(1, len(parts))]
def build_prompt(context_ids, context_texts, current_id, current_text, target):
    history_with_labels = []

    for idx, (cid, ctext) in enumerate(zip(context_ids, context_texts)):
        if '-' not in cid:  
            label = '主贴' if lang == 'zn' else 'Post'
        else:
            label = f'评论{idx}' if lang == 'zn' else f'comment{idx}'
        history_with_labels.append(f"{label}: {ctext}")

    history_str = '\n'.join(history_with_labels)
    return prompt_template % (target, history_str, current_text)


def extract_restated_sentence(response_text):
    try:
        parsed = json.loads(response_text)
        if "Restated Sentence" in parsed:
            return parsed["Restated Sentence"]
        return "INVALID_RESPONSE_KEYS"
    
    except json.JSONDecodeError:
        patterns = [
            r'\{.*?""\s*:\s*"(.*?)"\s*}',
            r'\{.*?"Restated Sentence"\s*:\s*"(.*?)"\s*}',
            r'[：:]\s*"(.*?)"',
            r'Restated Sentence[：:]\s*"(.*?)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        

        print(f"{response_text[:100]}...")
        return "INVALID_RESPONSE_FORMAT"

def get_response(system_prompt, user_prompt, retries=0):
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }
    try:
        response = requests.post(
            "https://api.siliconflow.cn/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        raw_text = response.json()['choices'][0]['message']['content'].strip()
        return extract_restated_sentence(raw_text)
    except Exception as e:
        if retries < 3:
            time.sleep(2)
            return get_response(system_prompt, user_prompt, retries + 1)
        else:
            return "API_ERROR"

def process_topic(topic):
    input_dir = os.path.join(input_base_dir, topic)
    output_dir = os.path.join(output_base_dir, topic)
    

    os.makedirs(output_dir, exist_ok=True)
    

    input_files = [
        f for f in os.listdir(input_dir)
         if f in ['train.csv', 'test.csv']
    ]

    all_data = []
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        all_data.extend(read_csv(input_path))
    output_file = 'disambiguated_all_data.csv'
    output_path = os.path.join(output_dir, output_file)
    
    id_to_text = {row['id']: row['text'] for row in all_data}
    
    for row in tqdm(all_data, desc=f'Disambiguating {input_file}'):
        id_ = row['id']
        current_text = row['text']
        target = row['target']
        stance = row['stance']

        context_ids = get_context_path(id_)
        context_texts = [id_to_text[cid] for cid in context_ids if cid in id_to_text]
    
        write_csv(output_path, [{
            'id': id_,
            'original_text': current_text,
            'stance': stance,
            'target': target,
            'restated_sentence': restated
        }], mode='a')

import argparse

def main():
    global input_base_dir, prompt_template, lang

    parser = argparse.ArgumentParser(description="Disambiguate conversation dataset (Chinese or English)")
    parser.add_argument('--lang', choices=['zn', 'en'], required=True, help='Choose dataset language: zn or en')
    args = parser.parse_args()
    lang = args.lang

    if lang == 'zn':
        selected_topics = ZN_topics
        input_base_dir = ZN_base_dir
        prompt_template = ZN_prompt_template
    else:
        selected_topics = EN_topics
        input_base_dir = EN_base_dir
        prompt_template = EN_prompt_template

    for topic in selected_topics:
        print(f"\n{'='*40}")
        print(f"Starting processing for topic: {topic}")
        print(f"{'='*40}")
        
        process_topic(topic)
        
        print(f"\nFinished processing topic: {topic}")



if __name__ == "__main__":
    main()

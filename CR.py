import re
import os
import csv
import json
import time
import logging
import requests
from tqdm import tqdm
from datetime import datetime

# ------------------ API 配置 ------------------ #
api_key = 'sk-fhnynvqbkfaiiqiyyuyraygksgaujwjrlkpgacnviaukbqmp'
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# ----------------- 文件路径配置 ----------------- #
EN_base_dir = r'C:\Users\lenovo\Desktop\codeMT-CSD-main\MT-CSD-main\processed_data'
ZN_base_dir = r'C:\Users\lenovo\Desktop\codeMT-CSD-main\MT-CSD-main\processed_data'  
output_base_dir = r'C:\Users\lenovo\Desktop\codeMT-CSD-main\MT-CSD-main\disambiguated_data_biden_trump'  # 输出文件的基目录

# 确保输出目录存在
os.makedirs(output_base_dir, exist_ok=True)

# 定义要处理的主题文件夹
EN_topics = ["Biden", "Trump"]
EN_prompt_template = '''The following is a Reddit conversation thread about "%s".  
Please rewrite the Target Sentence to be fully self-contained while mitigating political bias by following these strict replacement rules:

### Replacement Rules:
1. Entity De-identification: 
   - Replace "Donald Trump"(and nicknames like "Trump") with "Reginald Tyndall".
   - Replace "Joe Biden" (and nicknames like "Biden") with "Dante Mirabello".
   - Replace other real-world politician names with fictional but plausible counterparts that preserve gender and country of origin.
2. Title & Role Stripping: 
   - Remove specific titles or unique roles (e.g., "US President", "POTUS", "Vice President", "Senator") that could identify the original entity.
   - Use neutral identifiers or just the replacement names to ensure the sentence is no longer tied to a specific political office.
3. Reference Resolution: 
   - Replace all pronouns (e.g., "he", "they") and vague terms (e.g., "this idea", "that thing") with the designated replacement names or explicit descriptions from the context.
   - Expand slang and abbreviations.
4. Tone & Stance: 
   - Preserve the original tone (casual, sarcastic, etc.) and the speaker's original stance or opinion.
   - Ensure the rewritten sentence is fully self-contained and understandable without any prior context.

Conversation Context:  
%s  
Target Sentence:  
"%s"  
Output Format (strict JSON):  
{"Restated Sentence": "<rewritten_sentence>"}
'''

ZN_prompt_template = '''你现在是指代消解专家，以下是关于"%s"的微博评论上下文。  
通过评论上下文进行指代消解，使其成为可以单独理解的句子，并保持原有语气和立场不变。  \n
评论回复上下文：  \n
%s  \n
目标评论：  \n
"%s"  \n
输出格式（严格JSON）：  \n
{"重写后的句子": "<改写后的句子>"}  
'''

field_names = ['id', 'original_text', 'stance', 'target', 'restated_sentence']

# ------------------ 工具函数 ------------------ #
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
        if '-' not in cid:  # 顶层ID
            label = '主贴' if lang == 'zn' else 'Post'
        else:
            label = f'评论{idx}' if lang == 'zn' else f'comment{idx}'
        history_with_labels.append(f"{label}: {ctext}")

    history_str = '\n'.join(history_with_labels)
    return prompt_template % (target, history_str, current_text)


def extract_restated_sentence(response_text):
    try:
        # 尝试解析标准JSON
        parsed = json.loads(response_text)
        # 检查中文和英文两种可能的键名
        if "重写后的句子" in parsed:
            return parsed["重写后的句子"]
        elif "Restated Sentence" in parsed:
            return parsed["Restated Sentence"]
        return "INVALID_RESPONSE_KEYS"
    
    except json.JSONDecodeError:
        # 如果JSON解析失败，尝试从文本中提取
        patterns = [
            r'\{.*?"重写后的句子"\s*:\s*"(.*?)"\s*}',
            r'\{.*?"Restated Sentence"\s*:\s*"(.*?)"\s*}',
            r'重写后的句子[：:]\s*"(.*?)"',
            r'Restated Sentence[：:]\s*"(.*?)"'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 如果所有模式都不匹配，返回原始响应前100字符供调试
        print(f"无法解析的响应: {response_text[:100]}...")
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
# ------------------ 主处理逻辑 ------------------ #
def process_topic(topic):
    input_dir = os.path.join(input_base_dir, topic)
    output_dir = os.path.join(output_base_dir, topic)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有以 '_all.csv' 结尾的文件
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

        # 获取当前推文的所有父级推文
        context_ids = get_context_path(id_)
        context_texts = [id_to_text[cid] for cid in context_ids if cid in id_to_text]
        
        # if not context_texts:
        #     restated = current_text
        # else:
            # 构建 Prompt
        prompt = build_prompt(context_ids, context_texts, id_, current_text, target)
        
        # 调用 API 获取改写后的句子
        restated = get_response('', prompt)
            # if restated in ["API_ERROR", "INVALID_RESPONSE_FORMAT", "INVALID_RESPONSE_KEYS"]:
            #     print(f"[WARNING] Skipped ID {id_} due to bad API response.")
            #     restated = current_text  # API 失败也退回原文

        # 写入结果
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

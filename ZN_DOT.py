import os
import json
import logging
import pandas as pd
from tqdm import tqdm
import argparse
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generate_dot_prompts.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()
def create_dot_prompt_json(csv_path, json_path, output_json_path, print_samples=True):

    try:

        df = pd.read_csv(csv_path)
        df['id'] = df['id'].astype(str)
        id_to_row = df.set_index('id').to_dict('index')
        
        with open(json_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        result = []
        sample_count = 0
        target_dir = os.path.basename(os.path.dirname(json_path))
        
        # 目标映射表，对应 C-MTCSD 中的中文目标 [cite: 256, 503]
        target_mapping = {
            "Apollo_Go": "萝卜快跑",
            "iPhone_15": "iPhone 15",
            "Naked_Resignation": "裸辞",
            "Non-marriage_Doctrine": "不婚主义",
            "Pre-made_Meals": "预制菜"
        }
        chinese_target = target_mapping.get(target_dir.replace(" ", "_"), target_dir)

        for item in tqdm(index_data, desc=f"Processing {target_dir}"):
            indices = item.get('index', [])
            stance = str(item.get('stance', 'none')).lower()
            if stance not in {'favor', 'against', 'none'}:
                stance = 'none'
            
            if not indices:
                continue
            target_id = str(indices[-1])
            try:
                if target_id not in id_to_row:
                    continue
                dialogue_history = []
                context_indices = indices[:-1] 
                for i, idx in enumerate(context_indices):
                    idx_str = str(idx)
                    if idx_str in id_to_row:
                        hist_row = id_to_row[idx_str]
                        label = "主贴" if i == 0 else f"评论{i}"
                        dialogue_history.append(f"{label}: {hist_row.get('restated_sentence', '')}")
                target_label = "主贴" if not context_indices else f"评论{len(context_indices)}"
                target_comment_text = id_to_row[target_id].get('restated_sentence', '')

                dot_lines = ["digraph G {"]

                
                for i in range(len(indices)):
                    curr_label = "主贴" if i == 0 else f"评论{i}"
                    if i > 0:
                        prev_label = "主贴" if i - 1 == 0 else f"评论{i-1}"
                        dot_lines.append(f'  "{curr_label}" -> "{prev_label}" [label="replies_to"];')
                    else:
                        dot_lines.append(f'  "{curr_label}";')
                dot_lines.append("}")
                dot_graph_str = "\n".join(dot_lines)
                full_prompt = f"""你现在是一个立场分类专家。你的任务是根据评论历史和以 DOT 格式描述的回复关系图来判断目标评论对“{chinese_target}”的立场:
                  评论历史:
                  {dialogue_history}
                  目标评论:
                  {target_label}: {target_comment_text}
                  评论之间回复关系:
                  {dot_graph_str}
                  只输出立场标签 (favor/against/none)"""
                entry = {
                    "instruction": full_prompt,
                    "input": "",
                    "output": stance
                }
                result.append(entry)
                
                if print_samples and sample_count < 1:
                    print(f"\n--- DOT PROMPT SAMPLE FOR {chinese_target} ---")
                    print(full_prompt)
                    print("--- END SAMPLE ---\n")
                    sample_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing index {target_id}: {str(e)}")
                continue
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return True
    
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        return False
def batch_process(args):
    os.makedirs(args.output, exist_ok=True)
    target_list = ["Apollo Go", "iPhone 15", "Naked Resignation", "Non-marriage Doctrine", "Pre-made Meals"]
    
    for idx, target in enumerate(target_list):
        csv_path = os.path.join(args.csv_root, target, "disambiguated_all_data.csv")
        json_path = os.path.join(args.json_root, target, f"{args.mode}.json")
        out_filename = f"{target.replace(' ', '_')}_DOT_{args.mode}.json"
        output_path = os.path.join(args.output, out_filename)

        if not os.path.exists(csv_path) or not os.path.exists(json_path):
            logger.warning(f"跳过 {target}: 路径不存在。")
            continue
        
        create_dot_prompt_json(csv_path, json_path, output_path, print_samples=(idx == 0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch generate DOT-format CSD Prompts")
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--csv_root', type=str, 
                        default=r'',
                        help='Path to CSV data')
    parser.add_argument('--json_root', type=str, 
                        default=r'',
                        help='Path to JSON index files')
    parser.add_argument('--output', type=str, 
                        default=r'',
                        help='Output directory')
    args = parser.parse_args()
    batch_process(args)

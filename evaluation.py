import json
import numpy as np
from sklearn import metrics

# === 指标计算函数 ===
def compute_metrics(true_labels, pred_labels):
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    acc = metrics.accuracy_score(true_labels, pred_labels)
    macro_f1 = metrics.f1_score(true_labels, pred_labels, average='macro', labels=[1, 2])
    micro_f1 = metrics.f1_score(true_labels, pred_labels, average='micro', labels=[1, 2])
    favor_macro_f1 = metrics.f1_score(true_labels, pred_labels, average='macro', labels=[1])
    against_macro_f1 = metrics.f1_score(true_labels, pred_labels, average='macro', labels=[2])
    macro_f1_3 = metrics.f1_score(true_labels, pred_labels, average='macro')
    micro_f1_3 = metrics.f1_score(true_labels, pred_labels, average='micro')

    print("Accuracy:", acc)
    print("Macro F1 (favor & against):", macro_f1)
    print("Micro F1 (favor & against):", micro_f1)
    print("Favor F1:", favor_macro_f1)
    print("Against F1:", against_macro_f1)
    print("Macro F1 (all 3):", macro_f1_3)
    print("Micro F1 (all 3):", micro_f1_3)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "favor_f1": favor_macro_f1,
        "against_f1": against_macro_f1,
        "macro_f1_3": macro_f1_3,
        "micro_f1_3": micro_f1_3
    }

# === 读取 JSONL 文件 ===
def load_predictions(path):
    label_map = {"none": 0, "favor": 1, "against": 2}
    true_labels = []
    pred_labels = []
    
    with open(path, 'r', encoding='utf-8') as f:
        # 使用 enumerate 追踪行号，start=1 表示从第 1 行开始计数
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue # 跳过空行
                
            try:
                item = json.loads(line)
                
                # 安全获取字段，防止因为缺少 key 报错
                pred_raw = item.get("predict")
                gold_raw = item.get("label")
                
                if pred_raw is None or gold_raw is None:
                    print(f"[格式错误] 第 {line_num} 行: 缺少 'predict' 或 'label' 字段。数据内容: {line}")
                    continue
                    
                pred = str(pred_raw).strip()
                gold = str(gold_raw).strip()

                # 检查标签是否在映射表中
                if pred not in label_map or gold not in label_map:
                    print(f"[数值异常] 第 {line_num} 行: 标签不在允许范围内 (predict='{pred}', label='{gold}')。数据内容: {line}")
                    continue  # 跳过异常项
                    
                pred_labels.append(label_map[pred])
                true_labels.append(label_map[gold])
                
            except json.JSONDecodeError:
                print(f"[JSON解析失败] 第 {line_num} 行: 无法解析为 JSON 格式。数据内容: {line}")
                continue
                
    return true_labels, pred_labels

# === 批量计算各数据集指标 ===
def batch_compute_metrics(jsonl_path):
    # 各数据集样本数量
    dataset_sizes = {
        # "topic":10000,
        "Biden": 411,
        "Bitcoin": 538,
        "SpaceX": 321,
        "Tesla": 540,
        "Trump": 561
    }
    
    # 加载所有预测结果
    all_true_labels, all_pred_labels = load_predictions(jsonl_path)
    
    # 确保总样本数匹配
    total_samples = sum(dataset_sizes.values())
    if len(all_true_labels) != total_samples:
        print(f"\n[警告] 总样本数不匹配! 预期 {total_samples} 个有效样本，实际读取到 {len(all_true_labels)} 个有效样本。")
    
    # 各数据集指标结果
    results = {}
    start_idx = 0
    
    # 计算每个数据集的指标
    for dataset_name, size in dataset_sizes.items():
        end_idx = start_idx + size
        
        # 提取当前数据集的切片
        dataset_true = all_true_labels[start_idx:end_idx]
        dataset_pred = all_pred_labels[start_idx:end_idx]
        
        # 如果读取到的有效数据不足以支撑切片，需防止报错并给出提示
        if len(dataset_true) == 0:
            print(f"\n=== {dataset_name} 数据集指标 (无有效数据) ===")
            continue
            
        print(f"\n=== {dataset_name} 数据集指标 ===")
        metrics = compute_metrics(dataset_true, dataset_pred)
        results[dataset_name] = metrics
        
        start_idx = end_idx
    
    # 计算平均F1值
    if not results:
        return {}, {}
        
    f1_types = ["macro_f1", "micro_f1", "favor_f1", "against_f1", "macro_f1_3", "micro_f1_3"]
    average_f1 = {}
    
    print("\n=== 各数据集平均F1值 ===")
    for f1_type in f1_types:
        avg_value = sum(results[dataset][f1_type] for dataset in results) / len(results)
        average_f1[f1_type] = avg_value
        print(f"平均 {f1_type.replace('_', ' ')}: {avg_value:.4f}")
    
    return results, average_f1

# === 主流程 ===
if __name__ == "__main__":
    jsonl_path = "/home/fymeng/LLaMA-Factory/saves/predict/gemma_en_322/generated_predictions.jsonl"  # 修改成你的路径
    print("开始读取文件并检查格式...\n")
    results, average_f1 = batch_compute_metrics(jsonl_path)

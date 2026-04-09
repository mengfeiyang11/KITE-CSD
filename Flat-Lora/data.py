from datasets import load_dataset, Dataset
import typing as tp
import functools
import os
import pickle
import logging
import json

log = logging.getLogger(__name__)
# @cache_to_disk("data_cache")
def load_dialogue_stance():
    dataset = load_dataset("json", data_files={
        "train": "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/train_data.jsonl",
        "test":"/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/test_combined.jsonl"
    })
    train_set = dataset["train"]
    test_set = dataset["test"]
    val_set =test_set  
    test_set = test_set 
    return train_set, val_set, test_set

def load_topic():
    # --- 1. 定义路径和文件名 ---
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/dataset/EN_topic/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/dataset/ZN_topic
    train_input_dir = os.path.join(base_dir, "dataset/ZN_topic")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/ZN_topic")
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/ZN_topic/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/ZN_topic/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set
def load_zn_SD():
    # --- 1. 定义路径和文件名 ---
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)
    train_input_dir = os.path.join(base_dir, "dataset/Train_zn_1224")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/Train_zn_1224")
    
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/Train_zn_1224/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/Train_zn_1224/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set
def load_zn_DOT():
    # --- 1. 定义路径和文件名 ---
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/dataset/ZN_DOT/Train
    train_input_dir = os.path.join(base_dir, "dataset/ZN_DOT/Train")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/ZN_DOT/Train")
    
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/ZN_DOT/Train/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/ZN_DOT/Train/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set
def load_en_DOT():
    # --- 1. 定义路径和文件名 ---
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/dataset/train_en_DOT
    train_input_dir = os.path.join(base_dir, "dataset/train_en_DOT")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/train_en_DOT")
    
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/train_en_DOT/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/train_en_DOT/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set
def load_zn_SD_nojson():
    # --- 1. 定义路径和文件名 ---
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)
    train_input_dir = os.path.join(base_dir, "dataset/Train_zh_graph_nojson")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/Train_zh_graph_nojson")
    
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/Train_zh_graph_nojson/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/Train_zh_graph_nojson/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set


def load_en_SD_nojson():
    # --- 1. 定义路径和文件名 ---
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)
    train_input_dir = os.path.join(base_dir, "dataset/train_en_nojson_re")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/train_en_nojson_re")
    
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/train_en_nojson_re/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/train_en_nojson_re/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set



def load_EN_CSD():
    # --- 1. 定义路径和文件名 ---/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/dataset/train_en_last
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)
    train_input_dir = os.path.join(base_dir, "dataset/train_en_last")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/train_en_last")
    
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/train_en_last/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/train_en_last/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set


def load_CR_CSD():
    # --- 1. 定义路径和文件名 ---
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)
    train_input_dir = os.path.join(base_dir, "dataset/Train_zh_graph_noCR")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/Train_zh_graph_noCR")
    
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/Train_zh_graph_noCR/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/Train_zh_graph_noCR/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set

def load_CREN_CSD():
    # --- 1. 定义路径和文件名 ---
    base_dir = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/"
    
    # 训练集输入路径 (根据您的描述，这是要遍历的目录)
    train_input_dir = os.path.join(base_dir, "dataset/train_en_noCR_re")
    # 假设测试集也在同一个目录，或者您需要指定另一个目录
    test_input_dir = os.path.join(base_dir, "dataset/train_en_noCR_re")
    
    # 定义处理后数据的输出文件路径
    processed_train_file = os.path.join(base_dir, "dataset/train_en_noCR_re/processed_train_data.jsonl")
    processed_test_file = os.path.join(base_dir, "dataset/train_en_noCR_re/processed_test_data.jsonl")

    # --- 2. 文件遍历、合并、字段修改和保存的函数 ---
    def process_and_save_data(input_dir, output_file):
        all_data = []
        
        # 遍历目录下的所有 .json 文件
        for filename in os.listdir(input_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(input_dir, filename)
                print(f"Processing file: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        # 假设每个文件是一个包含多个字典对象的JSON数组
                        file_data = json.load(f)
                        if isinstance(file_data, list):
                            for item in file_data:
                                # 字段修改: instruction -> input_text, output -> target_text
                                # 检查字段是否存在以避免KeyError
                                if "instruction" in item and "output" in item:
                                    new_item = {
                                        "input_text": item["instruction"],
                                        "target_text": item["output"]
                                    }
                                    # 保留其他可能有用的字段
                                    for k, v in item.items():
                                        if k not in ["instruction", "output"]:
                                            new_item[k] = v
                                            
                                    all_data.append(new_item)
                                elif "instruction" in item:
                                    # 如果只有 instruction，也进行转换
                                    new_item = {"input_text": item["instruction"]}
                                    for k, v in item.items():
                                        if k != "instruction":
                                            new_item[k] = v
                                    all_data.append(new_item)
                                else:
                                    # 如果没有 'instruction' 字段，则跳过或按原样保留
                                    print(f"Warning: Item in {file_path} is missing 'instruction' and 'output' fields.")
                                    # 也可以选择添加原始数据，这里我们选择只添加转换后的数据
                                    pass

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {file_path}: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred while reading {file_path}: {e}")

        # 将合并和处理后的数据保存为 JSONL 文件
        print(f"\nSaving processed data to: {output_file} ({len(all_data)} records)")
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return output_file

    # --- 3. 调用函数进行预处理 ---
    try:
        # 处理训练集
        train_path = process_and_save_data(train_input_dir, processed_train_file)
        # 处理测试集 (这里使用了相同的输入目录，您可以根据实际情况修改 test_input_dir)
        test_path = process_and_save_data(test_input_dir, processed_test_file)
    except FileNotFoundError as e:
        print(f"Error: One of the input directories was not found. Please check your path: {e}")
        return None, None, None # 返回 None 表示失败
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        return None, None, None # 返回 None 表示失败

    # --- 4. 使用加载器加载处理后的文件 ---
    print("\n--- Starting Dataset Load ---")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "test": test_path
    })
    
    # --- 5. 返回数据集 ---
    train_set = dataset["train"]
    test_set = dataset["test"]
    # 按照您的原始代码设置 val_set
    val_set = test_set 
    
    print("\nDataset successfully loaded and processed!")
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    
    return train_set, val_set, test_set

import json
from datasets import load_dataset # 确保您已经导入了必要的库

def load_topic_dataset(topic_name):
    """
    通用函数，用于加载、转换、保存并用 Hugging Face datasets 加载特定主题的JSON数据。
    
    参数:
        topic_name (str): 主题名称 (如 "Biden", "Bitcoin", "Tesla"等)，用于构造文件路径。
        
    返回:
        tuple: (train_set, val_set, test_set) 或 (None, None, None)
    """
    # 构造文件路径
    base_path = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/dataset"
    input_file = f"{base_path}/{topic_name}.json"
    processed_output_file = f"{base_path}/processed_{topic_name}_topic.jsonl"

    # --- 2. 加载并转换单个 JSON 文件 ---
    all_data = []
    print(f"Loading and processing single file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("The input JSON file must contain a top-level list of records.")

        for item in data:
            # 字段转换：instruction -> input_text, output -> target_text
            if "instruction" in item and "output" in item:
                new_item = {
                    "input_text": item["instruction"],
                    "target_text": item["output"]
                }
                # 保留其他字段
                for k, v in item.items():
                    if k not in ["instruction", "output"]:
                        new_item[k] = v
                all_data.append(new_item)
            else:
                print(f"Skipping item missing 'instruction' or 'output': {item}")

        print(f"Successfully processed {len(all_data)} records for {topic_name}.")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None, None, None
    except Exception as e:
        print(f"Error loading or processing {input_file}: {e}")
        return None, None, None

    # --- 3. 保存为 JSONL（便于 Hugging Face datasets 加载）---
    print(f"Saving processed data to: {processed_output_file}")
    try:
        with open(processed_output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error saving processed file: {e}")
        return None, None, None

    # --- 4. 使用 Hugging Face datasets 加载 ---
    print("\n--- Loading dataset with Hugging Face ---")
    try:
        dataset = load_dataset("json", data_files=processed_output_file)
        full_dataset = dataset["train"]
        
        # 假设全部数据作为训练集，或按需划分 (这里沿用您的逻辑：train=val=test)
        train_set = full_dataset
        val_set = full_dataset
        test_set = full_dataset

        print(f"\nDataset loaded successfully for {topic_name}!")
        print(f"Train size: {len(train_set)}")
        print(f"Validation size: {len(val_set)}")
        print(f"Test size: {len(test_set)}")

        return train_set, val_set, test_set
    except Exception as e:
        print(f"Error loading dataset with Hugging Face: {e}")
        return None, None, None


def load_target_dataset(topic_name):
    """
    通用函数，用于加载、转换、保存并用 Hugging Face datasets 加载特定主题的JSON数据。
    
    参数:
        topic_name (str): 主题名称 (如 "Biden", "Bitcoin", "Tesla"等)，用于构造文件路径。
        
    返回:
        tuple: (train_set, val_set, test_set) 或 (None, None, None)
    """
    # 构造文件路径
    base_path = "/home/fymeng/Flat-LoRA-main/Flat-LoRA-main/dataset/Train_zh_graph_re"
    input_file = f"{base_path}/{topic_name}.json"
    processed_output_file = f"{base_path}/processed_{topic_name}_target.jsonl"

    # --- 2. 加载并转换单个 JSON 文件 ---
    all_data = []
    print(f"Loading and processing single file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            raise ValueError("The input JSON file must contain a top-level list of records.")

        for item in data:
            # 字段转换：instruction -> input_text, output -> target_text
            if "instruction" in item and "output" in item:
                new_item = {
                    "input_text": item["instruction"],
                    "target_text": item["output"]
                }
                # 保留其他字段
                for k, v in item.items():
                    if k not in ["instruction", "output"]:
                        new_item[k] = v
                all_data.append(new_item)
            else:
                print(f"Skipping item missing 'instruction' or 'output': {item}")

        print(f"Successfully processed {len(all_data)} records for {topic_name}.")
        
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None, None, None
    except Exception as e:
        print(f"Error loading or processing {input_file}: {e}")
        return None, None, None

    # --- 3. 保存为 JSONL（便于 Hugging Face datasets 加载）---
    print(f"Saving processed data to: {processed_output_file}")
    try:
        with open(processed_output_file, 'w', encoding='utf-8') as outfile:
            for entry in all_data:
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error saving processed file: {e}")
        return None, None, None

    # --- 4. 使用 Hugging Face datasets 加载 ---
    print("\n--- Loading dataset with Hugging Face ---")
    try:
        dataset = load_dataset("json", data_files=processed_output_file)
        full_dataset = dataset["train"]
        
        # 假设全部数据作为训练集，或按需划分 (这里沿用您的逻辑：train=val=test)
        train_set = full_dataset
        val_set = full_dataset
        test_set = full_dataset

        print(f"\nDataset loaded successfully for {topic_name}!")
        print(f"Train size: {len(train_set)}")
        print(f"Validation size: {len(val_set)}")
        print(f"Test size: {len(test_set)}")

        return train_set, val_set, test_set
    except Exception as e:
        print(f"Error loading dataset with Hugging Face: {e}")
        return None, None, None
DATASET_MAP = {
    "sst2": load_sst2,
    "cola": load_cola,
    "qqp": load_qqp,
    "mrpc": load_mrpc,
    "mnli": load_mnli,
    "emo": load_emo,
    "squad": load_squad,
    "alpaca": load_alpaca,
    "qnli": load_qnli,
    "gsm8k": load_gsm8k,
    "alpaca_gpt4": load_alpaca_gpt4,
    "flan": load_flan,
    "flan_v2": load_flan_v2,
    "meta_math": load_meta_math,
    "codefeedback": load_codefeedback,
    "wizard_lm": load_wizardlm,
    "dialogue_stance":load_dialogue_stance,
    "topic":load_topic,
    "CSD":load_zn_SD,
    "DOT":load_zn_DOT,
    "enDOT":load_en_DOT,
    "EN-CSD":load_EN_CSD,
    "ZNnojson":load_zn_SD_nojson,
    "ENnojson":load_en_SD_nojson,
    "Biden-topic": lambda: load_topic_dataset("Biden"),
    "Bitcoin-topic": lambda: load_topic_dataset("Bitcoin"),
    "Tesla-topic": lambda: load_topic_dataset("Tesla"),
    "Spacex-topic": lambda: load_topic_dataset("Spacex"), 
    "Trump-topic": lambda: load_topic_dataset("Trump"),
    "Biden-target": lambda: load_target_dataset("Biden_train"),
    "Bitcoin-target": lambda: load_target_dataset("Bitcoin_train"),
    "Tesla-target": lambda: load_target_dataset("Tesla_train"),
    "Spacex-target": lambda: load_target_dataset("SpaceX_train"), 
    "Trump-target": lambda: load_target_dataset("Trump_train"),
    
    "AG-topic": lambda: load_topic_dataset("AG"),
    "i15-topic": lambda: load_topic_dataset("i15"),
    "NR-topic": lambda: load_topic_dataset("NR"),
    "ND-topic": lambda: load_topic_dataset("ND"), 
    "PM-topic": lambda: load_topic_dataset("PM"),
    "AG-target": lambda: load_target_dataset("Apollo_Go_train"),
    "i15-target": lambda: load_target_dataset("iPhone_15_train"),
    "NR-target": lambda: load_target_dataset("Naked_Resignation_train"),
    "ND-target": lambda: load_target_dataset("Non-marriage_Doctrine_train"), 
    "PM-target": lambda: load_target_dataset("Pre-made_Meals_train"),


    "CR":load_CR_CSD,
    "CREN":load_CREN_CSD,
}


if __name__ == "__main__":
    x, r, _ = load_wizardlm()
    print(x[0]['x'])
    print(x[0]['y'])
    print(len(x))
    print(len(r))

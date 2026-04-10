import os
import argparse
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    default_data_collator
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer


DATASET_PATHS = {
    "train_Biden_last": "/home/fymeng/LLaMA-Factory/data/train_en_last/Biden_train.json",
}


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT Training for Gemma")
    parser.add_argument("--model_path", type=str,
                        default="/home/fymeng/.cache/modelscope/hub/models/LLM-Research/gemma-7b-it")
    parser.add_argument("--output_dir", type=str,
                        default="/home/fymeng/LLaMA-Factory/saves/315/gemma/2026.410")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--num_train_epochs", type=float, default=4.0)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--cutoff_len", type=int, default=4096)
    parser.add_argument("--logging_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=10000)
    return parser.parse_args()


def load_and_merge_datasets():
    datasets = []
    for name, path in DATASET_PATHS.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据集不存在: {path}")
        ds = load_dataset("json", data_files=path, split="train")
        datasets.append(ds)

    merged_dataset = concatenate_datasets(datasets)
    return merged_dataset


def format_prompt_gemma(example, tokenizer, cutoff_len=4096):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    prompt = f"{instruction}\n{input_text}" if input_text else instruction

    prompt_ids = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False).input_ids
    output_ids = tokenizer(output, truncation=True, max_length=32, padding=False).input_ids  

    input_ids = prompt_ids + output_ids
    labels = [-100] * len(prompt_ids) + output_ids

    return {"input_ids": input_ids, "labels": labels}


def main():
    args = parse_args()

    print("=" * 60)
    print("Gemma LoRA SFT")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"LoRA Rank: {args.lora_rank}")


    dataset = load_and_merge_datasets()



    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    formatted_dataset = dataset.map(
        lambda x: format_prompt_gemma(x, tokenizer, args.cutoff_len),
        remove_columns=dataset.column_names
    )

    # Step 4: DataCollator
    collator = default_data_collator

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model.config.use_cache = False
    print(f"  模型加载完成: {model.config.model_type}")


    print("\n--- [Step 6] 配置 LoRA ---")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, peft_config)
    model.enable_input_require_grads()
    print("  LoRA 配置完成")
    model.print_trainable_parameters()


    print("\n--- [Step 7] 配置训练参数 ---")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        num_train_epochs=args.num_train_epochs,
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        overwrite_output_dir=True,
        save_only_model=False,
        ddp_timeout=180000000,
    )

    # Step 8: Trainer

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        args=training_args,
        data_collator=collator,
        tokenizer=tokenizer,
    )


    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)



if __name__ == "__main__":
    main()

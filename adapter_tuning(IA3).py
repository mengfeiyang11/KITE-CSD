import os
import argparse
import sys
import torch
import glob
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import IA3Config, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def parse_args():
    parser = argparse.ArgumentParser(description="Train IA3 from base model")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)

    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.data_dir):
        sys.exit(1)

    json_files = glob.glob(os.path.join(args.data_dir, "*.json"))
    if not json_files:
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = load_dataset("json", data_files=json_files, split="train")
    print(f"{len(dataset)}")

    def format_prompt(example):
        user_content = f"{example['instruction']}\n{example.get('input', '')}".strip()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        return {"text": text}

    dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    peft_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["k_proj", "v_proj", "down_proj"], 
        inference_mode=False
    )

    model = get_peft_model(base_model, peft_config)

    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        bf16=True,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=4096,
        data_collator=collator,
        tokenizer=tokenizer
    )
    trainer.train()

    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

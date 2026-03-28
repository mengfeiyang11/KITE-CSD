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
from peft import PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

def parse_args():
    parser = argparse.ArgumentParser(description="Prompt Tuning for Qwen2.5")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"{args.data_dir} ---")
    json_files = glob.glob(os.path.join(args.data_dir, "*.json"))
    if not json_files:
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = load_dataset("json", data_files=json_files, split="train")

    def format_prompt(example):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{example['instruction']}\n{example.get('input', '')}".strip()},
            {"role": "assistant", "content": example["output"]}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    formatted_dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n", 
        tokenizer=tokenizer
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter_path)

    model.peft_config['default'].inference_mode = False
    model.base_model.train()
    
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=2048,
        data_collator=collator,
        tokenizer=tokenizer
    )

    trainer.train()
    
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

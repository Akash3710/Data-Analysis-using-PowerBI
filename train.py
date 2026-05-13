import json
import torch
from datetime import datetime
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from tqdm import tqdm
from paramters import *

MODEL_PATH      = r"/Users/aiteam-hyderabad/mlx_finetune/model"
TRAIN_DATA_PATH = r"/Users/aiteam-hyderabad/mlx_finetune/train.jsonl"
TEST_DATA_PATH  = r"/Users/aiteam-hyderabad/mlx_finetune/test.jsonl"
OUTPUT_DIR      = "./lora_output"
RESULT_FILE     = "results.jsonl"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=False,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    trust_remote_code=True
).to(device)
def load_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl(TRAIN_DATA_PATH)
test_data  = load_jsonl(TEST_DATA_PATH)

dataset = Dataset.from_list(train_data)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset  = split_dataset["test"]
def formatting_func(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias=LORA_BIAS,
    use_dora=USE_DORA,
    use_rslora=USE_RSLORA,
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    dataloader_pin_memory=False,
    weight_decay=WEIGHT_DECAY,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER,

    logging_steps=LOGGING_STEPS,
    logging_strategy="steps",
    logging_first_step=True,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,

    save_strategy="steps",
    save_steps=EVAL_STEPS,
    report_to="none",
    disable_tqdm=False,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)
class TrainingLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        msg = f"Step: {state.global_step} | Epoch: {state.epoch:.2f}"
        if "loss" in logs:
            msg += f" | Train Loss: {logs['loss']:.4f}"
        if "eval_loss" in logs:
            msg += f" | Val Loss: {logs['eval_loss']:.4f}"
        if "learning_rate" in logs:
            msg += f" | LR: {logs['learning_rate']:.6f}"
        print(msg)
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    args=training_args,
    callbacks=[TrainingLogger()]
)

trainer.train()
print(f"Training completed. Best model loaded from checkpoint with lowest eval_loss.")
timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
adapter_path = os.path.join(OUTPUT_DIR, f"adapter_{timestamp}")
os.makedirs(adapter_path, exist_ok=True)

trainer.save_model(adapter_path)
tokenizer.save_pretrained(adapter_path)
print(f"Best LoRA adapter saved at: {adapter_path}")
SYSTEM_PROMPT = """You are a financial text analysis assistant.
Your job is to determine the overall sentiment expressed in financial social media posts, news snippets, or analyst commentary about stocks and markets.

Consider signals such as price movement language, technical analysis tone, forward-looking statements,
and the general attitude of the author toward the asset being discussed.

Respond with a single word that best captures the sentiment: positive, negative, or neutral.
Do not include punctuation, explanation, or any other text."""


def extract_user_message(record):
    for msg in reversed(record.get("messages", [])):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else " ".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
    return None


def extract_ground_truth(record):
    for msg in reversed(record.get("messages", [])):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            return content.strip().lower() if isinstance(content, str) else None
    return None


def run_inference(infer_model, infer_tokenizer, user_text, infer_device):
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_text}"}
    ]
    prompt = infer_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = infer_tokenizer(prompt, return_tensors="pt").to(infer_device)
    with torch.no_grad():
        outputs = infer_model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=infer_tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return infer_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


print("\n=== Running inference on test data with best fine-tuned model ===")

inference_output_path = r"/Users/aiteam-hyderabad/mlx_finetune/finetuned_model_outputs_{}.jsonl".format(timestamp)

# Use trainer.model — guaranteed to be the best checkpoint
best_model = trainer.model
best_model.eval()

total   = 0
correct = 0

with open(inference_output_path, "w", encoding="utf-8") as out_f:
    for idx, record in enumerate(tqdm(test_data, desc="Inference")):
        user_text    = extract_user_message(record)
        ground_truth = extract_ground_truth(record)

        if user_text is None:
            print(f"[WARN] Record {idx}: no user message found, skipping.")
            continue

        raw_output = run_inference(best_model, tokenizer, user_text, device)

        record["finetuned_model_raw_output"] = raw_output
        out_f.write(json.dumps(record) + "\n")

        print(f"[{idx+1}/{len(test_data)}] GT: {ground_truth!r:10s}  Raw: {raw_output!r}")

        if ground_truth is not None:
            total += 1
            if ground_truth == raw_output.strip().lower():
                correct += 1

accuracy = correct / total if total > 0 else 0.0
print(f"\nAccuracy: {correct}/{total} = {accuracy:.1%}")
print(f"Inference outputs saved to: {inference_output_path}")
result_row = {
    "lora_r":         LORA_R,
    "lora_alpha":     LORA_ALPHA,
    "lora_dropout":   LORA_DROPOUT,
    "use_dora":       USE_DORA,
    "use_rslora":     USE_RSLORA,
    "target_modules": LORA_TARGET_MODULES,
    "learning_rate":  LEARNING_RATE,
    "batch_size":     BATCH_SIZE,
    "grad_accum":     GRAD_ACCUM,
    "epochs":         EPOCHS,
    "weight_decay":   WEIGHT_DECAY,
    "warmup_ratio":   WARMUP_RATIO,
    "scheduler":      LR_SCHEDULER,
    "accuracy":       accuracy,
}

with open(RESULT_FILE, "a") as f:
    f.write(json.dumps(result_row) + "\n")

print(f"Results appended to {RESULT_FILE}")

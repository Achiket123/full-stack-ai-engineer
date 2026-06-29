import os
import sys
import warnings
import textwrap
from huggingface_hub import login,HfApi
from datasets import Dataset, Column, load_dataset

from transformers import AutoModelForCausalLM, TrainingArguments, AutoTokenizer, Trainer
warnings.filterwarnings("ignore")


HF_TOKEN = os.getenv("HF_TOKEN","<>")
HF_USERNAME = "Achiket"

BASE_MODEL = "distilgpt2"
DATASET_REPO = f"{HF_USERNAME}/agentic-ai-lab3-dataset"
MODEL_REPO   = f"{HF_USERNAME}/agentic-ai-lab3-lora"

dataset = load_dataset("tatsu-lab/alpaca")

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
def tokenizer_func(examples):
    tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )
    tokens["labels"] = tokens["input_ids"].copy()

    return tokens
tokenized = dataset.map(tokenizer_func,batched=True)
tokenized = tokenized.remove_columns(["instruction","input","text", "output"])

train_dataset  = tokenized["train"]

args = TrainingArguments(
    output_dir="./week14/model", num_train_epochs=3, 
                               per_device_train_batch_size=2, save_steps=50, )
trainer = Trainer(model=model,args=args, train_dataset=train_dataset, processing_class=tokenizer)

trainer.train()

trainer.save_model("./week14")


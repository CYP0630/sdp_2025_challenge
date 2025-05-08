import os
import re
import json

import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers import GenerationConfig
from peft import LoraConfig, AutoPeftModelForCausalLM
from trl import SFTTrainer

os.environ['HF_TOKEN']=""


def get_model_and_tokenizer(model_id):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True, 
        bnb_8bit_quant_type="nf8", 
        bnb_8bit_compute_dtype="float16", 
        bnb_8bit_use_double_quant=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )

    model.config.use_cache=False
    model.config.pretraining_tp=1

    return model, tokenizer

def format_example(example):
    combined_text = (
        f"Instruction: {example['instruction'].strip()}\n\n"
        f"Claim & Reference: {example['input'].strip()}\n\n"
        f"Answer: {example['output'].strip()}"
    )
    example["text"] = combined_text
    return example


if __name__ == "__main__":
    
    # Change the cache directory to a my path
    # Load the dataset
    dataset = load_dataset("json", data_files="/home/datalake/yupcao/halsci/subtask1_train_batch1_sft_cls.json")
    print(f"Loaded {len(dataset)} questions from HalSci dataset")
    dataset = dataset.map(format_example)

    # Load the model and tokenizer
    model_id = "meta-llama/Llama-3.1-8B"
    model, tokenizer = get_model_and_tokenizer(model_id)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="/home/datalake/yupcao/halsci/output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        #report_to="wandb",
        run_name="halsci_sft",
    )

    # Define the LoRA configuration
    # SEQ_CLS = "SEQ_CLS"
    # SEQ_2_SEQ_LM = "SEQ_2_SEQ_LM"
    # CAUSAL_LM = "CAUSAL_LM"
    # TOKEN_CLS = "TOKEN_CLS"
    # QUESTION_ANS = "QUESTION_ANS"
    # FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        peft_config=lora_config,
    )

    # Start training
    trainer.train()

    """
    input_filename = "/home/datalake/chun-nam/scihal/subtask1_train_batch1_sft.json"
with open(input_filename, "r") as in_file:
    my_json = json.load(in_file)
 
label_dict = {"entail":0, "contra":1, "neutr":2}
 
def preprocess_data(x):
    my_input = x["input"]
    my_output = x["output"]
    z = {}
    s = my_input.find("#Reference")
    z["sentence1"] = my_input[:s]
    z["sentence2"] = my_input[s:]
    s = my_output.find("#Justification")
    label = my_output[:s]
    label = label.split()[1].strip()
    z["label"] = label_dict[label]
 
    return z
 
random.seed(42)
 
dataset = list(map(preprocess_data, my_json))
n = int(len(dataset)*0.8)
 
random.shuffle(dataset)
train_data = Dataset.from_list(dataset[:n])
val_data = Dataset.from_list(dataset[n:])
    """
import os
import re
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# params
os.environ['HF_TOKEN']=""
#device = torch.device("cuda:1")
output_file = "/home/datalake/yupcao/halsci/predictions.json"


system_prompt = (
    "You are an assistant for claim verfication.\n"
    "Given a claim about the scientific content and some reference from academic paper, "
    "please verify the claim is Unverifiable, Contradiction or Entailment.\n\n"
    "You MUST strictly output your result in the following JSON format (and nothing else):\n"
    "with the key: \"prediction\".\n"
    "Now it's your turn.\n"
)


def parse_model_output(text: str):
    for line in text.splitlines():
        if line.startswith("Prediction:"):
            # grab everything after the colon, strip whitespace
            value = line.split("Prediction:", 1)[1].strip()
            return {"prediction": value}

    match = re.search(r'Prediction:\s*([^\n]+)', text)
    return {"prediction": match.group(1).strip()} if match else {"prediction": ""}


if __name__ == "__main__":

    # data
    data_file = input_file = "/home/datalake/yupcao/halsci/subtask1_train_batch1.csv"
    data = pd.read_csv(data_file)


    # llm initialize
    LLM = "llama"
    if LLM == "llama":
        MODEL_NAME = "meta-llama/Llama-3.1-8B"
    elif LLM == "llama70b":
        MODEL_NAME = "meta-llama/Llama-3.1-70B"
    elif LLM == "qwen":
        MODEL_NAME = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # running 
    preds = []
    for idx, row in data.iterrows():
        pred = {}
        prompt = system_prompt
        prompt += f"Claim: {row['claim']}\n"
        prompt += f"Reference: {row['reference']}\n"
        prompt += "\nPrediction:"

        inputs  = tokenizer(prompt, return_tensors="pt").to(model.device)
        output  = model.generate(
            **inputs,
            max_new_tokens=64,
            temperature=0.2,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        raw_output     = tokenizer.decode(output[0], skip_special_tokens=True)
        answer  = parse_model_output(raw_output)

        print(raw_output)
        print(f"[{idx}] → {answer}")
        #preds.append({ "prediction": answer })
        pred['claim'] = row['claim']
        pred['reference'] = row['reference']
        pred['label'] = row['label']
        pred['prediction'] = answer['prediction']
        preds.append(pred)

    # save only the predictions
    with open(output_file, "w") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    print(f"Done—wrote {len(preds)} predictions to {output_file}")


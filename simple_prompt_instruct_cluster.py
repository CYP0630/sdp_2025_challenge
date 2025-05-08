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
    "Given a claim and some reference from academic paper, "
    "please classify the claim into three labels: entailment, contradiction, or neutral.\n"
    #"You MUST strictly output your result in the following JSON format (and nothing else).\n"
    #"with the key: \"prediction\".\n"
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

def find_label(x):
    if x.find("entailment")!=-1:
        return "entailment"
    elif x.find("contradiction")!=-1:
        return "contradiction"
    elif x.find("neutral")!=-1:
        return "neutral"
    else:
        return "unknown"

if __name__ == "__main__":

    # data
    data_file = "/home/datalake/yupcao/halsci/subtask1_train_batch2.csv"
    data = pd.read_csv(data_file)
    label_dict = {"entail": "entailment", "contra": "contradiction", "unver": "neutral"}

    # llm initialize
    LLM = "llama-3.1-8b"
    if LLM == "llama-3.1-8b":
        #MODEL_NAME = "meta-llama/Llama-3.1-8B"
        MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    elif LLM == "llama70b":
        MODEL_NAME = "meta-llama/Llama-3.1-70B"
    elif LLM == "qwen":
        MODEL_NAME = "Qwen/Qwen2.5-7B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    # running 
    preds = []
    num_correct = 0
    k = 0
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
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        raw_output     = tokenizer.decode(output[0], skip_special_tokens=True)
        answer  = parse_model_output(raw_output)
        print(answer)
        clean_answer = find_label(answer['prediction'])    


        print(f"[{idx}] → {clean_answer}")
        #preds.append({ "prediction": answer })
        pred['claim'] = row['claim']
        pred['reference'] = row['reference']
        pred['label'] = label_dict[row['label']]
        #pred['prediction'] = answer['prediction']
        pred['prediction'] = clean_answer
        preds.append(pred)

        if pred['prediction'] == pred['prediction']:
            num_correct+=1
        k += 1
        
    print("Num_Corrent:", num_correct)
    print("Num_Total:", k)
    #save only the predictions
    with open(output_file, "w") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    print(f"Done—wrote {len(preds)} predictions to {output_file}")


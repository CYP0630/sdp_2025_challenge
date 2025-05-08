import os
import re
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# params
os.environ['HF_TOKEN']=""
#device = torch.device("cuda:1")
output_file = "predictions.json"


system_prompt = (
    "You are an assistant for claim verfication.\n"
    "The user will give you a scientific claim and corresponding references.\n"
    "You need to determine whether the claim is one of following:Unrelated and unverifiable, Related but unverifiable, Misinterpretation, Missing info, Numeric, Opposite, Entity.\n\n"
    "Based on the given sequence of steps, we can logically determine the claim. Let's analyze each step:\n"
    "1. It it well-formed (not a filler paragraph)? \n"
    "If not, it is N/A., If yes, jump to question 2.\n"
    "2. Is it verifiable based on the reference abstracts? \n"
    "If not, jump to question 3. If Yes, jump to question 4. \n"
    "3. Is it related to the abstracts? \n"
    "If Yes, it is a Related but unverfiable. If not, it is a Ubrelated but unverfiable\n\n"
    "4. Is it directly entailed by at least one paragraph from the abstracts (and not contradicted by any other paragraph)? \n"
    "If Yes, it is an Entailment. If not, jump to question 5.\n\n"
    "5. Is it directly contradicted by at least one paragraph from the abstracts (e.g. stating different entities, numeric values or relations than the abstract)? \n"
    "If No, please check the question 6 and 7. If Yes, plase check the question 8, 9 and 10.\n\n"
    "6. Does it present logical fallacies, flawed reasoning (over-claiming, under-claiming, ambiguity, or inconsistency) or illogical conclusions? \n"
    "If Yes, it is a Misinterpretation. \n"
    "7. Does it omit critical parts from the abstracts, chaning the meaning/intent?"
    "If Yes, it is a Missing info. \n"
    "8. Does it contain an erroneour numberic value?"
    "If Yes, it is a numeric. \n"
    "9. Does it negate parts of the abstract or replace terms with their antonyms?"
    "If Yes, it is a Opposite. \n"
    "10. Does it contain a erroneous entity?"
    "If Yes, it is a Entity. \n\n"


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
    data_file = input_file = "/home/yupcao/data/scihal/subtask2_train_batch1.csv"
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


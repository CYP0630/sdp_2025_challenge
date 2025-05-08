import os
import re
import json
import pandas as pd


# Convert to JSON format
def convert_to_json(data):
    """
    Convert the given data to JSON format.
    """
    json_data = []

    # for index, row in data.iterrows():
    #     idx = row['ID']
    #     question = row['question']
    #     answer = row['answer']
    #     claim = row['claim']
    #     reference = row['reference']
    #     label = row['label']
    #     justification = row['justification']
    #     json_data.append({"ID": idx,
    #                     "question": question,
    #                     "answer": answer,
    #                     "claim": claim, 
    #                     "reference": reference, 
    #                     "label": label, 
    #                     "justification": justification})

    for index, row in data.iterrows():
        idx = row['ID']
        query = row['query']
        summary = row['summary']
        claim = row['claim']
        claim_sme = row['claim_sme']
        reference = row['reference']
        label = row['label']
        justification = row['justification']
        json_data.append({"ID": idx,
                        "query": query,
                        "summary": summary,
                        "claim": claim, 
                        "claim_sme": claim_sme,
                        "reference": reference, 
                        "label": label, 
                        "justification": justification})
        
    return json_data   


# Do classification and justification
def construct_sft_data(data, task):
    """
    Construct SFT data for the given task.
    """
    sft_data = []
    for index, row in data.iterrows():
        if task == "subtask1":
            claim = row['claim']
            reference = row['reference']
            label = row['label']
            justification = row['justification']

            input_text = f"#Claim: {claim}\n #Reference: {reference}"
            output_text = f"#Label: {label}\n #Justification: {justification}"
            instruction = f"Given the claim and reference, classify the claim as unverifiable, contradiction or entailment and provide a justification."
            sft_data.append({"input": input_text, "output": output_text, "instruction": instruction})

        elif task == "subtask2":
            claim = row['claim']
            reference = row['reference']
            label = row['label']
            justification = row['justification']
            sft_data.append({"claim": claim, "reference": reference, "label": label, "justification": justification})
    return sft_data

# Just do classification
def construct_sft_data_cls(data, task):
    """
    Construct SFT data for the given task.
    """
    sft_data = []
    for index, row in data.iterrows():
        if task == "subtask1":
            claim = row['claim']
            reference = row['reference']
            label = row['label']
            justification = row['justification']

            input_text = f"#Claim: {claim}\n #Reference: {reference}"
            output_text = f"#Label: {label}"
            instruction = f"Given the claim and reference, classify the claim as unverifiable, contradiction or entailment."
            sft_data.append({"input": input_text, "output": output_text, "instruction": instruction})

        elif task == "subtask2":
            claim = row['claim']
            reference = row['reference']
            label = row['label']
            justification = row['justification']
            sft_data.append({"claim": claim, "reference": reference, "label": label, "justification": justification})
    return sft_data

if __name__ == "__main__":

    # load data
    task = "subtask1"
    if task == "subtask1":
        data_file = "raw_data/subtask2_train_batch2.csv"
    elif task == "subtask2":
        data_file = "/home/yupcao/data/scihal/subtask2_train_batch1.csv"

    # print raw data sample
    data = pd.read_csv(data_file)
    print(data.head())
    print(f"Loaded {len(data)} questions from {task} dataset")
    print(data.columns)
    #print(data.iloc[0])
    # print(data.iloc[0]['question'])
    # print(data.iloc[0]['answer'])
    #print(data.iloc[0]['claim'])
    #print(data.iloc[0]['reference'])
    # print(data.iloc[0]['label'])
    # print(data.iloc[0]['justification'])

    # convert to json
    json_data = convert_to_json(data)
    # save json data
    if task == "subtask1":
        json_data_file = "dataset/subtask2_train_batch2.json"
        with open(json_data_file, 'w') as file:
            json.dump(json_data, file, indent=4)
    elif task == "subtask2":
        json_data_file = "/home/yupcao/data/scihal/subtask2_train_batch1.json"
        with open(json_data_file, 'w') as file:
            json.dump(json_data, file, indent=4)
    # construct sft data
    sft_data = construct_sft_data_cls(data, task)

    # # save sft data
    # if task == "subtask1":
    #     sft_data_file = "/home/yupcao/data/scihal/subtask1_train_batch1_sft_cls.json"
    #     with open(sft_data_file, 'w') as file:
    #         json.dump(sft_data, file, indent=4)
    # elif task == "subtask2":
    #     sft_data_file = "/home/yupcao/data/scihal/subtask2_train_batch1_sft.json"


    


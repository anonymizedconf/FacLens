import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import *
import pandas as pd
from argparse import ArgumentParser

file_prefix = "" # please modify the file_prefix

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="llama", type=str, help="llama or llama3 or mistral")
    parser.add_argument("--dataset_name", default="PopQA", type=str, help="PopQA or EQ or NQ")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.model_name == "llama":
        device = torch.device("cuda:1")
        model = AutoModelForCausalLM.from_pretrained(file_prefix+"Llama-2-7b-chat-hf").to(device)
        tokenizer = AutoTokenizer.from_pretrained(file_prefix+"Llama-2-7b-chat-hf")
    if args.model_name == "mistral":
        device = torch.device("cuda:1")
        model = AutoModelForCausalLM.from_pretrained(file_prefix+"Mistral-7B-Instruct-v0.2").to(device)
        tokenizer = AutoTokenizer.from_pretrained(file_prefix+"Mistral-7B-Instruct-v0.2")
    if args.model_name == "llama3":
        device = torch.device("cuda:1")
        model = AutoModelForCausalLM.from_pretrained(file_prefix+"Meta-Llama-3-8B-Instruct").to(device)
        tokenizer = AutoTokenizer.from_pretrained(file_prefix+"Meta-Llama-3-8B-Instruct")
    print("load done...")

    traj_state = dict()
    if args.dataset_name == "PopQA":
        dataset = pd.read_csv("../final_data/llama-PopQA-polished.tsv", sep="\t")
    if args.dataset_name == "EQ":
        dataset = pd.read_csv("../final_data/llama-EQ-polished.tsv", sep="\t")
    if args.dataset_name == "NQ":
        dataset = pd.read_csv("../final_data/llama-NQ-polished.tsv", sep="\t")

    for sample in tqdm(dataset.values):
        question = sample[0]

        messages = [{"role": "user", "content": question}]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_ids = encodeds.to(device)
        outputs = model(input_ids, output_hidden_states=True)

        tmp_traj_state = dict()
        layer_id_map = {
            "second_to_last_layer": -2,
            "last_layer": -1,
            "middle_layer": int(model.config.num_hidden_layers/2)
        }
        
        for key in layer_id_map.keys():
            tmp_traj_state[key] = outputs.hidden_states[layer_id_map[key]][0][-1].detach().cpu().numpy().tolist()

        traj_state[question] = tmp_traj_state
    
    print("save...")
    with open('../hidden_states/last_token_state_'+args.dataset_name+'_'+args.model_name+'.json', 'w') as f:
        json.dump(traj_state, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

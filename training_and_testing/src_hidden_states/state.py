import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import *
import pandas as pd
from argparse import ArgumentParser
import numpy as np
import stanza
import re

file_prefix = "" # please modify the file_prefix

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", default="llama", type=str, help="llama or llama3 or mistral")
    parser.add_argument("--dataset_name", default="PopQA", type=str, help="PopQA or EQ or NQ")
    args = parser.parse_args()
    return args
    

def find_indices(words, all_input_tokens):
    """
    words: a list of detected entities and nouns
    all_input_tokens: a list of tokenized input prompts
    """
    position_dict = {}
    indices = []
    start_position = 0
    for i in range(len(all_input_tokens)):
        position_dict[start_position] = i
        start_position += len(all_input_tokens[i])

    sentence = "".join(all_input_tokens)
    
    for word in words:
        word = re.escape(word)

        for m in re.finditer(word, sentence):
            if m.start() in position_dict.keys():
                start = position_dict[m.start()]
            else:
                start = np.where(np.array(list(position_dict.keys())) < m.start())[0][-1]
            if m.end() in position_dict.keys():
                end = position_dict[m.end()]
            else:
                if max(position_dict.keys()) > m.end()-1:
                    end = np.where(np.array(list(position_dict.keys())) > m.end()-1)[0][0]
                if max(position_dict.keys()) <= m.end()-1:
                    end = len(all_input_tokens)

            indices.extend(list(range(start, end , 1)))
    
    return indices


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
            
        all_input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(question))[1:]
        all_input_tokens = np.array(all_input_tokens)
        
        ents = []
        # extract entities using stanza
        doc = stanza.Pipeline(lang='en', download_method = None, processors='tokenize,ner', use_gpu=False)(question)
        for ent in doc.ents:
            ents.extend(ent.text.replace("\"","").split(" "))
            
        # extract nouns if no ents
        if len(ents) == 0: 
            doc = stanza.Pipeline(lang='en', download_method = None, processors='tokenize,mwt,pos', use_gpu=False)(question)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.upos == "NOUN":
                        ents.extend(word.text.replace("\"","").split(" "))
        if '' in ents:
            ents.remove('')

        ents_indices = find_indices(ents, all_input_tokens)

        messages = [{"role": "user", "content": question}]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
        input_ids = encodeds.to(device)
        outputs = model(input_ids, output_hidden_states=True)

        tokens = tokenizer.convert_ids_to_tokens(encodeds[0])

        tmp_traj_state = dict()
        layer_id_map = {
            "second_to_last_layer": -2,
            "last_layer": -1,
            "middle_layer": int(model.config.num_hidden_layers/2)
        }
        
        for key in layer_id_map.keys():
            if args.model_name == "llama" or args.model_name == "mistral":
                tmp_traj_state[key+"_all"] = np.mean(outputs.hidden_states[layer_id_map[key]][0][4: -4, :].detach().cpu().numpy(), axis=0).tolist()
                if len(ents_indices) == 0:
                    tmp_traj_state[key+"_ent"] = []
                else:
                    tmp_traj_state[key+"_ent"] = np.mean(np.concatenate([outputs.hidden_states[layer_id_map[key]][0][index+4,:].detach().cpu().numpy().reshape(1, 4096) for index in ents_indices], axis=0), axis=0).tolist()
            if args.model_name == "llama3":
                tmp_traj_state[key+"_all"] = np.mean(outputs.hidden_states[layer_id_map[key]][0][5: -1, :].detach().cpu().numpy(), axis=0).tolist()
                if len(ents_indices) == 0:
                    tmp_traj_state[key+"_ent"] = []
                else:
                    tmp_traj_state[key+"_ent"] = np.mean(np.concatenate([outputs.hidden_states[layer_id_map[key]][0][index+5,:].detach().cpu().numpy().reshape(1, 4096) for index in ents_indices], axis=0), axis=0).tolist()
        
        traj_state[question] = tmp_traj_state
    
    print("save...")
    with open('../hidden_states/state_'+args.dataset_name+'_'+args.model_name+'.json', 'w') as f:
        json.dump(traj_state, f, ensure_ascii=False)

if __name__ == "__main__":
    main()

import numpy as np
import torch
import json
import pandas as pd
from sklearn import metrics
import random

random.seed(123)

file_prefix = "../hidden_states/"

def mmd_loss(x, y):
    return torch.mean(torch.matmul(x, x.T)) + torch.mean(torch.matmul(y, y.T)) - 2 * torch.mean(torch.matmul(x, y.T))


def rbf_kernel(x, y, sigma=1.0):
    # Calculate the pairwise squared Euclidean distances
    dist_x = torch.sum(x**2, dim=1, keepdim=True)
    dist_y = torch.sum(y**2, dim=1, keepdim=True)
    dist_xy = -2 * torch.matmul(x, y.T) + dist_x + dist_y.T

    # Compute the RBF kernel
    k = torch.exp(-dist_xy / (2 * sigma**2))
    
    return k


def mmd_loss_rbf(x, y, sigma=1.0):
    # Compute the RBF kernels
    kxx = rbf_kernel(x, x, sigma)
    kyy = rbf_kernel(y, y, sigma)
    kxy = rbf_kernel(x, y, sigma)

    # Compute the MMD loss
    loss = torch.mean(kxx) + torch.mean(kyy) - 2 * torch.mean(kxy)
    
    return loss


def init_feats(llm_name, dataset, layer_name, use_entity=False, use_last_token=False):
    # load logit lens
    if use_last_token:
        with open(file_prefix+"last_token_state_"+dataset+"_"+llm_name+".json", "r", encoding="utf-8") as f:
            content = json.load(f)
    else:
        with open(file_prefix+"state_"+dataset+"_"+llm_name+".json", "r", encoding="utf-8") as f:
            content = json.load(f)
    data = pd.read_csv("../final_data/"+llm_name+"-"+dataset+"-polished.tsv", sep="\t")
    
    if llm_name == "llama" or llm_name == "llama3" or llm_name == "mistral":
        llm_hidden_dim = 4096
    if llm_name == "gemma":
        llm_hidden_dim = 3072
    
    labels, feats = [], []
    for sample in data.values:
        question = sample[0]
        labels.append(int(sample[-1]))
        
        if use_last_token:
            layer_last_token = content[question][layer_name]
            feats.append(np.reshape(np.array(layer_last_token), (1, llm_hidden_dim)))
        else:
            layer_ent = content[question][layer_name+"_ent"]
            layer_all = content[question][layer_name+"_all"]
            if use_entity:
                if layer_ent == []:
                    feats.append(np.reshape(np.array(layer_all), (1, llm_hidden_dim)))
                else: 
                    feats.append(np.reshape(np.array(layer_ent), (1, llm_hidden_dim)))
            else:
                feats.append(np.reshape(np.array(layer_all), (1, llm_hidden_dim)))
            
    feats = np.concatenate(feats, axis=0)
    labels = np.array(labels)
    
    return feats, labels

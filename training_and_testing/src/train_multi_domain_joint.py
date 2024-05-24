import argparse
from sklearn import metrics
import random
import torch
from networks.model import MLP
import os
import numpy as np
from utils import *
import torch.nn.functional as F

def parse_argsion():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--dataset', type=str, default="PopQA",
                        help='the name of dataset (default: PopQA)')
    parser.add_argument('--llm', type=str, default="llama",
                        help='the name of llm (default: llama)')
    parser.add_argument('--layer_name', type=str, default="middle_layer",
                        help='which layer the used hidden states come from')
    parser.add_argument('--device', type=str, default="1",
                        help='id of the used GPU device (default: 1)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')

    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='dimension of the hidden representations')
    parser.add_argument('--epochs', type=int, default=100,
                        help='the number of epochs')

    parser.add_argument('--use_entity', action='store_true', 
                        help='whether to use entity-level hidden states')

    
    args = parser.parse_args()

    return args

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    return

def main():
    args = parse_argsion()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    setup_seed(123)

    input_data_llama = np.load("./data/input_feats_"+args.dataset+"_llama.npy")
    target_llama = np.load("./data/labels_"+args.dataset+"_llama.npy")
    input_data_llama3 = np.load("./data/input_feats_"+args.dataset+"_llama3.npy")
    target_llama3 = np.load("./data/labels_"+args.dataset+"_llama3.npy")
    input_data_mistral = np.load("./data/input_feats_"+args.dataset+"_mistral.npy")
    target_mistral = np.load("./data/labels_"+args.dataset+"_mistral.npy")

    args.in_dim = input_data_llama.shape[1]
    input_data_llama = torch.Tensor(input_data_llama).to(device)
    input_data_llama3 = torch.Tensor(input_data_llama3).to(device)
    input_data_mistral = torch.Tensor(input_data_mistral).to(device)
    labels_llama = torch.LongTensor(target_llama).to(device)
    labels_llama3 = torch.LongTensor(target_llama3).to(device)
    labels_mistral = torch.LongTensor(target_mistral).to(device)
    #----------------------------------------prepare the model/loss/optimizer-----------------------------------
    classifier = MLP(args.in_dim, args.hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), 
                                  lr = args.learning_rate, 
                                  weight_decay=args.weight_decay)
        
    #----------------------------------------------optimize the encoder------------------------------------------
    max_val_llama, max_val_llama3, max_val_mistral = 0, 0, 0
    for epoch in range(args.epochs):
        classifier.train()
        _, output_llama = classifier(input_data_llama)
        _, output_llama3 = classifier(input_data_llama3)
        _, output_mistral = classifier(input_data_mistral)

        loss = F.cross_entropy(output_llama[:int(0.2*len(labels_llama))], labels_llama[:int(0.2*len(labels_llama))]) +\
                  F.cross_entropy(output_llama3[:int(0.2*len(labels_llama3))], labels_llama3[:int(0.2*len(labels_llama3))]) +\
                  F.cross_entropy(output_mistral[:int(0.2*len(labels_mistral))], labels_mistral[:int(0.2*len(labels_mistral))])

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            output_llama = classifier(input_data_llama)[1].detach()
            prob_llama = F.softmax(output_llama, dim=1)
            confs_llama, preds_llama = prob_llama.max(1)

            output_llama3 = classifier(input_data_llama3)[1].detach()
            prob_llama3 = F.softmax(output_llama3, dim=1)
            confs_llama3, preds_llama3 = prob_llama3.max(1)

            output_mistral = classifier(input_data_mistral)[1].detach()
            prob_mistral = F.softmax(output_mistral, dim=1)
            confs_mistral, preds_mistral = prob_mistral.max(1)

        p_llama = prob_llama[:, 1].detach().cpu().numpy()
        p_llama3 = prob_llama3[:, 1].detach().cpu().numpy()
        p_mistral = prob_mistral[:, 1].detach().cpu().numpy()

        score_val_llama = metrics.roc_auc_score(target_llama[int(0.2*len(labels_llama)):int(0.3*len(labels_llama))], p_llama[int(0.2*len(labels_llama)):int(0.3*len(labels_llama))])
        score_test_llama = metrics.roc_auc_score(target_llama[int(0.3*len(labels_llama)):], p_llama[int(0.3*len(labels_llama)):])

        score_val_llama3 = metrics.roc_auc_score(target_llama3[int(0.2*len(labels_llama3)):int(0.3*len(labels_llama3))], p_llama3[int(0.2*len(labels_llama3)):int(0.3*len(labels_llama3))])
        score_test_llama3 = metrics.roc_auc_score(target_llama3[int(0.3*len(labels_llama3)):], p_llama3[int(0.3*len(labels_llama3)):])

        score_val_mistral = metrics.roc_auc_score(target_mistral[int(0.2*len(labels_mistral)):int(0.3*len(labels_mistral))], p_mistral[int(0.2*len(labels_mistral)):int(0.3*len(labels_mistral))])
        score_test_mistral = metrics.roc_auc_score(target_mistral[int(0.3*len(labels_mistral)):], p_mistral[int(0.3*len(labels_mistral)):])

        if score_val_llama > max_val_llama:
            max_val_llama = score_val_llama
            max_test_llama = score_test_llama
            best_p_llama = p_llama

        if score_val_llama3 > max_val_llama3:
            max_val_llama3 = score_val_llama3
            max_test_llama3 = score_test_llama3
            best_p_llama3 = p_llama3

        if score_val_mistral > max_val_mistral:
            max_val_mistral = score_val_mistral
            max_test_mistral = score_test_mistral
            best_p_mistral = p_mistral

    print("Prediction on learned features...")
    print('validation AUC score:{0:f} on LLaMA'.format(max_val_llama), 'testing AUC score:{0:f} on LLaMA'.format(max_test_llama))
    print('validation AUC score:{0:f} on LLaMA3'.format(max_val_llama3), 'testing AUC score:{0:f} on LLaMA3'.format(max_test_llama3))
    print('validation AUC score:{0:f} on Mistral'.format(max_val_mistral), 'testing AUC score:{0:f} on Mistral'.format(max_test_mistral))
    print("\n")

    np.save("best_p_domain_mixture/best_p_mistral_"+args.dataset+"_multi_"+str(args.learning_rate)+".npy", best_p_mistral)
    np.save("best_p_domain_mixture/best_p_llama_"+args.dataset+"_multi_"+str(args.learning_rate)+".npy", best_p_llama)
    np.save("best_p_domain_mixture/best_p_llama3_"+args.dataset+"_multi_"+str(args.learning_rate)+".npy", best_p_llama3)

    fp=open("./log/res_multi_objective.log", "a")
    fp.write("llm: {}, dataset: {}, lr: {}, val llama:{}, test llama:{}, val llama3:{}, test llama3:{}, val mistral:{}, test mistral:{}\n".format(args.llm, args.dataset, args.learning_rate, max_val_llama, max_test_llama, max_val_llama3, max_test_llama3, max_val_mistral, max_test_mistral))
    fp.close()

if __name__ == '__main__':
    main()

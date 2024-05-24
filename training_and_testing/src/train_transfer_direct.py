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
    parser.add_argument('--dataset2', type=str, default="PopQA",
                        help='the name of dataset2 (default: PopQA)')
    parser.add_argument('--llm2', type=str, default="llama",
                        help='the name of llm2 (default: llama)')
    parser.add_argument('--device', type=str, default="0",
                        help='id of the used GPU device (default: 0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')

    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')

    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='dimension of the hidden representations')
    parser.add_argument('--epochs', type=int, default=100,
                        help='the number of epochs')
    
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
    

    input_data_1 = np.load("./data/input_feats_"+args.dataset+"_"+args.llm+".npy")
    target_1 = np.load("./data/labels_"+args.dataset+"_"+args.llm+".npy")
    input_data_2 = np.load("./data/input_feats_"+args.dataset2+"_"+args.llm2+".npy")
    target_2 = np.load("./data/labels_"+args.dataset2+"_"+args.llm2+".npy")

    args.in_dim = input_data_1.shape[1]
    input_data_1 = torch.Tensor(input_data_1).to(device)
    input_data_2 = torch.Tensor(input_data_2).to(device)
    labels_1 = torch.LongTensor(target_1).to(device)
    labels_2 = torch.LongTensor(target_2).to(device)
    
    #----------------------------------------prepare the model/loss/optimizer-----------------------------------
    classifier = MLP(args.in_dim, args.hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), 
                                  lr = args.learning_rate, 
                                  weight_decay=args.weight_decay)
        
    #----------------------------------------------optimize the encoder------------------------------------------
    max_val_1 = 0
    for _ in range(args.epochs):
        classifier.train()
        _, output_1 = classifier(input_data_1)
    
        loss = F.cross_entropy(output_1[:int(0.2*len(labels_1))], labels_1[:int(0.2*len(labels_1))])

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

       
        classifier.eval()
        with torch.no_grad():
            output_1 = classifier(input_data_1)[1].detach()
            prob_1 = F.softmax(output_1, dim=1)
            confs_1, preds_1 = prob_1.max(1)

            output_2 = classifier(input_data_2)[1].detach()
            prob_2 = F.softmax(output_2, dim=1)
            confs_2, preds_2 = prob_2.max(1)

        p_1 = prob_1[:, 1].detach().cpu().numpy()
        p_2 = prob_2[:, 1].detach().cpu().numpy()
        
        score_val_1 = metrics.roc_auc_score(target_1[int(0.2*len(target_1)):int(0.3*len(target_1))], p_1[int(0.2*len(target_1)):int(0.3*len(target_1))])
        score_val_2 = metrics.roc_auc_score(target_2[int(0.2*len(target_2)):int(0.3*len(target_2))], p_2[int(0.2*len(target_2)):int(0.3*len(target_2))])
        score_test_1 = metrics.roc_auc_score(target_1[int(0.3*len(target_1)):], p_1[int(0.3*len(target_1)):])
        score_test_2 = metrics.roc_auc_score(target_2[int(0.3*len(target_2)):], p_2[int(0.3*len(target_2)):])

        if score_val_1 > max_val_1:
            max_val_1 = score_val_1
            max_val_2 = score_val_2
            max_test_1 = score_test_1
            max_test_2 = score_test_2

    print("Prediction on learned features...")
    print("testing AUC score on domain_1:{0:f}".format(max_test_1), "transfer testing AUC score on domain_2:{0:f}".format(max_test_2))
    print("\n")

    # write down the evaluation results
    fp=open("./log/res_direct_transfer.log", "a")
    fp.write("llm: {}, dataset: {}, llm2: {}, dataset2: {}, transfer testing AUC score:{}, transfer testing AUC score (100 epochs):{} \n".format(args.llm, args.dataset, args.llm2, args.dataset2, max_test_2, score_test_2))
    fp.close()

if __name__ == '__main__':
    main()

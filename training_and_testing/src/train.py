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
    parser.add_argument('--train_ratio', type=float, default=0.2,
                        help='the ratio of training data')

    parser.add_argument('--use_entity', action='store_true', 
                        help='whether to use entity-level hidden states')
    parser.add_argument('--use_last_token', action='store_true', 
                        help='whether to the hidden states of the last tokens')

    
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

    input_data, target = init_feats(args.llm, args.dataset, args.layer_name, use_entity=args.use_entity, use_last_token=args.use_last_token)
    
    if args.use_last_token and args.layer_name == "middle_layer":
        np.save("./data/input_feats_"+args.dataset+"_"+args.llm+".npy", input_data)
        np.save("./data/labels_"+args.dataset+"_"+args.llm+".npy", target)

    args.in_dim = input_data.shape[1]
    input_data = torch.Tensor(input_data).to(device)
    labels = torch.LongTensor(target).to(device)
    #----------------------------------------prepare the model/loss/optimizer-----------------------------------
    classifier = MLP(args.in_dim, args.hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), 
                                  lr = args.learning_rate, 
                                  weight_decay=args.weight_decay)
        
    #----------------------------------------------optimize the encoder------------------------------------------
    max_val = 0
    best_p = None
    for epoch in range(args.epochs):
        classifier.train()
        _, output = classifier(input_data)
        loss = F.cross_entropy(output[:int(args.train_ratio*len(labels))], labels[:int(args.train_ratio*len(labels))].to(device))
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

       
        classifier.eval()
        with torch.no_grad():
            output = classifier(input_data)[1].detach()
            prob = F.softmax(output, dim=1)
            confs, preds = prob.max(1)

        p = prob[:, 1].detach().cpu().numpy()
        
        score_val = metrics.roc_auc_score(target[int(args.train_ratio*len(labels)):int((args.train_ratio+0.1)*len(labels))], p[int(args.train_ratio*len(labels)):int((args.train_ratio+0.1)*len(labels))])
        score_test = metrics.roc_auc_score(target[int((args.train_ratio+0.1)*len(labels)):], p[int((args.train_ratio+0.1)*len(labels)):])

        if score_val > max_val:
            max_val = score_val
            max_test = score_test
            best_p = p
            
    # np.save("best_p_domain_mixture/best_p_"+args.llm+"_"+args.dataset+"_single_"+str(args.learning_rate)+".npy", best_p)

    print("Prediction on learned features...")
    print('validation AUC score:{0:f}'.format(max_val), 'testing AUC score:{0:f}'.format(max_test))
    print("\n")

    # write down the evaluation results
    fp=open("./log/res_single_llm_"+args.llm+"_new.log", "a")
    fp.write("llm: {}, dataset: {}, layer_name: {}, use_entity: {}, use_last_token: {}, lr: {}, validation AUC score:{}, testing AUC score:{} \n".format(args.llm, args.dataset, args.layer_name, args.use_entity, args.use_last_token, args.learning_rate, max_val, max_test))
    fp.close()

if __name__ == '__main__':
    main()

# usr/bin/env python
# coding: utf-8

import os
import argparse
from pprint import pprint

import torch
import numpy as np
import pandas as pd
import json
import time
from sklearn.manifold import TSNE

from metrics import NLEEP, LogME_Score, SFDA_Score,get_gbc_score,NCTI_Score, LDA_Score, SA

models_hub = ['byol', 'deepcluster-v2', 'infomin', 'insdis', 'moco-v1',
             'moco-v2', 'pcl-v1', 'pcl-v2', 'sela-v2', 'simclr-v1',
             'simclr-v2', 'swav']


def save_score(score_dict, fpath):
    with open(fpath, "w") as f:
        # write dict 
        json.dump(score_dict, f)

def save_time(time_dict, fpath):
    with open(fpath, "w") as f:
        # write dict
        json.dump(time_dict, f)

def exist_score(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False

def exist_time(model_name, fpath):
    with open(fpath, "r") as f:
        result = json.load(f)
        if model_name in result.keys():
            return True
        else:
            return False

# Main code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via logistic regression.')
    parser.add_argument('-m', '--model', type=str, default='deepcluster-v2',
                        help='name of the pretrained model to load and evaluate (deepcluster-v2 | supervised)')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        help='name of the dataset to evaluate on')
    parser.add_argument('-me', '--metric', type=str, default='logme',
                        help='name of the method for measuring transferability')
    parser.add_argument('--nleep-ratio', type=float, default=5,
                        help='the ratio of the Gaussian components and target data classess')
    parser.add_argument('--parc-ratio', type=float, default=2,
                        help='PCA reduction dimension')
    parser.add_argument('--output-dir', type=str, default='results_metrics_group2',
                        help='dir of output score')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--sigma', type=float, default=0.6)
    parser.add_argument('--type', type=str, default=None)

    args = parser.parse_args()
    pprint(args)

    score_dict = {}
    metric = args.metric

    if args.type == None:
        fpath_parent = os.path.join(args.output_dir,args.metric)
    else:
        fpath_parent = os.path.join(args.output_dir, args.metric, args.type)

    if not os.path.exists(fpath_parent):
        os.makedirs(fpath_parent)
    fpath = os.path.join(fpath_parent, f'{args.dataset}_metrics.json')

    if not os.path.exists(fpath):
        save_score(score_dict, fpath)
    else:
        with open(fpath, "r") as f:
            score_dict = json.load(f)
            

    start_time = time.time()
    for model in models_hub:
        args.model = model
        if exist_score(model, fpath):
            print(f'{model} has been calculated')
            continue

        model_npy_feature = os.path.join('.', 'results_f', 'group2', f'{args.model}_{args.dataset}_feature.npy')
        model_npy_label = os.path.join('.', 'results_f', 'group2', f'{args.model}_{args.dataset}_label.npy')


        X_features, y_labels = np.load(model_npy_feature), np.load(model_npy_label)

        print(f'x_trainval shape:{X_features.shape} and y_trainval shape:{y_labels.shape}')
        print(f'Calc Transferabilities of {args.model} on {args.dataset}')


        if args.metric == 'logme':
            score_dict[args.model] = LogME_Score(X_features, y_labels)
        elif args.metric == 'nleep':
            ratio = 1 if args.dataset in ('food', 'pets') else args.nleep_ratio
            score_dict[args.model] = NLEEP(X_features, y_labels, component_ratio=ratio)
        elif args.metric == 'sfda':
            score_dict[args.model] = SFDA_Score(X_features, y_labels) 
        elif args.metric == 'gbc':
            score_dict[args.model] = get_gbc_score(torch.from_numpy(X_features), torch.from_numpy(y_labels),'spherical')
        elif args.metric == 'ncti':
            score_dict[args.model] = NCTI_Score(X_features, y_labels)
        elif args.metric == 'lda':
            score_dict[args.model] = LDA_Score(X_features, y_labels)
        elif args.metric == 'SA':
            score_dict[args.model] = SA(X_features, y_labels,alpha=args.alpha, type=args.type,
                                                   sigma=args.sigma, model= model, dataset = args.dataset)
        else:
            raise NotImplementedError

        print(f'{args.metric} of {args.model}: {score_dict[args.model]}\n')
        

    if args.metric in ['ncti']:
        all_score = []
        cls_score = []
        cls_compact = []

        for model in models_hub:
            all_score.append(score_dict[model][0])
            cls_score.append(score_dict[model][1])
            cls_compact.append(score_dict[model][2])
            
        all_score = np.array(all_score)
        cls_score = np.array(cls_score)
        cls_compact = np.array(cls_compact)

        all_score_min = all_score.min()
        all_score_div = all_score.max() - all_score.min()

        cls_score_min = cls_score.min()
        cls_score_div = cls_score.max() - cls_score.min()

        cls_compact_min = cls_compact.min()
        cls_compact_div = cls_compact.max() - cls_compact_min

        for model in models_hub:
            print(model)
            mascore = (score_dict[model][0] - all_score_min)/all_score_div
            mcscore = (score_dict[model][1] - cls_score_min)/cls_score_div
            cpscore = (score_dict[model][2] - cls_compact_min)/cls_compact_div
            print(mascore)
            print(mcscore)
            print(cpscore)
            score_dict[model] = mcscore  + mascore - cpscore
        print("seli")
        
        # print(((all_score - all_score_min)/all_score_div).var())
        print(all_score)
        print("ncc")
        
        # print(((cls_score- cls_score_min)/cls_score_div).var())
        print(cls_score)
        print("vc")
        # print(((cls_compact- cls_compact_min)/cls_compact_div).var())
        print(cls_compact)

    end_time = time.time()
    score_dict['duration'] = end_time- start_time
    print(end_time-start_time)
    save_score(score_dict, fpath)

    results = sorted(score_dict.items(), key=lambda i: i[1], reverse=True)
    print(f'Models ranking on {args.dataset} based on {args.metric}: ')
    pprint(results)
    results = {a[0]: a[1] for a in results}
    save_score(results, fpath)

#!/usr/bin/env python
# coding: utf-8

from rank_correlation import (load_score, recall_k, rel_k, pearson_coef, 
                            wpearson_coef, w_kendall_metric, kendall_metric)
import os

models_hub = ["resnet34", "resnet50", "resnet101", "resnet152", "mobilenet_v2", "mnasnet1_0",
          "densenet121", "densenet169", "densenet201",
          "googlenet", "inception_v3"]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via finetuning.')
    parser.add_argument('-d', '--dataset', type=str, default='deepcluster-v2', help='name of the pretrained model to load and evaluate')
    parser.add_argument('-me', '--method', type=str, default='logme', help='name of the pretrained model to load and evaluate')
    parser.add_argument('-dir', '--directory', type=str, default=".", help='directory of metric score')
    parser.add_argument('-o', type=float, default=0,
                        help='name of used transferability metric')
    parser.add_argument('--type', type=str, default=None)
    parser.add_argument('--nsigma',type = float, default =0.6)
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()
    
    finetune_acc = {
        'aircraft': {'resnet34': 84.06, 'resnet50': 84.64, 'resnet101': 85.53, 'resnet152': 86.29, 'densenet121': 84.66, 
                    'densenet169': 84.19, 'densenet201': 85.38, 'mnasnet1_0': 66.48, 'mobilenet_v2': 79.68, 
                    'googlenet': 80.32, 'inception_v3': 80.15}, 
        'caltech101': {'resnet34': 91.15, 'resnet50': 91.98, 'resnet101': 92.38, 'resnet152': 93.1, 'densenet121': 91.5, 
                    'densenet169': 92.51, 'densenet201': 93.14, 'mnasnet1_0': 89.34, 'mobilenet_v2': 88.64, 
                    'googlenet': 90.85, 'inception_v3': 92.75}, 
        'cars': {'resnet34': 88.63, 'resnet50': 89.09, 'resnet101': 89.47, 'resnet152': 89.88, 'densenet121': 89.34, 
                    'densenet169': 89.02, 'densenet201': 89.44, 'mnasnet1_0': 72.58, 'mobilenet_v2': 86.44, 
                    'googlenet': 87.76, 'inception_v3': 87.74}, 
        'cifar10': {'resnet34': 96.12, 'resnet50': 96.28, 'resnet101': 97.39, 'resnet152': 97.53, 'densenet121': 96.45, 
                    'densenet169': 96.77, 'densenet201': 97.02, 'mnasnet1_0': 92.59, 'mobilenet_v2': 94.74, 
                    'googlenet': 95.54, 
                    'inception_v3': 96.18}, 
        'cifar100': {'resnet34': 81.94, 'resnet50': 82.8, 'resnet101': 84.88, 'resnet152': 85.66, 'densenet121': 82.75, 
                    'densenet169': 84.26, 'densenet201': 84.88, 'mnasnet1_0': 72.04, 'mobilenet_v2': 78.11, 
                    'googlenet': 79.84, 
                    'inception_v3': 81.49}, 
        'dtd': {'resnet34': 72.96, 'resnet50': 74.72, 'resnet101': 74.8, 'resnet152': 76.44, 'densenet121': 74.18, 
                    'densenet169': 74.72, 'densenet201': 76.04, 'mnasnet1_0': 70.12, 'mobilenet_v2': 71.72, 
                    'googlenet': 72.53, 
                    'inception_v3': 72.85}, 
        'flowers': {'resnet34': 95.2, 'resnet50': 96.26, 'resnet101': 96.53, 'resnet152': 96.86, 'densenet121': 97.02, 
                    'densenet169': 97.32, 'densenet201': 97.1, 'mnasnet1_0': 95.39, 'mobilenet_v2': 96.2, 
                    'googlenet': 95.76, 
                    'inception_v3': 95.73},
        'food': {'resnet34': 81.99, 'resnet50': 84.45, 'resnet101': 85.58, 'resnet152': 86.28, 'densenet121': 84.99, 
                    'densenet169': 85.84, 'densenet201': 86.71, 'mnasnet1_0': 71.35, 'mobilenet_v2': 81.12, 
                    'googlenet': 79.3, 
                    'inception_v3': 81.76}, 
        'pets': {'resnet34': 93.5, 'resnet50': 93.88, 'resnet101': 93.92, 'resnet152': 94.42, 'densenet121': 93.07, 
                    'densenet169': 93.62, 'densenet201': 94.03, 'mnasnet1_0': 91.08, 'mobilenet_v2': 91.28, 
                    'googlenet': 91.38, 
                    'inception_v3': 92.14},
        'sun397': {'resnet34': 61.02, 'resnet50': 63.54, 'resnet101': 63.76, 'resnet152': 64.82, 'densenet121': 63.26, 
                    'densenet169': 64.1, 'densenet201': 64.57, 'mnasnet1_0': 56.56, 'mobilenet_v2': 60.29, 
                    'googlenet': 59.89, 
                    'inception_v3': 59.98}, 
        'voc2007': {'resnet34': 84.6, 'resnet50': 85.8, 'resnet101': 85.68, 'resnet152': 86.32, 'densenet121': 85.28, 
                    'densenet169': 85.77, 'densenet201': 85.67, 'mnasnet1_0': 81.06, 'mobilenet_v2': 82.8, 
                    'googlenet': 82.58, 
                    'inception_v3': 83.84}
        }


    dset = args.dataset
    metric = args.method
    dir = args.directory
    score_path = os.path.join(dir,"results_metrics_group1", metric,args.type, f"{dset}_metrics.json")
    score, _ = load_score(score_path)
    tw_vanilla_ft = w_kendall_metric(score, finetune_acc, dset)
    print("Kendall correlation for vanilla finetuning: {:12s} {} score : {:2.3f}".format(dset,metric, tw_vanilla_ft))

    lbft_acc ={ 'aircraft': {'inception_v3': 47.98, 'mobilenet_v2': 53.33, 'mnasnet1_0': 52.05, 'densenet121': 67.81, 'densenet169': 74.29, 'densenet201': 71.51, 'resnet34': 70.94, 'resnet50': 76.43, 'resnet101': 75.57, 'resnet152': 74.68, 'googlenet': 64.55},
    'caltech101': {'inception_v3': 90.25, 'mobilenet_v2': 86.78, 'mnasnet1_0': 88.92, 'densenet121': 90.24, 'densenet169': 92.51, 'densenet201': 92.02, 'resnet34': 90.42, 'resnet50': 91.4, 'resnet101': 91.92, 'resnet152': 92.45, 'googlenet': 90.31},
    'cars': {'inception_v3': 56.6, 'mobilenet_v2': 69.83, 'mnasnet1_0': 65.08, 'densenet121': 81.72, 'densenet169': 83.94, 'densenet201': 83.51, 'resnet34': 83.07, 'resnet50': 84.93, 'resnet101': 85.19, 'resnet152': 85.75, 'googlenet': 78.06},
    'cifar10': {'inception_v3': 83.76, 'mobilenet_v2': 87.06, 'mnasnet1_0': 74.4, 'densenet121': 93.7, 'densenet169': 96.06, 'densenet201': 95.74, 'resnet34': 93.94, 'resnet50': 86.49, 'resnet101': 96.34, 'resnet152': 96.18, 'googlenet': 92.67},
    'cifar100': {'inception_v3': 63.46, 'mobilenet_v2': 66.59, 'mnasnet1_0': 40.68, 'densenet121': 78.23, 'densenet169': 84.2, 'densenet201': 83.85, 'resnet34': 80.74, 'resnet50': 84.46, 'resnet101': 85.04, 'resnet152': 84.73, 'googlenet': 76.1},
    'dtd': {'inception_v3': 68.99, 'mobilenet_v2': 73.19, 'mnasnet1_0': 68.03, 'densenet121': 72.93, 'densenet169': 74.36, 'densenet201': 74.41, 'resnet34': 71.54, 'resnet50': 74.57, 'resnet101': 74.79, 'resnet152': 75.16, 'googlenet': 72.82},
    'flowers': {'inception_v3': 90.76, 'mobilenet_v2': 94.95, 'mnasnet1_0': 93.95, 'densenet121': 97.36, 'densenet169': 96.47, 'densenet201': 97.36, 'resnet34': 96.42, 'resnet50': 97.25, 'resnet101': 96.48, 'resnet152': 95.41, 'googlenet': 95.08},
    'food': {'inception_v3': 63.77, 'mobilenet_v2': 72.24, 'mnasnet1_0': 67.05, 'densenet121': 79.29, 'densenet169': 82.06, 'densenet201': 81.94, 'resnet34': 78.04, 'resnet50': 82.8, 'resnet101': 83.02, 'resnet152': 82.86, 'googlenet': 72.69},
    'pets': {'inception_v3': 87.78, 'mobilenet_v2': 90.41, 'mnasnet1_0': 90.54, 'densenet121': 91.26, 'densenet169': 93.65, 'densenet201': 91.89, 'resnet34': 92.84, 'resnet50': 93.87, 'resnet101': 93.44, 'resnet152': 93.93, 'googlenet': 89.67},
    'sun397': {'inception_v3': 81.6, 'mobilenet_v2': 83.26, 'mnasnet1_0': 73.0, 'densenet121': 90.39, 'densenet169': 96.83, 'densenet201': 97.02, 'resnet34': 94.84, 'resnet50': 96.28, 'resnet101': 97.41, 'resnet152': 96.56, 'googlenet': 92.4},
    'voc2007': {'inception_v3': 80.88, 'mobilenet_v2': 82.03, 'mnasnet1_0': 82.37, 'densenet121': 84.42, 'densenet169': 86.03, 'densenet201': 85.36, 'resnet34': 84.39, 'resnet50': 85.67, 'resnet101': 85.76, 'resnet152': 86.15, 'googlenet': 80.75}}

    tw_lbft = w_kendall_metric(score, lbft_acc, dset)
    print("Kendall correlation for modified fine-tuning:{:12s} {}:{:2.3f}".format(dset,metric, tw_lbft))
    
    lft_acc = {'aircraft': {'inception_v3': 28.21, 'mobilenet_v2': 42.24, 'mnasnet1_0': 41.72, 'densenet121': 43.61, 'densenet169': 47.15, 'densenet201': 46.39, 'resnet34': 38.19, 'resnet50': 40.63, 'resnet101': 41.21, 'resnet152': 42.98, 'googlenet': 36.22},
    'caltech101': {'inception_v3': 88.48, 'mobilenet_v2': 87.35, 'mnasnet1_0': 87.85, 'densenet121': 90.03, 'densenet169': 90.76, 'densenet201': 91.31, 'resnet34': 89.8, 'resnet50': 89.75, 'resnet101': 89.81, 'resnet152': 91.42, 'googlenet': 88.31},
    'cars': {'inception_v3': 27.6, 'mobilenet_v2': 49.77, 'mnasnet1_0': 46.19, 'densenet121': 51.78, 'densenet169': 56.2, 'densenet201': 57.32, 'resnet34': 32.04, 'resnet50': 50.91, 'resnet101': 50.6, 'resnet152': 52.07, 'googlenet': 43.83},
    'cifar10': {'inception_v3': 69.87, 'mobilenet_v2': 76.97, 'mnasnet1_0': 69.55, 'densenet121': 81.39, 'densenet169': 83.08, 'densenet201': 84.52, 'resnet34': 78.61, 'resnet50': 83.57, 'resnet101': 85.24, 'resnet152': 85.33, 'googlenet': 78.45},
    'cifar100': {'inception_v3': 46.39, 'mobilenet_v2': 57.46, 'mnasnet1_0': 37.49, 'densenet121': 62.11, 'densenet169': 64.53, 'densenet201': 67.51, 'resnet34': 59.43, 'resnet50': 65.41, 'resnet101': 67.64, 'resnet152': 67.81, 'googlenet': 59.73},
    'dtd': {'inception_v3': 61.28, 'mobilenet_v2': 67.77, 'mnasnet1_0': 65.69, 'densenet121': 68.09, 'densenet169': 69.95, 'densenet201': 70.64, 'resnet34': 66.7, 'resnet50': 70.74, 'resnet101': 69.57, 'resnet152': 70.74, 'googlenet': 66.12},
    'flowers': {'inception_v3': 83.01, 'mobilenet_v2': 92.27, 'mnasnet1_0': 92.37, 'densenet121': 93.23, 'densenet169': 94.15, 'densenet201': 93.01, 'resnet34': 90.71, 'resnet50': 93.05, 'resnet101': 92.3, 'resnet152': 93.06, 'googlenet': 89.53},
    'food': {'inception_v3': 46.31, 'mobilenet_v2': 62.6, 'mnasnet1_0': 62.65, 'densenet121': 65.37, 'densenet169': 67.81, 'densenet201': 68.11, 'resnet34': 60.56, 'resnet50': 65.79, 'resnet101': 66.5, 'resnet152': 67.55, 'googlenet': 55.34},
    'pets': {'inception_v3': 85.85, 'mobilenet_v2': 89.73, 'mnasnet1_0': 89.56, 'densenet121': 91.46, 'densenet169': 92.6, 'densenet201': 92.57, 'resnet34': 91.27, 'resnet50': 91.76, 'resnet101': 92.34, 'resnet152': 92.67, 'googlenet': 89.41},
    'sun397': {'inception_v3': 63.72, 'mobilenet_v2': 73.25, 'mnasnet1_0': 79.63, 'densenet121': 76.37, 'densenet169': 80.78, 'densenet201': 80.38, 'resnet34': 71.96, 'resnet50': 83.29, 'resnet101': 75.61, 'resnet152': 75.72, 'googlenet': 76.82},
    'voc2007': {'inception_v3': 77.01, 'mobilenet_v2': 80.88, 'mnasnet1_0': 81.18, 'densenet121': 82.73, 'densenet169': 84.07, 'densenet201': 83.34, 'resnet34': 82.46, 'resnet50': 83.28, 'resnet101': 83.85, 'resnet152': 84.13, 'googlenet': 80.32},}

    tw_lft = w_kendall_metric(score, lft_acc, dset)
    print("Kendall correlation for linear probing:{:12s} {}:{:2.3f}".format(dset,metric, tw_lft))

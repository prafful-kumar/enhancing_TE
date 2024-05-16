#!/usr/bin/env python
# coding: utf-8

from rank_correlation import (load_score, recall_k, rel_k, pearson_coef, 
                            wpearson_coef, w_kendall_metric, kendall_metric)
import os

models_hub = ['byol', 'deepcluster-v2', 'infomin', 'insdis', 'moco-v1',
              'moco-v2', 'pcl-v1', 'pcl-v2', 'sela-v2', 'simclr-v1',
              'simclr-v2', 'swav']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate pretrained self-supervised model via finetuning.')
    parser.add_argument('-d', '--dataset', type=str, default='deepcluster-v2', help='name of the pretrained model to load and evaluate')
    parser.add_argument('-me', '--method', type=str, default='logme', help='name of the pretrained model to load and evaluate')
    parser.add_argument('-dir', '--directory', type=str, default="./results_metrics_group2", help='directory of metric score')
    parser.add_argument('-o', type=float, default=0,
                        help='name of used transferability metric')
    args = parser.parse_args()
    
    # finetune acc
    finetune_acc = {
        'aircraft': {'byol': 82.1, 'deepcluster-v2': 82.43, 'infomin': 83.78, 'insdis': 79.7, 'moco-v1': 81.85,
                     'moco-v2': 83.7, 'pcl-v1': 82.16, 'pcl-v2': 83.0, 'sela-v2': 85.42, 'simclr-v1': 80.54,
                     'simclr-v2': 81.5, 'swav': 83.04},
        'caltech101': {'byol': 91.9, 'deepcluster-v2': 91.16, 'infomin': 80.86, 'insdis': 77.21, 'moco-v1': 79.68, 
                    'moco-v2': 82.76, 'pcl-v1': 88.6, 'pcl-v2': 87.52, 'sela-v2': 90.53, 'simclr-v1': 90.94, 
                    'simclr-v2': 88.58, 'swav': 89.49}, 
        'cars': {'byol': 89.83, 'deepcluster-v2': 90.16, 'infomin': 86.9, 'insdis': 80.21, 'moco-v1': 82.19, 
                    'moco-v2': 85.55, 'pcl-v1': 87.15, 'pcl-v2': 85.56, 'sela-v2': 89.85, 'simclr-v1': 89.98, 
                    'simclr-v2': 88.82, 'swav': 89.81}, 
        'cifar10': {'byol': 96.98, 'deepcluster-v2': 97.17, 'infomin': 96.72, 'insdis': 93.08, 'moco-v1': 94.15, 
                    'moco-v2': 96.48, 'pcl-v1': 96.42, 'pcl-v2': 96.55, 'sela-v2': 96.85, 'simclr-v1': 97.09, 
                    'simclr-v2': 96.22, 'swav': 96.81},
        'cifar100': {'byol': 83.86, 'deepcluster-v2': 84.84, 'infomin': 70.89, 'insdis': 69.08, 'moco-v1': 71.23, 
                    'moco-v2': 71.27, 'pcl-v1': 79.44, 'pcl-v2': 79.84, 'sela-v2': 84.36, 'simclr-v1': 84.49, 
                    'simclr-v2': 78.91, 'swav': 83.78}, 
        'dtd': {'byol': 76.37, 'deepcluster-v2': 77.31, 'infomin': 73.47, 'insdis': 66.4, 'moco-v1': 67.36, 
                    'moco-v2': 72.56, 'pcl-v1': 73.28, 'pcl-v2': 69.3, 'sela-v2': 76.03, 'simclr-v1': 73.97, 
                    'simclr-v2': 74.71, 'swav': 76.68}, 
        'flowers': {'byol': 96.8, 'deepcluster-v2': 97.05, 'infomin': 95.81, 'insdis': 93.63, 'moco-v1': 94.32, 
                    'moco-v2': 95.12, 'pcl-v1': 95.62, 'pcl-v2': 95.87, 'sela-v2': 96.22, 'simclr-v1': 95.33, 
                    'simclr-v2': 95.39, 'swav': 97.11}, 
        'food': {'byol': 85.44, 'deepcluster-v2': 87.24, 'infomin': 78.82, 'insdis': 76.47, 'moco-v1': 77.21, 
                    'moco-v2': 77.15, 'pcl-v1': 77.7, 'pcl-v2': 80.29, 'sela-v2': 86.37, 'simclr-v1': 82.2, 
                    'simclr-v2': 82.23, 'swav': 87.22}, 
        'pets': {'byol': 91.48, 'deepcluster-v2': 90.89, 'infomin': 90.92, 'insdis': 84.58, 'moco-v1': 85.26, 
                    'moco-v2': 89.06, 'pcl-v1': 88.93, 'pcl-v2': 88.72, 'sela-v2': 89.61, 'simclr-v1': 88.53, 
                    'simclr-v2': 89.18, 'swav': 90.59}, 
        'sun397': {'byol': 63.69, 'deepcluster-v2': 66.54, 'infomin': 57.67, 'insdis': 51.62, 'moco-v1': 53.83, 
                    'moco-v2': 56.28, 'pcl-v1': 58.36, 'pcl-v2': 58.82, 'sela-v2': 65.74, 'simclr-v1': 63.46, 
                    'simclr-v2': 60.93, 'swav': 66.1}, 
        'voc2007': {'byol': 85.13, 'deepcluster-v2': 85.38, 'infomin': 81.41, 'insdis': 76.33, 'moco-v1': 77.94, 
                    'moco-v2': 78.32, 'pcl-v1': 81.91, 'pcl-v2': 81.85, 'sela-v2': 85.52, 'simclr-v1': 83.29, 
                    'simclr-v2': 83.08, 'swav': 85.06}
        }
        

    dset = args.dataset
    metric = args.method
    dir = args.directory
    score_path = '{}/{}/{}_metrics.json'.format(dir,metric, dset)
    score, _ = load_score(score_path)

    tw_vanilla_ft = w_kendall_metric(score, finetune_acc, dset)
    print("Kendall correlation for vanilla finetuning:{:12s} {}:{:2.3f}".format(dset,metric, tw_vanilla_ft))


    lbft_acc = {'aircraft': {'byol': 73.09, 'deepcluster-v2': 71.16, 'infomin': 77.46, 'insdis': 70.65, 'moco-v1': 73.16, 'moco-v2': 75.7, 'pcl-v1': 76.6, 'pcl-v2': 76.65, 'sela-v2': 69.85, 'simclr-v1': 67.45, 'simclr-v2': 74.04, 'swav': 71.65},
    'caltech101': {'byol': 91.05, 'deepcluster-v2': 89.62, 'infomin': 84.77, 'insdis': 76.14, 'moco-v1': 78.4, 'moco-v2': 85.18, 'pcl-v1': 85.63, 'pcl-v2': 85.07, 'sela-v2': 87.56, 'simclr-v1': 90.48, 'simclr-v2': 85.19, 'swav': 88.49},
    'cars': {'byol': 85.15, 'deepcluster-v2': 83.26, 'infomin': 85.57, 'insdis': 79.48, 'moco-v1': 81.4, 'moco-v2': 84.37, 'pcl-v1': 83.82, 'pcl-v2': 84.94, 'sela-v2': 81.66, 'simclr-v1': 77.08, 'simclr-v2': 84.83, 'swav': 82.84},
    'cifar10': {'byol': 98.5, 'deepcluster-v2': 95.95, 'infomin': 96.31, 'insdis': 94.16, 'moco-v1': 94.35, 'moco-v2': 96.23, 'pcl-v1': 96.94, 'pcl-v2': 97.75, 'sela-v2': 95.56, 'simclr-v1': 96.96, 'simclr-v2': 97.38, 'swav': 95.72},
    'cifar100': {'byol': 92.52, 'deepcluster-v2': 84.62, 'infomin': 82.15, 'insdis': 77.2, 'moco-v1': 77.97, 'moco-v2': 82.12, 'pcl-v1': 85.98, 'pcl-v2': 87.97, 'sela-v2': 83.75, 'simclr-v1': 87.65, 'simclr-v2': 89.9, 'swav': 83.69},
    'dtd': {'byol': 74.68, 'deepcluster-v2': 75.21, 'infomin': 74.63, 'insdis': 70.74, 'moco-v1': 71.49, 'moco-v2': 73.46, 'pcl-v1': 73.03, 'pcl-v2': 72.45, 'sela-v2': 74.36, 'simclr-v1': 71.65, 'simclr-v2': 72.45, 'swav': 75.85},
    'flowers': {'byol': 96.16, 'deepcluster-v2': 95.87, 'infomin': 96.18, 'insdis': 91.91, 'moco-v1': 92.24, 'moco-v2': 95.47, 'pcl-v1': 94.78, 'pcl-v2': 95.11, 'sela-v2': 94.94, 'simclr-v1': 92.13, 'simclr-v2': 95.5, 'swav': 95.66},
    'food': {'byol': 83.11, 'deepcluster-v2': 83.29, 'infomin': 84.25, 'insdis': 79.05, 'moco-v1': 78.7, 'moco-v2': 82.57, 'pcl-v1': 80.82, 'pcl-v2': 82.62, 'sela-v2': 82.62, 'simclr-v1': 75.77, 'simclr-v2': 83.09, 'swav': 83.31},
    'pets': {'byol': 89.6, 'deepcluster-v2': 89.82, 'infomin': 88.6, 'insdis': 80.28, 'moco-v1': 83.05, 'moco-v2': 87.78, 'pcl-v1': 85.83, 'pcl-v2': 87.05, 'sela-v2': 88.54, 'simclr-v1': 85.43, 'simclr-v2': 84.92, 'swav': 87.62},
    'sun397': {'byol': 99.81, 'deepcluster-v2': 99.79, 'infomin': 98.62, 'insdis': 97.96, 'moco-v1': 98.02, 'moco-v2': 97.87, 'pcl-v1': 99.13, 'pcl-v2': 99.17, 'sela-v2': 99.53, 'simclr-v1': 99.48, 'simclr-v2': 99.85, 'swav': 99.8},
    'voc2007': {'byol': 84.06, 'deepcluster-v2': 84.59, 'infomin': 82.56, 'insdis': 76.5, 'moco-v1': 78.1, 'moco-v2': 81.2, 'pcl-v1': 80.87, 'pcl-v2': 81.42, 'sela-v2': 85.19, 'simclr-v1': 82.29, 'simclr-v2': 80.76, 'swav': 84.22},
    }
    
    tw_lbft = w_kendall_metric(score, lbft_acc, dset)
    print("Kendall correlation for Convolutional fine-tuning:{:12s} {}:{:2.3f}".format(dset,metric, tw_lbft))
    
    lft_acc = {'aircraft': {'byol': 43.48, 'deepcluster-v2': 47.44, 'infomin': 12.81, 'insdis': 10.93, 'moco-v1': 10.88, 'moco-v2': 11.51, 'pcl-v1': 7.46, 'pcl-v2': 13.99, 'sela-v2': 31.31, 'simclr-v1': 42.75, 'simclr-v2': 39.96, 'swav': 43.25},
    'caltech101': {'byol': 89.83, 'deepcluster-v2': 89.34, 'infomin': 80.61, 'insdis': 51.26, 'moco-v1': 54.23, 'moco-v2': 78.43, 'pcl-v1': 70.13, 'pcl-v2': 82.41, 'sela-v2': 84.62, 'simclr-v1': 88.72, 'simclr-v2': 86.66, 'swav': 87.85},
    'cars': {'byol': 43.45, 'deepcluster-v2': 56.19, 'infomin': 7.24, 'insdis': 3.82, 'moco-v1': 3.32, 'moco-v2': 5.52, 'pcl-v1': 3.9, 'pcl-v2': 8.2, 'sela-v2': 24.4, 'simclr-v1': 43.23, 'simclr-v2': 42.54, 'swav': 45.94},
    'cifar10': {'byol': 84.07, 'deepcluster-v2': 79.43, 'infomin': 58.89, 'insdis': 42.81, 'moco-v1': 45.01, 'moco-v2': 54.22, 'pcl-v1': 50.7, 'pcl-v2': 69.79, 'sela-v2': 73.0, 'simclr-v1': 83.77, 'simclr-v2': 80.74, 'swav': 75.93},
    'cifar100': {'byol': 57.71, 'deepcluster-v2': 55.19, 'infomin': 22.09, 'insdis': 15.65, 'moco-v1': 15.68, 'moco-v2': 24.09, 'pcl-v1': 22.68, 'pcl-v2': 32.66, 'sela-v2': 38.91, 'simclr-v1': 61.6, 'simclr-v2': 55.51, 'swav': 47.59},
    'dtd': {'byol': 71.28, 'deepcluster-v2': 72.45, 'infomin': 65.11, 'insdis': 56.33, 'moco-v1': 54.41, 'moco-v2': 64.89, 'pcl-v1': 52.23, 'pcl-v2': 65.9, 'sela-v2': 72.07, 'simclr-v1': 67.07, 'simclr-v2': 71.97, 'swav': 74.15},
    'flowers': {'byol': 92.75, 'deepcluster-v2': 93.65, 'infomin': 63.58, 'insdis': 58.0, 'moco-v1': 54.56, 'moco-v2': 59.73, 'pcl-v1': 36.81, 'pcl-v2': 69.71, 'sela-v2': 87.64, 'simclr-v1': 88.42, 'simclr-v2': 91.34, 'swav': 92.54},
    'food': {'byol': 61.17, 'deepcluster-v2': 68.62, 'infomin': 37.98, 'insdis': 27.06, 'moco-v1': 26.89, 'moco-v2': 34.86, 'pcl-v1': 21.12, 'pcl-v2': 36.15, 'sela-v2': 58.06, 'simclr-v1': 58.55, 'simclr-v2': 63.24, 'swav': 66.42},
    'pets': {'byol': 87.13, 'deepcluster-v2': 87.08, 'infomin': 80.96, 'insdis': 50.77, 'moco-v1': 53.03, 'moco-v2': 73.62, 'pcl-v1': 68.08, 'pcl-v2': 75.51, 'sela-v2': 82.27, 'simclr-v1': 79.86, 'simclr-v2': 81.79, 'swav': 85.23},
    'sun397': {'byol': 66.36, 'deepcluster-v2': 81.41, 'infomin': 38.14, 'insdis': 31.08, 'moco-v1': 31.01, 'moco-v2': 34.81, 'pcl-v1': 25.84, 'pcl-v2': 39.06, 'sela-v2': 65.42, 'simclr-v1': 82.51, 'simclr-v2': 76.51, 'swav': 77.48},
    'voc2007': {'byol': 74.79, 'deepcluster-v2': 80.91, 'infomin': 74.28, 'insdis': 52.17, 'moco-v1': 55.92, 'moco-v2': 70.54, 'pcl-v1': 67.99, 'pcl-v2': 72.15, 'sela-v2': 77.46, 'simclr-v1': 78.87, 'simclr-v2': 77.76, 'swav': 79.28}}

    tw_lft = w_kendall_metric(score, lft_acc, dset)
    print("Kendall correlation for linear probing:{:12s} {}:{:2.3f}".format(dset,metric, tw_lft))
    

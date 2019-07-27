import os, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from dataloader import feature_extractor
from tqdm import tqdm


class precision_and_recall(object):
    def __init__(self, args):
        # parameters
        self.args = args
        # self.data_dir = args.data_dir
        self.result_dir = args.result_dir
        self.batch_size = args.batch_size
        self.cpu = args.cpu
        self.seed = args.seed
        self.data_size = args.data_size

    def run(self):
        
        # load data using vgg16
        extractor = feature_extractor(self.args)
        generated_features, real_features = extractor.extract()
        # print(generated_features)
        # equal number of samples
        data_num = min(len(generated_features), len(real_features))
        print(f'data num: {data_num}')

        if data_num <= 0:
            print("there is no data")
            return
        generated_features = generated_features[:data_num]
        real_features = real_features[:data_num]

        # get precision and recall
        precision = self.manifold_estimate(real_features, generated_features, 3)
        recall = self.manifold_estimate(generated_features, real_features, 3)
 
        print(precision)        
        print(recall)

    def manifold_estimate(self, A_features, B_features, k):
        
        KNN_list_in_A = {}
        for A in tqdm(A_features, ncols=80):
            pairwise_distances = np.zeros(shape=(len(A_features)))

            for i, A_prime in enumerate(A_features):
                d = torch.norm((A-A_prime), 2)
                pairwise_distances[i] = d

            v = np.partition(pairwise_distances, k)[k]
            KNN_list_in_A[A] = v

        n = 0 

        for B in tqdm(B_features, ncols=80):
            for A_prime in A_features:
                d = torch.norm((B-A_prime), 2)
                if d <= KNN_list_in_A[A_prime]:
                    n+=1
                    break

        return n/len(B_features)


class realism(object):
    def __init__(self, args):
        # parameters
        self.args = args
        # self.data_dir = args.data_dir
        self.result_dir = args.result_dir
        self.batch_size = args.batch_size
        self.cpu = args.cpu
        self.seed = args.seed   

    def run(self):
        return
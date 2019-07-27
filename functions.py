import os, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import operator

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
        self.data_size = args.data_size
        self.k = 3

    def run(self):
        
        # load data using vgg16
        extractor = feature_extractor(self.args)
        generated_features, real_features, _ = extractor.extract()
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
        precision = self.manifold_estimate(real_features, generated_features, self.k)
        recall = self.manifold_estimate(generated_features, real_features, self.k)
 
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
        self.k = 3  

    def run(self):

        # load data using vgg16
        extractor = feature_extractor(self.args)
        generated_features, real_features, generated_img_paths = extractor.extract()

        # equal number of samples
        data_num = min(len(generated_features), len(real_features))
        print(f'data num: {data_num}')

        if data_num <= 0:
            print("there is no data")
            return
        generated_features = generated_features[:data_num]
        real_features = real_features[:data_num]
        generated_img_paths = generated_img_paths[:data_num]

        KNN_list_in_real = self.calculate_real_NNK(real_features, self.k, data_num)

        for i, generated_feature in enumerate(tqdm(generated_features, ncols=80)):

            max_value = 0
            for real_feature, KNN_radius in KNN_list_in_real:
                d = torch.norm((real_feature-generated_feature), 2)
                value = KNN_radius/d
                if max_value < value:
                    max_value = value

            # print images with specific names
            if 'high_realism' in generated_img_paths[i] or 'low_realism' in generated_img_paths[i]:
                print(f'{generated_img_paths[i]} realism score: {max_value}')

        return

    def calculate_real_NNK(self, real_features, k, data_num):
        KNN_list_in_real = {}
        for real_feature in tqdm(real_features, ncols=80):
            pairwise_distances = np.zeros(shape=(len(real_features)))

            for i, real_prime in enumerate(real_features):
                d = torch.norm((real_feature-real_prime), 2)
                pairwise_distances[i] = d

            v = np.partition(pairwise_distances, k)[k]
            KNN_list_in_real[real_feature] = v

        # remove half of larger values
        KNN_list_in_real = sorted(KNN_list_in_real.items(), key=operator.itemgetter(1)) 
        KNN_list_in_real = KNN_list_in_real[:int(data_num/2)]


        return KNN_list_in_real
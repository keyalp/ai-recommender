import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.utils.data
from utils import Utils

#Building and preparing dataset
class Netflix1MDataset(torch.utils.data.Dataset):
    """
    Netflix 1M Dataset
    :param dataset_path: Netflix dataset path
    """
    def __init__(self, dataset_path, num_negatives_train=4, num_negatives_test=100, sep=','):
        colnames = ["user_id", 'item_id', 'label', 'timestamp']
        train_data = pd.read_csv(f''+dataset_path['train_dataset'], sep=sep, header=None, names=colnames).to_numpy()
        test_data = pd.read_csv(f''+dataset_path['test_dataset'], sep=sep, header=None, names=colnames).to_numpy()

        # TAKE items, targets and test_items
        self.targets = train_data[:, 2]
        self.items = self.preprocess_items(train_data)

        # Save dimensions of max users and items and build training matrix
        self.field_dims = np.max(self.items, axis=0) + 1 # ([ 7795, 10295+7795])
        ######################################################## OPTIONAL
        self.train_mat = Utils.build_adj_mx(self.field_dims[-1], self.items.copy())

        # Generate train interactions with 4 negative samples for each positive
        self.negative_sampling(num_negatives=num_negatives_train)

        # Build test set by passing as input the test item interactions
        self.test_set = self.build_test_set(self.preprocess_items(test_data),
                                            num_neg_samples_test = num_negatives_test)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.interactions[index]

    def preprocess_items(self, data, users=7795):
        reindexed_items = data[:, :2].astype(np.int32) #- Comença en 0 1  # -1 because ID begins from 1
        #users, items = np.max(reindexed_items, axis=0)[:2] + 1 # [ 7795, 10295])
        # Reindex items (we need to have [users + items] nodes with unique idx)
        reindexed_items[:, 1] = reindexed_items[:, 1] + users

        return reindexed_items

    def negative_sampling(self, num_negatives=4):
        self.interactions = []
        data = np.c_[(self.items, self.targets)].astype(int)
        max_users, max_items = self.field_dims[:2] # number users (7795), number items (10295+7795)

        for x in tqdm(data, desc="Performing negative sampling on test data..."):  # x are triplets (u, i , 1)
            # Append positive interaction
            self.interactions.append(x)
            # Copy user and maintain last position to 0. Now we will need to update neg_triplet[1] with j
            neg_triplet = np.vstack([x, ] * (num_negatives))
            neg_triplet[:, 2] = np.zeros(num_negatives)

            # Generate num_negatives negative interactions
            for idx in range(num_negatives):
                j = np.random.randint(max_users, max_items)
                # IDEA: Loop to exclude true interactions (set to 1 in adj_train) user - item
                while (x[0], j) in self.train_mat:
                    j = np.random.randint(max_users, max_items)
                neg_triplet[:, 1][idx] = j
            self.interactions.append(neg_triplet.copy())

        self.interactions = np.vstack(self.interactions)

    def build_test_set(self, gt_test_interactions, num_neg_samples_test=99):
        max_users, max_items = self.field_dims[:2] # number users (7795), number items (10295+7795)
        test_set = []
        for pair in tqdm(gt_test_interactions, desc="BUILDING TEST SET..."):
            negatives = []
            for t in range(num_neg_samples_test):
                j = np.random.randint(max_users, max_items)
                while (pair[0], j) in self.train_mat or j == pair[1]:
                    j = np.random.randint(max_users, max_items)
                negatives.append(j)
            #APPEND TEST SETS FOR SINGLE USER
            single_user_test_set = np.vstack([pair, ] * (len(negatives)+1))
            single_user_test_set[:, 1][1:] = negatives
            test_set.append(single_user_test_set.copy())
        return test_set
    


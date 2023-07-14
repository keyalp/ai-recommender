from tqdm import tqdm
import torch.utils.data
import scipy.sparse as sp
import math
import numpy as np
    
class Utils():
    def build_adj_mx(dims, interactions):
        train_mat = sp.dok_matrix((dims, dims), dtype=np.float32)
        for x in tqdm(interactions, desc="BUILDING ADJACENCY MATRIX..."):
            train_mat[x[0], x[1]] = 1.0
            train_mat[x[1], x[0]] = 1.0

        return train_mat
    
    def getHitRatio(recommend_list, gt_item):
        if gt_item in recommend_list:
            return 1
        else:
            return 0

    def getNDCG(recommend_list, gt_item):
        idx = np.where(recommend_list == gt_item)[0]
        if len(idx) > 0:
            return math.log(2)/math.log(idx+2)
        else:
            return 0
        
    def coverage (test_set,num_items,rank,model,device):
        recommend_list_all_users=[]
        for user_test in test_set:
            predictions = model.predict(user_test, device)
            _, indices = torch.topk(predictions, rank)
            recommend_list_user = user_test[indices.cpu().detach().numpy()][:, 1]
            for item in recommend_list_user:
                recommend_list_all_users.append(item)
        #print(len(recommend_list_all_users)) #77950
        num_items_recommended =len(np.unique(recommend_list_all_users))
        #print(num_items_recommended) #4675
        cov = num_items_recommended/num_items
        #print(num_items_recommended,"/",num_items)
        return cov
    
    def coverage_rnd(test_set, num_items, rank, rnd_model):
        random_recommend_list = rnd_model.predict(test_set)
        num_items_recommended = len(np.unique(random_recommend_list))
        cov = num_items_recommended / num_items
        return cov
from torch.utils.data import DataLoader, Dataset
from IPython import embed
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import csv
import os
import scipy.sparse as sp
from tqdm import tqdm, trange
import torch.utils.data
from preprocessing import Netflix1MDataset
from statistics import mean
from train import Train
from models import FactorizationMachineModel, RandomModel, PopularityModel
from utils import Utils

def format_pytorch_version(version):
    return version.split('+')[0]

def format_cuda_version(version):
    return 'cu' + version.replace('.', '')

#TODO dividir-ho en diferents funcions: Random, Pop, Test..
#TODO Que fem amb el train_model.test --> posar aquesta funció a cada model o que hem de fer

def main():

    TORCH_version = torch.__version__
    TORCH = format_pytorch_version(TORCH_version)
    CUDA_version = torch.version.cuda
    CUDA = format_cuda_version(CUDA_version)
    device = torch.device("cuda")

    dataset_path = 'sample_data/Subset1M'
    full_dataset= Netflix1MDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)

    #print(full_dataset.train_mat[:10])
    print(full_dataset.interactions[:20])

    df = pd.DataFrame(full_dataset.test_set)
    df.to_csv("sample_data/full_dataset_testset.csv", index= False)

    #Model, loss and optimizer definition
    model = FactorizationMachineModel(full_dataset.field_dims[-1], 32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    topk = 10
    #print (full_dataset.test_set[0])
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

    train_model = Train(model,optimizer,data_loader,criterion,device,topk)

    train_model.train_epochs

    #print(full_dataset.test_set[1])

    ########## TEST EVALUATION ##########
    user_test = full_dataset.test_set[1]
    out = model.predict(user_test,device)
    #print(out.shape)
    # Print first 10 predictions, where 1st one is the one for the GT
    out[:10]
    values, indices = torch.topk(out, 10)
    #print(values)
    #print(indices.cpu().detach().numpy())

    # RANKING LIST TO RECOMMEND
    recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]
    print('Recommended List: ',recommend_list)
    gt_item = 14966
    print(gt_item in recommend_list)

    #-------- Put in a class??

    coverage_per_item = 100*Utils.coverage(full_dataset.test_set,10295,topk,model,device)
    print(f'{coverage_per_item:.2f}')

    ########## RANDOM EVAL ##########
    # Check Init performance
    hr, ndcg = train_model.test(model, full_dataset, device, topk=topk)
    print("initial HR: ", hr)
    print("initial NDCG: ", ndcg)

    rnd_model = RandomModel(data_loader.dataset.field_dims)

    #Posar aixo en una funcio de train_epoch del Random o relacionat
    for epoch_i in range(20):
        hr, ndcg = train_model.test(rnd_model, full_dataset, device, topk=topk)
        print(f'epoch {epoch_i}:')
        print(f'Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')

    coverage_per_item = 100*Utils.coverage_rnd(data_loader.dataset.interactions, 10295, topk)
    #coverage_per_item = coverage(random_recommend_list,10295,topk)
    print(coverage_per_item)

    ########## POPULARITY EVAL?? ##########
    pop_model = PopularityModel(10295)
    user_test_pop = full_dataset.test_set[5][0][0]
    #print(user_test_pop)
    ranked_sorted = pop_model.fit(data_loader.dataset.interactions)
    pop_recommend_list = pop_model.predict(ranked_sorted, data_loader.dataset.interactions, user_test_pop)
    print (pop_recommend_list[:10])


    #print(len(pop_recommend_list[:0]))
    usersID = 7795
    topk = 10
    items_for_all_users = []

    for i in range(usersID):
        # extrec la llista de recomanacions per cada usuari
        pop_recommend_list_user = pop_model.predict(ranked_sorted, data_loader.dataset.interactions, i)
        # em quedo només amb la primera columna, item movieID
        #moviesTopk = [row[0] for row in pop_recommend_list_user][:topk]
        #print(moviesTopk)
        # afegeixo aquests valors en un nou array
        items_for_all_users.append(pop_recommend_list_user)
        print (i)

    flattened_items_for_all_users = np.array(items_for_all_users).flatten()
    num_items_recommended = np.unique(flattened_items_for_all_users)
    print(len(num_items_recommended))
    #coverage_pop = num_items_recommended / 10295
    #print(coverage_pop)
    
    #73 items diferents
    print (len(num_items_recommended)/10295)

if __name__ == "__main__":
    main()
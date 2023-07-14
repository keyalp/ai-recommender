from torch.utils.data import DataLoader
import numpy as np
import torch.utils.data
from preprocessing import Netflix1MDataset
from train_fm_model import TrainFmModel
from models import FactorizationMachineModel, RandomModel, PopularityModel
from utils import Utils
from test_model import Test
from tqdm import tqdm

#Global Variables:
hyperparams = {
    'topk':10,
    'lr' : 0.001, 
    'num_items' : 10295
}

dataset_path = {
    'train_dataset' : 'sample_data/Subset1M_traindata.csv',
    'test_dataset' : 'sample_data/Subset1M_testdata.csv'
}

def main():
    #Optional
    #TORCH_version = torch.__version__
    #print(TORCH_version)
    #TORCH = format_pytorch_version(TORCH_version)
    #CUDA_version = torch.version.cuda
    #CUDA = format_cuda_version(CUDA_version)

    device = torch.device("cuda")

    #Preprocessing
    full_dataset = Netflix1MDataset(dataset_path, num_negatives_train=4, num_negatives_test=99)
    data_loader = DataLoader(full_dataset, batch_size=256, shuffle=True, num_workers=0)

    model = 0
    while(model!=4):
        print("\n(1) Factorization Machine\n(2) Random\n(3) Popularity\n(4) Exit")
        model = int(input("Choose a model: "))
        if model == 1:
            run_fm_model(full_dataset,data_loader,device)
        elif model == 2:
            run_rand_model(full_dataset,data_loader,device)
        elif model == 3:
            run_pop_model(full_dataset,data_loader)
        elif model == 4:
            print("Closing Application")
        else:
            print("Wrong option")
        
    #print(full_dataset.train_mat[:10])
    #print(full_dataset.interactions[:20])

    #df = pd.DataFrame(full_dataset.test_set)
    #df.to_csv("sample_data/full_dataset_testset.csv", index= False)
    
########## FM MODEL ##########
def run_fm_model(full_dataset,data_loader,device):
    #Model, loss and optimizer definition
    model_fm = FactorizationMachineModel(full_dataset.field_dims[-1], 32).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(params=model_fm.parameters(), lr=hyperparams['lr'])
    #print (full_dataset.test_set[0])
    
    train_model_fm = TrainFmModel(model_fm,optimizer,data_loader,criterion,device,hyperparams['topk'])
    train_model_fm.do_epochs

    #print(full_dataset.test_set[1])

    ###TEST EVALUATION FM 
    user_test = full_dataset.test_set[1]
    out = model_fm.predict(user_test,device)
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

    coverage_per_item = 100*Utils.coverage(full_dataset.test_set,hyperparams['num_items'],hyperparams['topk'],model_fm,device)
    print(f'{coverage_per_item:.2f}')
    
    # Check Init performance
    hr, ndcg = Test.testModel(model_fm, full_dataset, device, topk=hyperparams['topk'])
    print("initial HR: ", hr)
    print("initial NDCG: ", ndcg)


########## RANDOM MODEL ##########
def run_rand_model(full_dataset,data_loader,device):
    rnd_model = RandomModel(data_loader.dataset.field_dims)
    topk = hyperparams['topk']
    #Posar aixo en una funcio de train_epoch del Random o relacionat
    for epoch_i in range(20):
        hr, ndcg = Test.testModel(rnd_model, full_dataset, device, topk=topk)
        print(f'epoch {epoch_i}:')
        print(f'Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')

    coverage_per_item = 100*Utils.coverage_rnd(data_loader.dataset.interactions, hyperparams['num_items'], topk, rnd_model)
    #coverage_per_item = coverage(random_recommend_list,hyperparams['num_items'],topk)
    print(coverage_per_item)

########## POPULARITY MODEL ##########
def run_pop_model(full_dataset,data_loader):
    topk = hyperparams['topk']
    pop_model = PopularityModel(hyperparams['num_items'],topk)
    user_test_pop = full_dataset.test_set[5][0][0]
    #print(user_test_pop)
    ranked_sorted = pop_model.fit(data_loader.dataset.interactions)
    pop_recommend_list = pop_model.predict(ranked_sorted, data_loader.dataset.interactions, user_test_pop,topk)
    print (pop_recommend_list[:10])

    #print(len(pop_recommend_list[:0]))
    #usersID = 7795
    usersID = 100
    items_for_all_users = []

    for i in tqdm(range(usersID)):
        # extrec la llista de recomanacions per cada usuari
        pop_recommend_list_user = pop_model.predict(ranked_sorted, data_loader.dataset.interactions, i,topk)
        # em quedo només amb la primera columna, item movieID
        #moviesTopk = [row[0] for row in pop_recommend_list_user][:topk]
        #print(moviesTopk)
        # afegeixo aquests valors en un nou array
        items_for_all_users.append(pop_recommend_list_user)
        #print (i)

    flattened_items_for_all_users = np.array(items_for_all_users).flatten()
    num_items_recommended = np.unique(flattened_items_for_all_users)
    print(len(num_items_recommended))
    #coverage_pop = num_items_recommended / 10295
    #print(coverage_pop)

    #73 items diferents
    print (len(num_items_recommended)/10295)

#Optional
def format_pytorch_version(version):
    return version.split('+')[0]

def format_cuda_version(version):
    if version is not None:
        return 'cu' + version.replace('.', '')
    else:
        return '' 

if __name__ == "__main__":
    main()


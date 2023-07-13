
from statistics import mean
from utils import Utils

class Train(): 
    def __init__(self,model, optimizer, data_loader, criterion, device, topk, log_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        self.log_interval = log_interval
        self.topk = topk

    def train_one_epoch(self):
        self.model.train()
        total_loss = []

        for i, (interactions) in enumerate(self.model.data_loader):
            interactions = interactions.to(self.model.device)
            #############################
            # Exercice 2: Select the column that contains the labels and store it into targets variable.
            #############################
            targets = interactions[:,2]
            predictions = self.model.model(interactions[:,:2])

            loss = self.model.criterion(predictions, targets.float())
            self.model.model.zero_grad()
            loss.backward()
            self.model.optimizer.step()
            total_loss.append(loss.item())

        return mean(total_loss)
    
    def test(self):
        # Test the HR and NDCG for the model @topK
        self.model.eval()

        HR, NDCG = [], []

        for user_test in self.full_dataset.test_set:
            gt_item = user_test[0][1]

            predictions = self.model.predict(user_test, self.device)
            _, indices = self.torch.topk(predictions, self.topk)
            recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

            HR.append(Utils.getHitRatio(recommend_list, gt_item))
            NDCG.append(Utils.getNDCG(recommend_list, gt_item))
            
        return mean(HR), mean(NDCG)

    def train_epochs(self):
        #Start training the model
        # DO EPOCHS NOW
        tb = False
        topk = 10
        for epoch_i in range(20):
            train_loss = self.train_one_epoch(self)
            hr, ndcg = self.test(self.model.model, self.model.full_dataset, self.model.device, topk=topk)
            print(f'epoch {epoch_i}:')
            print(f'training loss = {train_loss:.4f} | Eval: HR@{topk} = {hr:.4f}, NDCG@{topk} = {ndcg:.4f} ')
            """
            if tb:
                tb_fm.add_scalar('train/loss', train_loss, epoch_i)
                tb_fm.add_scalar('eval/HR@{topk}', hr, epoch_i)
                tb_fm.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)

                hr, ndcg = test(rnd_model, full_dataset, device, topk=topk)
                tb_rnd.add_scalar('eval/HR@{topk}', hr, epoch_i)
                tb_rnd.add_scalar('eval/NDCG@{topk}', ndcg, epoch_i)
            """

            #Defining Test function
    
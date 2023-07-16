from statistics import mean
from test_model import Test

class TrainFmModel(): 
    def __init__(self,model, optimizer, data_loader, full_dataset, criterion, device, topk, log_interval=100):
        self.model = model
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.full_dataset = full_dataset
        self.criterion = criterion
        self.device = device
        self.log_interval = log_interval
        self.topk = topk

    def train_one_epoch(self):
        self.model.train()
        total_loss = []

        for i, (interactions) in enumerate(self.data_loader):
            interactions = interactions.to(self.device)
            targets = interactions[:,2]
            predictions = self.model(interactions[:,:2])

            loss = self.criterion(predictions, targets.float())
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss.append(loss.item())

        return mean(total_loss)

    def do_epochs(self):
        #Start training the model
        # DO EPOCHS NOW
        tb = False
        topk = 10
        print("Training...")
        for epoch_i in range(20):
            train_loss = self.train_one_epoch()
            hr, ndcg = Test.testModel(self.model, self.full_dataset, self.device, topk=topk)
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
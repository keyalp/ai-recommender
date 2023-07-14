import torch.utils.data
from statistics import mean
from utils import Utils


#Defining Test function

class Test():

    def testModel(model, full_dataset, device, topk=10): 
        # Test the HR and NDCG for the model @topK
        model.eval()

        HR, NDCG = [], []

        for user_test in full_dataset.test_set:
            gt_item = user_test[0][1]

            predictions = model.predict(user_test, device)
            _, indices = torch.topk(predictions, topk)
            recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]

            HR.append(Utils.getHitRatio(recommend_list, gt_item))
            NDCG.append(Utils.getNDCG(recommend_list, gt_item))
        return mean(HR), mean(NDCG)
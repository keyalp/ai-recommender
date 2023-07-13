import torch.utils.data
from utils import FeaturesLinear, FM_operation

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        #############################
        # Exercice 1: Write code of the model with a usual embedding layer in Pytorch documentation. Then, check that in the
        # forward pass, we are building the expression of factorization machines.
        #############################

        self.linear = FeaturesLinear(field_dims,1)
        self.embedding = torch.nn.Embedding(field_dims, embedding_dim=embed_dim)
        self.fm = FM_operation(reduce_sum=True)

        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, interaction_pairs):
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs) + self.fm(self.embedding(interaction_pairs))

        return out.squeeze(1)

    def predict(self, interactions, device):
        # return the score, inputs are numpy arrays, outputs are tensors
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores
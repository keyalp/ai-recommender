import torch.utils.data
#from utils import FeaturesLinear, FM_operation
import random
import numpy as np

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

        self.linear = self.FeaturesLinear(field_dims,1)
        self.embedding = torch.nn.Embedding(field_dims, embedding_dim=embed_dim)
        self.fm = self.FM_operation(reduce_sum=True)

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

    # Linear part of the equation
    class FeaturesLinear(torch.nn.Module):

        def __init__(self, field_dims, output_dim=1):
            super().__init__()

            self.emb = torch.nn.Embedding(field_dims, output_dim)
            self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

        def forward(self, x):
            """
            :param x: Long tensor of size ``(batch_size, num_fields)``
            """
            # self.fc(x).shape --> [batch_size, num_fields, 1]
            # torch.sum(self.fc(x), dim=1).shape --> ([batch_size, 1])
            return torch.sum(self.emb(x), dim=1) + self.bias

    # FM part of the equation
    class FM_operation(torch.nn.Module):

        def __init__(self, reduce_sum=True):
            super().__init__()
            self.reduce_sum = reduce_sum

        def forward(self, x):
            """
            :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
            """
            square_of_sum = torch.sum(x, dim=1) ** 2
            sum_of_square = torch.sum(x ** 2, dim=1)
            ix = square_of_sum - sum_of_square
            if self.reduce_sum:
                ix = torch.sum(ix, dim=1, keepdim=True)
            return 0.5 * ix
    
    
class RandomModel(torch.nn.Module):
    def __init__(self, dims):
        super(RandomModel, self).__init__()
        """
        Simple random based recommender system
        """
        self.all_items = list(range(dims[0], dims[1]))

    def forward(self):
        pass

    def predict(self, interactions, device=None):
         # Asegurarse de que el tamaño del muestreo no sea mayor que el tamaño de self.all_items
        sample_size = min(len(interactions), len(self.all_items))
        return torch.FloatTensor(random.sample(self.all_items, sample_size))
        #return torch.FloatTensor(random.sample(self.all_items, len(interactions)))


class PopularityModel:
    def __init__(self, num_items,topk):
        self.num_items = num_items
        self.item_popularity = None
        self.topk = topk

    def forward(self):
        pass

    def fit(self, interactions):
        """
        Ajusta el modelo de popularidad utilizando las interacciones de los usuarios.
        """
        # Calcula la popularidad de los elementos basándose en el recuento de interacciones
        movieid_column = interactions[:, 1]
        rating_column = interactions[:, 2]
        # Crear una máscara booleana para identificar los elementos con rating igual a 1
        mask = rating_column == 1
        # Filtrar los movieid correspondientes a los elementos con rating igual a 1
        rated_movieids = movieid_column[mask]
        # Obtener los items únicos con rating igual a 1
        unique_rated_movieids, counts = np.unique(rated_movieids, return_counts=True)
        #print(unique_rated_movieids)
        # Normaliza la popularidad para obtener una distribución de probabilidad
        n_interactions = len(interactions)
        self.item_popularity =  counts / n_interactions
        #print(n_interactions)
        #print("movieID:",unique_rated_movieids,"counts:", counts, "popularity:" ,self.item_popularity)
        ranked_list = []
        for i in range(len(unique_rated_movieids)):
            columna1 = unique_rated_movieids[i]
            columna2 = self.item_popularity[i]
            ranked_list.append([columna1, columna2])
        ranked_sorted = sorted(ranked_list, key=lambda x: x[1], reverse=True)
        #print(len(ranked_sorted)) # aquelles pelicules q tenen alguna interaccio
        #com sé si l'usuari ja ha vist alguna de les pelis de la llista?
        return ranked_sorted

    def predict(self, ranked_sorted1, interactions, userID,topk):
        #if user-item-interaction = 1 descarta'l del ranking
        # Obtener los movieID con interacción igual a 1
        movieID_interaccion_1 = interactions[((interactions[:, 2] == 1) & (interactions[:,0] == userID)), 1]
        #print(len(movieID_interaccion_1)) # aquelles pelicules q usuari té interaccio
        # Eliminar las filas de ranked_sorted donde movieID está en movieID_interaccion_1
        #ranked_sorted = ranked_sorted[~np.isin(ranked_sorted[:, 0], movieID_interaccion_1)]
        #Filtrar las filas de ranked_sorted donde movieID está en movieID_interaccion_1
        ranked_sorted1 = [row for row in ranked_sorted1 if row[0] not in movieID_interaccion_1]
        #print(len(ranked_sorted))
        return ranked_sorted1[:topk]
    
    class PopularityModel: #Traduced
        def __init__(self, num_items, topk):
            self.num_items = num_items
            self.item_popularity = None
            self.topk = topk

        def forward(self):
            pass

        def fit(self, interactions):
            """
            Fit the popularity model using user interactions.
            """
            # Calculate item popularity based on interaction counts
            movieid_column = interactions[:, 1]
            rating_column = interactions[:, 2]
            # Create a boolean mask to identify items with rating equal to 1
            mask = rating_column == 1
            # Filter movieids corresponding to items with rating equal to 1
            rated_movieids = movieid_column[mask]
            # Get unique items with rating equal to 1
            unique_rated_movieids, counts = np.unique(rated_movieids, return_counts=True)
            # Normalize popularity to obtain a probability distribution
            n_interactions = len(interactions)
            self.item_popularity = counts / n_interactions
            ranked_list = []
            for i in range(len(unique_rated_movieids)):
                column1 = unique_rated_movieids[i]
                column2 = self.item_popularity[i]
                ranked_list.append([column1, column2])
            ranked_sorted = sorted(ranked_list, key=lambda x: x[1], reverse=True)
            return ranked_sorted

        def predict(self, ranked_sorted, interactions, userID, topk):
            # Exclude ranked items if user-item interaction = 1
            # Get movieIDs with interaction equal to 1
            movieID_interact_1 = interactions[((interactions[:, 2] == 1) & (interactions[:,0] == userID)), 1]
            # Remove rows from ranked_sorted where movieID is in movieID_interact_1
            ranked_sorted = [row for row in ranked_sorted if row[0] not in movieID_interact_1]
            return ranked_sorted[:topk]
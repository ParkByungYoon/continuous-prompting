import torch
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import MessagePassing

import torch
from torch import nn, Tensor


class RecsysContinuousPromptModel(torch.nn.Module):
    def __init__(self, num_users, num_items, edge_index_path):
        super().__init__()
        self.model = LightGCN(num_users=num_users, num_items=num_items)
        edge_index = torch.load(edge_index_path).type(torch.long)
        self.edge_index = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_users + num_items, num_users + num_items))
        
    def forward(self, user_id, item_ids):
        device = next(self.model.parameters()).device
        edge_index = self.edge_index.to(device)
        user_id = user_id.to(device)
        item_ids = item_ids.to(device)
        users_emb_final, _, items_emb_final, _ = self.model(edge_index)

        return torch.cat([users_emb_final[user_id], items_emb_final[item_ids]], dim=1)


class LightGCN(MessagePassing):
    """LightGCN Model as proposed in https://arxiv.org/abs/2002.02126
    """

    def __init__(self, num_users, num_items, embedding_dim=64, K=3, add_self_loops=False):
        """Initializes LightGCN Model

        Args:
            num_users (int): Number of users
            num_items (int): Number of items
            embedding_dim (int, optional): Dimensionality of embeddings. Defaults to 8.
            K (int, optional): Number of message passing layers. Defaults to 3.
            add_self_loops (bool, optional): Whether to add self loops for message passing. Defaults to False.
        """
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding_dim, self.K = embedding_dim, K
        self.add_self_loops = add_self_loops

        self.users_emb = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_dim) # e_u^0
        self.items_emb = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.embedding_dim) # e_i^0

        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index: SparseTensor):
        """Forward propagation of LightGCN Model.

        Args:
            edge_index (SparseTensor): adjacency matrix

        Returns:
            tuple (Tensor): e_u_k, e_u_0, e_i_k, e_i_0
        """
        # compute \tilde{A}: symmetrically normalized adjacency matrix
        edge_index_norm = gcn_norm(
            edge_index, add_self_loops=self.add_self_loops)

        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight]) # E^0
        embs = [emb_0]
        emb_k = emb_0

        # multi-scale diffusion
        for i in range(self.K):
            emb_k = self.propagate(edge_index_norm, x=emb_k)
            embs.append(emb_k)

        embs = torch.stack(embs, dim=1)
        emb_final = torch.mean(embs, dim=1) # E^K

        users_emb_final, items_emb_final = torch.split(
            emb_final, [self.num_users, self.num_items]) # splits into e_u^K and e_i^K

        # returns e_u^K, e_u^0, e_i^K, e_i^0
        return users_emb_final, self.users_emb.weight, items_emb_final, self.items_emb.weight

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        # computes \tilde{A} @ x
        return matmul(adj_t, x)
    
    def bce_loss(self, pred: Tensor, edge_label: Tensor, **kwargs) -> Tensor:
        r"""Computes the model loss for a link prediction objective via the
        :class:`torch.nn.BCEWithLogitsLoss`.

        Args:
            pred (torch.Tensor): The predictions.
            edge_label (torch.Tensor): The ground-truth edge labels.
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch.nn.BCEWithLogitsLoss` loss function.
        """
        loss_fn = torch.nn.BCEWithLogitsLoss(**kwargs)
        return loss_fn(pred, edge_label.to(pred.dtype))

    def bpr_loss(self, users_emb_final, users_emb_init, pos_items_emb_final, pos_items_emb_init, neg_items_emb_final, neg_items_emb_init, lambda_val):
        """Bayesian Personalized Ranking Loss as described in https://arxiv.org/abs/1205.2618

        Args:
            users_emb_final (torch.Tensor): e_u_k
            users_emb_0 (torch.Tensor): e_u_0
            pos_items_emb_final (torch.Tensor): positive e_i_k
            pos_items_emb_0 (torch.Tensor): positive e_i_0
            neg_items_emb_final (torch.Tensor): negative e_i_k
            neg_items_emb_0 (torch.Tensor): negative e_i_0
            lambda_val (float): lambda value for regularization loss term

        Returns:
            torch.Tensor: scalar bpr loss value
        """
        reg_loss = lambda_val * (users_emb_init.norm(2).pow(2) +
                                pos_items_emb_init.norm(2).pow(2) +
                                neg_items_emb_init.norm(2).pow(2)) # L2 loss

        pos_scores = torch.mul(users_emb_final, pos_items_emb_final)
        pos_scores = torch.sum(pos_scores, dim=-1) # predicted scores of positive samples
        neg_scores = torch.mul(users_emb_final, neg_items_emb_final)
        neg_scores = torch.sum(neg_scores, dim=-1) # predicted scores of negative samples

        loss = -torch.mean(torch.nn.functional.softplus(pos_scores - neg_scores)) + reg_loss

        return loss
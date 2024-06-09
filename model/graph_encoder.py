import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv


class GraphContinuousPromptModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = GIN(input_dim, hidden_dim)
        
    def forward(self, x, edge_index):
        device = next(self.model.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        # return torch.mean(self.model(x, edge_index), dim=0).unsqueeze(dim=0)
        return self.model(x, edge_index)
    

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        for i in range(num_layers):
            conv = GCNConv(input_dim, hidden_dim)
            self.convs.append(conv)
            input_dim = hidden_dim

    def forward(self, x, edge_index):
        if x.size()[0] == 1:
            x = x.view(-1, self.input_dim)
            edge_index = edge_index.view(2,-1)

        for conv in self.convs:
            x = F.relu((conv(x, edge_index)))
        return x
    

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(input_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            conv = GINConv(mlp, train_eps=True)

            self.convs.append(conv)
            input_dim = hidden_dim

    def forward(self, x, edge_index):
        if x.size()[0] == 1:
            x = x.view(-1, self.input_dim)
            edge_index = edge_index.view(2,-1)

        for conv in self.convs:
            x = F.relu((conv(x, edge_index)))
        return x
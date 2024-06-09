import torch

class BasicProjection(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.projection_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 2048),
        )

    def forward(self, x):
        return self.projection_module(x)
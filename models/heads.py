import torch

class MLP(torch.nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=in_features, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):
        return self.model(x)
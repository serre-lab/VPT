import torch

class LinearModel(torch.nn.Module):
    def __init__(self, hidden_size, num_classes=1, dropout_rate=0.3):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.linear = torch.nn.Linear(hidden_size, num_classes)
        self.batchnorm = torch.nn.BatchNorm1d(hidden_size, affine=False, eps=1e-6)
    def forward(self, x):
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
    
class LinearModelMulti(torch.nn.Module):
    def __init__(self, hidden_size, num_classes=1, dropout_rate=0.3):
        super().__init__()
        print(hidden_size)
        self.linear = torch.nn.Sequential(torch.nn.BatchNorm1d(hidden_size[0], affine=False, eps=1e-6), 
                                            torch.nn.Dropout(p=dropout_rate),
                                            torch.nn.Linear(hidden_size[0], hidden_size[1]),
                                            torch.nn.BatchNorm1d(hidden_size[1], affine=False, eps=1e-6),
                                            torch.nn.Dropout(p=dropout_rate),
                                            torch.nn.Linear(hidden_size[1], num_classes))
    def forward(self, x):
        return self.linear(x)
        
class LinearProbeModel(torch.nn.Module):
    def __init__(self, backbone, linear_model):
        super().__init__()
        self.backbone = backbone
        self.linear_model = linear_model
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.linear_model(x)
        return x
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LogisticRegression, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        

    def forward(self, x):
        x = self.layer(x)
        x = F.sigmoid(x)
        return x
    

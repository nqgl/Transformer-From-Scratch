import torch as t
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, nonlinearity=nn.LeakyReLU, bias = True, bias_only_first_layer=False, layer_norm=False):
        super(MLP, self).__init__()
        layer_sizes = [input_size] + layer_sizes + [output_size]
        self.layers = [nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias or (not bias_only_first_layer or i == 0)) for i in range(len(layer_sizes)-1)]
        if layer_norm: # between last layer and output
            self.layers.insert(-1, nn.LayerNorm(layer_sizes[-1]))

        if type(nonlinearity) in (list, tuple):
            nonlinearity = [nonlinearity[i]() for i in range(len(layer_sizes)-1)]
        else:
            nonlinearity = [nonlinearity() for i in range(len(layer_sizes)-1)]
        sequence = [layer for pair in zip(self.layers, nonlinearity) for layer in pair][:-1]
        self.mlp = nn.Sequential(*sequence)
        

    def forward(self, x):
        x = self.mlp(x)
        return x
    
if __name__ == '__main__':
    m = MLP(10, [10, 10], 10)
    print(m)
    x = t.rand(1, 10)
    print(x)
    print(m(x))
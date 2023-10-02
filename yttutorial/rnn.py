import torch as t
import torch.nn as nn
import torch.nn.functional as F
import mlp


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers = 10, hiddenNonlinearity=nn.LeakyReLU()):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.W = mlp.MLP(input_size + hidden_size, [input_size + hidden_size] * hidden_layers, hidden_size + output_size, layer_norm=False) # input to hidden
        # self.i2o = mlp.MLP(input_size + hidden_size, [input_size + hidden_size] * hidden_layers, output_size) # input to output
        self.width = hidden_size + output_size
        self.depth = hidden_layers
        self.hiddenNonlinearity = hiddenNonlinearity
        h0 = t.rand(1, hidden_size)
        self.h_0 = nn.Parameter(h0/h0.norm(1))
        self.h_0.requires_grad = True
        self.dropout = nn.Dropout(0.05)
        # self.layer_norm = nn.LayerNorm(self.width) #layer norm off because it seems in rnn its bad. Because it normalizes over the batch dimension, which is the time dimension in rnn.

    def forward(self, input, hidden):
        combined = t.cat((input, hidden), 1)
        hx = self.W(combined)
        # hx = self.layer_norm(hx)
        h = self.dropout(hx[:, :self.hidden_size])
        y = hx[:, self.hidden_size:]/self.width ** 0.5

        # y = t.cat((t.sigmoid(y[:, :1]), t.softmax(y[:, 1:], dim=1)), dim=1)
        # y = t.clamp(y, 1e6, 1 - 1e6)
        return y, h 
    
    def initHidden(self, batchsize):
        return t.zeros(batchsize, self.hidden_size, device=self.h_0.device) + self.h_0

    
class StandardRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers = 10, hiddenNonlinearity=nn.LeakyReLU()):
        super(StandardRNN, self).__init__()
        self.hidden_size = hidden_size
        self.W = mlp.MLP(input_size + hidden_size, [input_size + hidden_size] * hidden_layers, hidden_size)
        self.i2o = mlp.MLP(input_size + hidden_size, [], output_size)
        # self.W = mlp.MLP(input_size + hidden_size, [input_size + hidden_size] * hidden_layers, hidden_size + output_size, layer_norm=False) # input to hidden
        # self.i2o = mlp.MLP(input_size + hidden_size, [input_size + hidden_size] * hidden_la   ers, output_size) # input to output
        self.width = hidden_size + output_size
        self.depth = hidden_layers
        self.hiddenNonlinearity = hiddenNonlinearity
        h0 = t.rand(1, hidden_size)
        self.h_0 = nn.Parameter(h0/h0.norm(2))
        self.h_0.requires_grad = True
        self.dropout = nn.Dropout(0.)
        self.layer_norm = nn.LayerNorm(self.width) #layer norm off because it seems in rnn its bad. Because it normalizes over the batch dimension, which is the time dimension in rnn.

    def forward(self, input, hidden):
        combined = t.cat((input, hidden), 1)
        h = self.W(combined)
        y = self.i2o(combined)
        h = self.layer_norm(h)
        h = self.dropout(h)
        h = self.hiddenNonlinearity(h)
        y = y/self.width ** 0.5
        
        # y = t.clamp(y, 1e6, 1 - 1e6)
        return y, h 
    
    def initHidden(self, batchsize):
        return t.zeros(batchsize, self.hidden_size, device=self.h_0.device) + self.h_0

    


def most_recent_model():
    import tokenizer 
    import os
    models = os.listdir("./models")
    models.sort()
    print(models[-1])
    # parse the model path to get the parameters
    import re
    print(models[-1])
    match = re.match(r"sspeare_rnn_(\d+)\.(\d+)-(\d+)_\d+-\d+-\d+_epoch\d+\.pt", models[-1])
    print(match.group(2), match.group(3))
    model = RNN(tokenizer.ALPHABET_SIZE, int(match.group(3)) - tokenizer.ALPHABET_SIZE, tokenizer.ALPHABET_SIZE, hidden_layers=int(match.group(2)))
    model.load_state_dict(t.load("./models/" + models[-1]))
    return model


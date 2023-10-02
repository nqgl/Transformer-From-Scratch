import torch as t
import torch.nn as nn
import torch.nn.functional as F
import transformer.mlp as mlp
import transformer_attention
from typing import List
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, P_dropout):
        self.multihead_attention = transformer_attention.MultiHeadSDPAttention(d_model, h)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(P_dropout)
        self.feed_forward = mlp.MLP(d_model, [d_ff], d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(P_dropout)

    def forward(self, x):
        att = self.multihead_attention(x)
        att = self.dropout1(att)
        x = self.layer_norm1(att + x) 
        ff = self.feed_forward(x)
        ff = self.dropout2(ff)
        x = self.layer_norm2(ff + x)
        return x
    
class EncoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, h, d_ff, P_dropout, N):
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, h, d_ff, P_dropout) for i in range(N)])
        self.linear = nn.Linear(d_model, d_model)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerDecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, h, d_ff, P_dropout):
        super(TransformerDecoderOnlyLayer, self).__init__()
        self.multihead_attention :transformer_attention.MaskedMultiHeadSDPAttention = transformer_attention.MaskedMultiHeadSDPAttention(d_model, h)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(P_dropout)
        self.feed_forward = mlp.MLP(d_model, [d_ff], d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(P_dropout)

    def forward(self, x):
        att = self.multihead_attention(x)
        att = self.dropout1(att)
        x = self.layer_norm1(att + x) 
        ff = self.feed_forward(x)
        ff = self.dropout2(ff)
        x = self.layer_norm2(ff + x)
        return x
    
    def generate_first(self, x):
        att = self.multihead_attention.generate_first(x)
        x = self.layer_norm1(att + x) 
        ff = self.feed_forward(x)
        x = self.layer_norm2(ff + x)
        return x


    def generate_next(self, x):
        next_att = self.multihead_attention.generate_next(x)
        x = self.layer_norm1(next_att + x)
        ff = self.feed_forward(x)
        x = self.layer_norm2(ff + x)
        return x    

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, h, d_ff, P_dropout, N):
        super(DecoderOnlyTransformer, self).__init__()
        self.layers :List[TransformerDecoderOnlyLayer] = nn.ModuleList([TransformerDecoderOnlyLayer(d_model, h, d_ff, P_dropout) for i in range(N)])
        self.linear = nn.Linear(d_model, d_model)
        self.d_model = d_model

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x)
        return x/self.d_model ** 0.5
    
    def set_mask_sight(self, foresight, hindsight=None):
        for layer in self.layers:
            layer.multihead_attention.set_mask_sight(foresight, hindsight)
    
    def begin_generate(self, length, batchsize = 1, device = t.device('cuda')):
        for l in self.layers:
            l.multihead_attention.begin_generate(length, batchsize, device)

    def generate_first(self, x):
        for l in self.layers:
            x = l.generate_first(x)
        x = self.linear(x)
        return x/self.d_model ** 0.5
    
    def generate_next(self, x):
        for l in self.layers:
            x = l.generate_next(x)
        x = self.linear(x)
        return x/self.d_model ** 0.5
        
def most_recent_model(path=None):
    import os
    import glob
    import re
    import pickle
    import numpy as np
    files = glob.glob('./models/sspeare_ztransformer_*.pt' if path is None else path)
    files = sorted(files, key=lambda s: int(re.findall(r'\d+', s)[-1]))
    print(files[-1])
    model = DecoderOnlyTransformer(128,4, 512, 0.2, 6)
    model.load_state_dict(t.load(files[-1]))
    return model

def main():
    model = most_recent_model()

if __name__ == '__main__':
    main()

def inspect(model :DecoderOnlyTransformer):
    i_values = []
    for layer in model.layers:
        i_values.append(layer.multihead_attention.i)
        for head in layer.multihead_attention.attentions:
            # i_values.append(head.i)
            i_values.append(head.SDPAttention.i)
    print(model.layers[1].multihead_attention.O)
    input()
    print(model.layers[1].multihead_attention.attentions[0].SDPAttention.V)
    input()
    return i_values
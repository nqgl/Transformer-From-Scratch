import torch as t
import torch.nn as nn
import torch.nn.functional as F
import mlp
import transformer_attention
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
        self.multihead_attention = transformer_attention.MaskedMultiHeadSDPAttention(d_model, h)
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
    

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, h, d_ff, P_dropout, N):
        super(DecoderOnlyTransformer, self).__init__()
        self.layers = nn.ModuleList([TransformerDecoderOnlyLayer(d_model, h, d_ff, P_dropout) for i in range(N)])
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
    
class PTDecoderOnlyTransformer(nn.Module):
    def __init__(self, d_model, h, d_ff, P_dropout, N):
        super(PTDecoderOnlyTransformer, self).__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, h, d_ff, P_dropout) for i in range(N)])
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.linear(x)
        return x


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

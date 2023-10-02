import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, d_v):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V):
        scaled_dot_product = torch.bmm(Q, torch.einsum('ijk->ikj', K)) / self.d_k ** 0.5
        attention = self.softmax(scaled_dot_product)
        output = torch.bmm(attention, V)
        return output

class MaskGenerator():
    def __init__(self, mask_value=0):
        self.mask_value = mask_value
        self.masks = {}
        self.future_sight = 0
        self.past_sight = None

    def make_mask(self, input):
        mask_shape = input.shape
        if mask_shape not in self.masks:
            # batch, sequence, d_model
            mask = torch.zeros(mask_shape, dtype=torch.bool, device=input.device)
            for i in range(mask_shape[1]):
                mask[:, i, i + 1 + self.future_sight:] = True # wrong for testing purposes
                if self.past_sight is not None:
                    mask[:, i, :max(0, i - self.past_sight)] = True # wrong for testing purposes
                
            self.masks[mask_shape] = mask
        return self.masks[mask_shape]
    
class MaskedScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, d_v):
        super(MaskedScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.softmax = nn.Softmax(dim=2)
        self.mask_generator = MaskGenerator()
        self.attention_pattern = None

    def forward(self, Q, K, V):
        scaled_dot_product = torch.bmm(Q, torch.einsum('ijk->ikj', K)) / self.d_k ** 0.5
        mask = self.mask_generator.make_mask(scaled_dot_product)
        scaled_dot_product[mask] = -1e9
        attention = self.softmax(scaled_dot_product)
        self.attention_pattern = attention
        output = torch.bmm(attention, V)
        return output


    
class LearnedProjectionAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, Attention=ScaledDotProductAttention):
        super(LearnedProjectionAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
        self.SDPAttention = Attention(d_k, d_v)
        
    def forward(self, input):
        Q = self.W_Q(input)
        K = self.W_K(input)
        V = self.W_V(input)
        # print(Q.shape, K.shape, V.shape)
        output = self.SDPAttention(Q, K, V)
        return output
    
class MultiHeadSDPAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MultiHeadSDPAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = self.d_k
        assert d_model % h == 0
        self.attentions = nn.ModuleList([LearnedProjectionAttention(self.d_k, self.d_v, d_model) for i in range(h)])
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, input):
        attentions = [attention(input) for attention in self.attentions]
        attentions_concat = torch.cat(attentions, dim=2)
        output = self.W_O(attentions_concat)
        return output
    

class MaskedMultiHeadSDPAttention(nn.Module):
    def __init__(self, d_model, h):
        super(MaskedMultiHeadSDPAttention, self).__init__()
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.d_v = self.d_k
        assert d_model % h == 0
        self.attentions = nn.ModuleList([LearnedProjectionAttention(self.d_k, self.d_v, d_model, Attention=MaskedScaledDotProductAttention) for i in range(h)])
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, input):
        attentions = [attention(input) for attention in self.attentions]
        attentions_concat = torch.cat(attentions, dim=2)
        output = self.W_O(attentions_concat)
        return output
    
    def set_mask_sight(self, foresight, hindsight=None):
        for attention in self.attentions:
            attention.SDPAttention.mask_generator.future_sight = foresight
            attention.SDPAttention.mask_generator.past_sight = hindsight
            attention.SDPAttention.mask_generator.masks = {}

def main():
    import sspear_parse
    import tokenizer
    tokens = sspear_parse.sspeare_tensor[200:300]
    tokens = tokens.unsqueeze(0)
    att = LearnedProjectionAttention(10, 10, tokenizer.ALPHABET_SIZE)
    att = MaskedMultiHeadSDPAttention(tokenizer.ALPHABET_SIZE, 4)
    att = MaskedScaledDotProductAttention(10, 10)
    att.mask_generator.past_sight = -1
    # print(att(tokens))
    Q, K, V = torch.ones(1, 10, 10), torch.ones(1, 10, 10), torch.ones(1, 10, 10)
    o = att(Q, K, V)
    print(att.attention_pattern)
if __name__ == '__main__':
    main()

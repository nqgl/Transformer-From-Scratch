import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.save_states = False
        # self.Q = None
        self.K = None
        self.V = None
        self.output = None
        self.i = 0

    def forward(self, Q, K, V):
        scaled_dot_product = torch.bmm(Q, torch.einsum('ijk->ikj', K)) / self.d_k ** 0.5
        mask = self.mask_generator.make_mask(scaled_dot_product)
        scaled_dot_product[mask] = -1e9
        attention = self.softmax(scaled_dot_product)
        output = torch.bmm(attention, V)
        return output

    def unmasked_forward(self, Q, K, V):
        scaled_dot_product = torch.bmm(Q, torch.einsum('ijk->ikj', K)) / self.d_k ** 0.5
        attention = self.softmax(scaled_dot_product)
        output = torch.bmm(attention, V)
        return output


    def begin_generate(self, length, batchsize = 1, device = torch.device('cuda')):
        self.save_states = True
        # self.Q = torch.zeros(batchsize, length, self.d_k, dtype=torch.float, device=device)
        self.K = torch.zeros(batchsize, length, self.d_k, dtype=torch.float, device=device)
        self.V = torch.zeros(batchsize, length, self.d_v, dtype=torch.float, device=device)
        self.output = torch.zeros(batchsize, length, self.d_v, dtype=torch.float, device=device)

    def generate_first(self, Q, K, V):
        self.i = Q.shape[1]
        assert self.save_states and Q.shape[1] == K.shape[1] == V.shape[1]
        # self.Q[:, :self.i, :] = Q
        self.K[:, :self.i, :] = K
        self.V[:, :self.i, :] = V
        self.output[:, :self.i, :] = self.forward(Q, K, V)
        return self.output[:, :self.i, :]

    def generate_next(self, q, k, v):
        assert self.save_states
        # self.Q[:, self.i:self.i+1, :] = q
        self.K[:, self.i:self.i+1, :] = k
        self.V[:, self.i:self.i+1, :] = v
        if False:
            scaled_dot_product = torch.bmm(q, torch.einsum('ijk->ikj', self.K[:,:self.i + 1, :])) / self.d_k ** 0.5
            attention = self.softmax(scaled_dot_product)
            output = torch.bmm(attention, self.V[:,:self.i + 1, :])
        else:
            output = self.unmasked_forward(q, self.K[:,:self.i + 1, :], self.V[:,:self.i + 1, :])
        self.output[:, self.i:self.i+1, :] = output 
        self.i += 1 
        return output
    
class LearnedProjectionAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, Attention=MaskedScaledDotProductAttention):
        super(LearnedProjectionAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)
        self.SDPAttention :MaskedScaledDotProductAttention = Attention(d_k, d_v)
        
    def forward(self, input):
        Q = self.W_Q(input)
        K = self.W_K(input)
        V = self.W_V(input)
        # print(Q.shape, K.shape, V.shape)
        output = self.SDPAttention(Q, K, V)
        return output


    def begin_generate(self, length, batchsize = 1, device = torch.device('cuda')):
        return self.SDPAttention.begin_generate(length, batchsize, device)

    def generate_first(self, input):
        Q = self.W_Q(input)
        K = self.W_K(input)
        V = self.W_V(input)
        return self.SDPAttention.generate_first(Q, K, V)

    def generate_next(self, next_input):
        q, k, v = self.W_Q(next_input), self.W_K(next_input), self.W_V(next_input)
        return self.SDPAttention.generate_next(q, k, v)

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
        self.O = None
        self.i = None
    
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

    def begin_generate(self, length, batchsize = 1, device = torch.device('cuda')):
        for attention in self.attentions:
            attention.begin_generate(length, batchsize, device)
        self.O = torch.zeros(batchsize, length, self.d_model, dtype=torch.float, device=device)
        self.i = 0

    
    def generate_first(self, input):
        attentions = [attention.generate_first(input) for attention in self.attentions]
        attentions_concat = torch.cat(attentions, dim=2)
        self.i = input.shape[1]
        self.O[:, :self.i, :] = self.W_O(attentions_concat)
        return self.O[:, :self.i, :]

    def generate_next(self, next_input):
        attentions = [attention.generate_next(next_input) for attention in self.attentions]
        attentions_concat = torch.cat(attentions, dim=2)
        o = self.W_O(attentions_concat)
        self.O[:, self.i:self.i+1, :] = o
        self.i += 1
        return o


def main():
    from ..traindata import parsedfiles, tokenizer
    tokens = parsedfiles.sspeare_tensor[200:300]
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

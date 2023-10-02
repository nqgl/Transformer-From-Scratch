import torch
alphabet_old = "abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n\t\'"

alphabet = "*abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?:'\"/_#$^+=<>()[]\n\t\'"
char_dict = {char: idx + 1 for idx, char in enumerate(alphabet)}
char_dict[""] = 0
def char_to_index(c):
    c = c.lower()
    if c in char_dict:
        return char_dict[c]
    else:
        return 1

dtype = torch.float32
ALPHABET_SIZE = len(alphabet) + 1 + 64
print(ALPHABET_SIZE)
device = torch.device('cuda')

char_to_tensor = {}
for char in alphabet:
    char_to_tensor[char] = torch.zeros(ALPHABET_SIZE, dtype=dtype, device=device)
    char_to_tensor[char][char_to_index(char)] = 1
upper_alphabet = alphabet.upper()
for char in upper_alphabet:
    char_to_tensor[char] = torch.zeros(ALPHABET_SIZE, dtype=dtype, device=device)
    char_to_tensor[char][char_to_index(char)] = 1
    char_to_tensor[char][0] = 1

def char_to_tensor_without_pe(char):
    if char not in char_to_tensor:
        char = "*"
    tensor = char_to_tensor[char]
    return tensor

pos_embeds = {}
def positional_embedding(pos, d_model):
    import math
    if pos not in pos_embeds:
        pe = torch.zeros(d_model, device=device)
        for i in range(d_model):
            if i % 2 == 0:
                pe[i] = math.sin(pos / 10000 ** (i / d_model))
            else:
                pe[i] = math.cos(pos / 10000 ** ((i - 1) / d_model))
        pos_embeds[pos] = pe
    return pos_embeds[pos]

def tensor_positional_embedding_to_index(tensor):
    sin_vals = tensor[64::2]
    cos_vals = tensor[65::2]
    sin_vals = torch.asin(sin_vals)
    cos_vals = torch.acos(cos_vals)
    pos_l = []
    for k in range(32):
        i = 2 * k
        pos_l.append(sin_vals[k].item() * 10000 ** (i / 64))
        pos_l.append(cos_vals[k].item() * 10000 ** ((i - 1) / 64))
    return pos_l


def string_to_tensor(string):
    tensor = torch.zeros(len(string), ALPHABET_SIZE, dtype=dtype, device=device)
    for idx, char in enumerate(string):
        tensor[idx] = char_to_tensor_without_pe(char)
        pe = positional_embedding(idx, ALPHABET_SIZE - len(alphabet) - 1)
        tensor[idx][len(alphabet) + 1:] += pe
    return tensor

def tensor_to_string(tensor):
    string = ""
    for idx in range(tensor.size()[0]):
        char_idx = torch.argmax(tensor[idx][1:]).item() + 1
        if char_idx > 0:
            char = alphabet[char_idx - 1]
            if char.isalpha() and tensor[idx][0] > 0.5:
                char = char.upper()
            string += char
    return string
import random



def tensor_to_string_stochastic(tensor, temperature=0, topk=5):
    tensor = tensor[:, :ALPHABET_SIZE//2]
    # print(tensor)
    if temperature != 0:
        tensor = tensor / temperature
    else:
        tensor = tensor * 1e6
    # print("logits", tensor)
    probs = torch.softmax(tensor[0, 1:], dim=0)
    # print("probs", probs)
    # input() 
    values, indices = torch.topk(torch.softmax(tensor[0, 1:], dim=0), topk)
    # print(values, indices)
    indices += 1
    values /= values.sum()
    r = random.random()
    # print(values)
    s = 0
    idx = 0
    for i in range(values.size()[0]):
        idx = i
        if r < values[i].item():
            break
        else:
            r -= values[i].item()
    string = ""
    char_idx = indices[idx].item()
    # print(max(probs))
    # print("selected item with", probs[char_idx - 1].item(), "probability")
    if char_idx > 0:
        char = alphabet[char_idx - 1]
        if char.isalpha() and tensor[0][0] > 0 :
            char = char.upper()
        string += char
    return string


def tensor_to_full_string(tensor, temperature=0, topk=5):
    s = ""
    if len(tensor.size()) == 2:
        tensor = tensor.unsqueeze(0)
    for i in range(tensor.size()[1]):
        s += tensor_to_string_stochastic(tensor[0,i].reshape(1,-1), temperature=temperature, topk=topk)
    return s


def sample_rnn(model, start_string, length=100, temperature=1, topk=5):
    model.eval()
    with torch.no_grad():
        input = string_to_tensor(start_string)
        hidden = model.initHidden(1)
        for char in start_string:
            output, hidden = model(string_to_tensor(char), hidden)
        output_string = start_string
        char = tensor_to_string_stochastic(output, temperature=temperature, topk=topk)
        output_string += char
        input = string_to_tensor(char)

        for i in range(length):
            output, hidden = model(input, hidden)
            char = tensor_to_string_stochastic(output, temperature=temperature, topk=topk)
            output_string += char
            input = string_to_tensor(char)
        return output_string
    
def sample_transformer(model, start_string, length=100, temperature=1, topk=5, show=True):
    model.eval()
    with torch.no_grad():
        input = string_to_tensor(start_string)
        output_string = start_string
        if show:
            print(output_string, end="")
        for i in range(length):
            output = model(input.reshape(1,  -1, ALPHABET_SIZE))
            torch.cuda.empty_cache()
            char = tensor_to_string_stochastic(output[:,-1], temperature=temperature, topk=topk)
            print(char, end="") if show else None
            output_string += char
            newchar = char_to_tensor_without_pe(char).clone()
            newchar[len(alphabet) + 1:] += positional_embedding(input.size()[0], ALPHABET_SIZE - len(alphabet) - 1)
            # input = string_to_tensor(output_string)
            input = torch.cat((input, newchar.reshape(1, -1)), dim=0)
        if show:
            print()
        return output_string

def sample_transformer_generate_mode(model, start_string, length=100, temperature=1, topk=5, show=True):
    import transformer
    model.eval()
    with torch.no_grad():
        input = string_to_tensor(start_string)
        output_string = start_string
        model.begin_generate(length + len(start_string), batchsize=1)
        output = model.generate_first(input.reshape(1,  -1, ALPHABET_SIZE))
        char_pred = output[:, -1]
        # print(transformer.inspect(model))
        if show:
            print(output_string, end="")
        for i in range(length):
            if (i + len(start_string)) % 400 == 0:
                print("\n", i + len(start_string))
            torch.cuda.empty_cache()
            char = tensor_to_string_stochastic(char_pred, temperature=temperature, topk=topk)
            print(char, end="") if show else None
            output_string += char
            newchar_tensor = char_to_tensor_without_pe(char).clone()
            newchar_tensor[len(alphabet) + 1:] += positional_embedding(len(start_string) + i, ALPHABET_SIZE - len(alphabet) - 1)
            char_pred = model.generate_next(newchar_tensor.reshape(1, -1, ALPHABET_SIZE))[:,-1, :]
            # print(transformer.inspect(model))
            # print(f"char_pred{char_pred}")
            # input = string_to_tensor(output_string)
            # input = torch.cat((input, newchar_tensor.reshape(1, -1)), dim=0)
        if show:
            print()
        return output_string


class LazyTokenized:
    def __init__(self, string):
        self.string = string
        self.tensor = torch.zeros(len(string), ALPHABET_SIZE, dtype=dtype, device=device)
        self.accessed = torch.zeros(len(string), dtype=torch.bool, device=device)

    def __getitem__(self, idx):
        if type(idx) == slice:
            start, stop, step = idx.indices(len(self.string) * 2)
            if stop > len(self.string):
                start -= stop - len(self.string) +  1
                stop = len(self.string) - 1
            idx = slice(start, stop, step)
            access = self.accessed[idx]
            if not torch.all(access):
                self.tensor[idx] = string_to_tensor(self.string[idx])
                self.accessed[idx] = True
            return self.tensor[idx]
        else:
            raise NotImplementedError()

# s = "here is a ySsample string"
# print(s)
# te = string_to_tensor(s)
# print(te)
# print(te.shape)
# print(tensor_to_string(te))

def get_char_freqs(string):
    string = string.lower()
    cfv = torch.zeros(ALPHABET_SIZE, dtype=dtype)
    cfv[0] = 1
    for char in string:
        cfv[char_dict[char]] += 1
    cfv.to(device)
    return cfv
def main():
    import transformer
    # open model at models/sspear_rnn_1.pt
    import rnn
    # model = rnn.RNN(ALPHABET_SIZE, 220, ALPHABET_SIZE, hidden_layers=130)
    # # model.load_state_dict(torch.load("./models/sspear_rnn_1.pt"))
    model1600 =  transformer.most_recent_model("/home/g/learn/dl/torch/models/sspeare_ztransformer_1784.128-4_512-0.1-6_epoch4.pt")
    model800 = transformer.most_recent_model("/home/g/learn/dl/torch/models/sspeare_ztransformer_1779.128-4_512-0.1-6_epoch49.pt")
    # model = rnn.RNN(ALPHABET_SIZE, 50, ALPHABET_SIZE, hidden_layers=5)
    # model.load_state_dict(torch.load("/home/g/learn/dl/torch/models/sspear_rnn_10.5-50_5-100-100.pt"))
    # # s = sample_rnn(model.to(device), "q", length=100, temperature=0.5, topk=5)
    # get most recent model's path from models/
    import os
    import train.parsedfiles as parsedfiles
    model = model800
    model2 = transformer.most_recent_model("/home/g/learn/dl/torch/models/sspeare_ztransformer_1839*")
    model = model2
    # sample_transformer(model.to(device), sspear_parse.shakespeare[6200:7000], length=800, temperature=1, topk=10)
    # models = os.listdir("./models")
    # models.sort()
    # print(models[-1])
    # parse the model path to get the parameters
    # import re
    # print(models[-1])
    # match = re.match(r"sspeare_rnn_(\d+)\.(\d+)-(\d+)_\d+-\d+-\d+_epoch\d+\.pt", models[-1])
    # print(match.group(2), match.group(3))
    # model = rnn.StandardRNN(ALPHABET_SIZE, int(match.group(3)) - ALPHABET_SIZE, ALPHABET_SIZE, hidden_layers=int(match.group(2)))
    # model.load_state_dict(torch.load("./models/" + models[-1]))
    # # sample from the model
    test_str = "PROMETHIUS."
    sample_transformer_generate_mode(model.to(device), test_str, length=4000, temperature=0, topk=5)
    sample_transformer_generate_mode(model.to(device), input("prompt?:\n"), length=800, temperature=0, topk=10)
    # f = open("./textsamples/" + models[-1] + ".txt", "w")
    # f.write(s) 
    # f.close()
    # print(s)
    # model.eval()
    # with torch.no_grad():
    #     y=model(string_to_tensor("This is the 100th Etext file presented").reshape(1,-1, ALPHABET_SIZE))
    # s = ""
    # for i in range(y.size()[1]):
    #     s += tensor_to_string_stochastic(y[0, i].reshape(1,-1), temperature=0, topk=5)
    # print("100th", s)
    # print("100th", tensor_to_full_string(y, temperature=0.5, topk=5))
    # # print(models[-1])
    # t = string_to_tensor("This is the 100th Etext file presented")
    # print(t.shape)
    # print(tensor_positional_embedding_to_index(t[4]))

if __name__ == "__main__":
    main()

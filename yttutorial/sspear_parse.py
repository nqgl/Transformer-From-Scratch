shakespeare = open('/home/g/learn/dl/torch/yttutorial/t8.shakespeare.txt', 'r').read()
startpoints = [i for i in range(len(shakespeare)) if shakespeare.startswith('\n\n', i)]
s = startpoints[504]
print(len(shakespeare))
snippets500 = [shakespeare[s:s+100] for s in startpoints]
# snippets = [shakespeare[s:s+400] for s in startpoints]
snippets = [shakespeare[startpoints[i]:startpoints[i+1]] for i in range(len(startpoints)-1)]
# snippets += [shakespeare[startpoints[i]:startpoints[i+3]] for i in range(len(startpoints)-3)]
# snippets += [shakespeare[startpoints[i]:startpoints[i+9]] for i in range(len(startpoints)-9)]

splengths = [startpoints[i+1]-startpoints[i] for i in range(len(startpoints)-1)]

import tokenizer
import torch
import time
cuda0 = torch.device('cuda:0')

sspeare_tensor = tokenizer.LazyTokenized(shakespeare)
# snippets = [tokenizer.string_to_tensor(s) for s in snippets]
# snippets500 = [tokenizer.string_to_tensor(s) for s in snippets500]



# nextwords500 = [torch.zeros(len(snippets500), tokenizer.ALPHABET_SIZE) for i in range(len(snippets500))]
# for i in range(500):
#     for j in range(len(snippets500)):
#         nextwords500[i][j] += snippets500[j][i]
# that's just the same as next 
import random
def nexwords500batched(batchsize, runlength=500):
    # shuffle snippets
    random.shuffle(startpoints)
    for b in range(len(snippets500)//batchsize):
        snippet_starts = startpoints[b*batchsize:(b+1)*batchsize]
        snippets500batch = [sspeare_tensor[s: s+runlength] for s in snippet_starts]
        nextwords500batch = torch.zeros(runlength, batchsize, tokenizer.ALPHABET_SIZE, dtype=torch.float16)
        einsumsolution = torch.einsum('ijk->jik', torch.stack(snippets500batch))
        yield einsumsolution


#batches = nexwords500batched(10)
# print(len(nextwords500))    

char_freqs = torch.tensor([1.0000e+00, 2.8915e+05, 6.1956e+04, 8.8185e+04, 1.4946e+05, 4.4720e+05,
        8.0516e+04, 6.8199e+04, 2.3687e+05, 2.5399e+05, 4.7790e+03, 3.5408e+04,
        1.7002e+05, 1.1145e+05, 2.4326e+05, 3.1460e+05, 5.8464e+04, 3.5820e+03,
        2.3786e+05, 2.4899e+05, 3.2978e+05, 1.2895e+05, 3.7569e+04, 8.9390e+04,
        5.2940e+03, 9.4370e+04, 1.6310e+03, 2.9900e+02, 9.2800e+02, 3.6600e+02,
        3.3000e+02, 9.3000e+01, 8.2000e+01, 6.3000e+01, 4.1000e+01, 4.0000e+01,
        9.4800e+02, 1.2939e+06, 0.0000e+00, 8.3174e+04, 1.7199e+04, 7.8025e+04,
        8.8440e+03, 1.0476e+04, 1.8270e+03, 0.0000e+00, 4.7000e+02, 5.0000e+00,
        0.0000e+00, 3.3000e+01, 7.1000e+01, 8.0000e+00, 1.0000e+00, 0.0000e+00,
        1.0000e+00, 0.0000e+00, 2.1000e+01, 6.3000e+01, 1.0000e+00, 1.0000e+00,
        0.0000e+00, 8.0740e+03, 1.0000e+00, 4.6800e+02, 4.4100e+02, 6.2800e+02,
        6.2900e+02, 2.0850e+03, 2.0770e+03, 0.0000e+00, 2.0000e+00, 1.2446e+05,
        0.0000e+00, 3.1069e+04]).to(cuda0)

if __name__ == "__main__":
    import tokenizer
    print(tokenizer.get_char_freqs(shakespeare))
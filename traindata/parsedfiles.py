shakespeare = open('traindata/texts/t8.shakespeare.txt', 'r').read()
moby_dick = open("traindata/texts/hm/moby10b.txt", "r").read()
startpoints = [i for i in range(len(shakespeare)) if shakespeare.startswith('\n\n', i)]
s = startpoints[504]
print(len(shakespeare))
snippets500 = [shakespeare[s:s+100] for s in startpoints]
# snippets = [shakespeare[s:s+400] for s in startpoints]
snippets = [shakespeare[startpoints[i]:startpoints[i+1]] for i in range(len(startpoints)-1)]
# snippets += [shakespeare[startpoints[i]:startpoints[i+3]] for i in range(len(startpoints)-3)]
# snippets += [shakespeare[startpoints[i]:startpoints[i+9]] for i in range(len(startpoints)-9)]

splengths = [startpoints[i+1]-startpoints[i] for i in range(len(startpoints)-1)]

import traindata.tokenizer as tokenizer
import torch
import time
print(len(moby_dick))
sspeare_tensor = tokenizer.LazyTokenized(shakespeare)
moby_tensor = tokenizer.LazyTokenized(moby_dick)
# snippets = [tokenizer.string_to_tensor(s) for s in snippets]
# snippets500 = [tokenizer.string_to_tensor(s) for s in snippets500]




melville_books_list = [
        open("traindata/texts/hm/moby10b.txt", "r").read(),
        open("traindata/texts/hm/8118-0.txt").read(),
        open("traindata/texts/hm/pg11231.txt").read(),
        open("traindata/texts/hm/pg34970.txt").read(),
        open("traindata/texts/hm/pg12384.txt").read(),
        open("traindata/texts/hm/piazza.txt").read(),
        open("traindata/texts/hm/pg10712.txt").read(),
        open("traindata/texts/hm/pg21816.txt").read(),
        open("traindata/texts/hm/typee.txt").read()]
melville_booktensors_list = [tokenizer.LazyTokenized(x) for x in melville_books_list]
print([len(b) for b in melville_books_list])
hhgttg = open("traindata/texts/hhgttg.txt").read()
hhgttg_tensor = tokenizer.LazyTokenized(hhgttg)
wiki = open("traindata/texts/wiki/train.jsonl").read()
wiki_tensor = tokenizer.LazyTokenized(wiki)
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


def main():
    print(len(wiki) / (6 * 500 * 2 * 4200))
if __name__ == "__main__":
    main()
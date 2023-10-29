import transformer
import torch
from traindata import parsedfiles, tokenizer
import torch.nn.functional as F

def get_get_tokens(dataset):
    def get_tokens(length=1000, batchsize=1, batches=1, epoch=0, skew=0):
        skew = int(skew * length)
        if batches is None:
            batches = len(dataset)//(length*batchsize)
        else:
            skew += int(epoch * length * batchsize * batches)
        for j in range(batches):
            
            l = []
            for i in range(batchsize):
                l.append(dataset[batchsize*length*j + i*length + skew:batchsize*length*j + (i+1)*length + skew])
            yield torch.stack(l).cuda()
    return get_tokens

def criterion(y_pred, y_i):
    # caps_pred = F.sigmoid(y_pred[:, :1])
    # print(y_pred.shape, y_i.shape)
    capsloss = F.binary_cross_entropy_with_logits(y_pred[:, :, :1], y_i[:, :, :1]) 
    y_pred_flat = y_pred[:,:,1:64].reshape(-1, 63)
    y_i_flat = y_i[:,:,1:64].reshape(-1, 63)         #clear 

    # print(y_pred_flat.shape, y_i_flat.shape)
    # print(torch.sum(y_i_flat, dim=1))
    charloss = F.cross_entropy(y_pred_flat, y_i_flat, label_smoothing=0.01)
    # charloss1 = 0
    # for i in range(y_pred.shape[0]):
    #     for t in range(y_pred.shape[1]):
    #         charloss1 += F.cross_entropy(y_pred[i, t, 1:64], y_i[i, t, 1:64])
    # charloss2 = 0
    # for i in range(y_pred_flat.shape[0]):
    #     charloss2 += F.cross_entropy(y_pred_flat[i], y_i_flat[i])
    # charloss3 = F.cross_entropy(y_pred_flat, y_i_flat, reduction="sum", label_smoothing=0.01)

    # print("losses:", charloss3, capsloss)
    # print(charloss1, charloss2, charloss, charloss3)
    return charloss + capsloss/64

def train(model, epochs = 10, batchsize = 7, runlength = 1000, get_tokens=get_get_tokens(parsedfiles.sspeare_tensor), batches = None, lr=0.015):
    model.cuda()
    n_b = 0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    for epoch in range(epochs):
        model.train()
        #adjust learning rate
        if epoch > 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.995
        # decay learning rate   
        running_loss = 0.0
        s = ""
        for batch in get_tokens(length=runlength, batchsize=batchsize, batches=batches, skew=(1 + epoch)/epochs, epoch=epoch):
            # print(batch. shape)
            n_b += 1
            batch.cuda()
            x = batch[:, :-1]
            y = batch[:, 1:]
            optimizer.zero_grad()
            y_pred = model(x)
            # print(y_pred)
            loss = criterion(y_pred, y)
            print(f"Batch{n_b} loss: {loss.item()}")
            p = "ACTUAL\n" + tokenizer.tensor_to_full_string(y[0,:,:]) + "\nPRED\n" + tokenizer.tensor_to_full_string(y_pred[0,:,:])
            s += p if n_b % 50 == 0 else ""
            print(p)
            loss.backward()
            optimizer.step()
            # torch.cuda.empty_cache()
            running_loss += loss.item()
            # s =  tokenizer.sample_transformer(model, "The q", length=10, temperature=1, topk=10 )
            # model.train()
            # print(s)
        print(f"Epoch loss: {running_loss}")
        params = (128,4, 512, 0.1, 6)
        import os
        os.makedirs("./models", exist_ok=True)
        k = len(os.listdir("./models"))
        k = "0" * (4 - len(str(k))) + str(k)
        torch.save(model.state_dict(), "./models/sspeare_ztransformer_{}.{}-{}_{}-{}-{}_epoch{}.pt".format(k, *params, epoch))

     #   s +=  tokenizer.sample_transformer(model, "Th", length=100, temperature=0.3, topk=10 )
     #   print(s)
      #  s +=  tokenizer.sample_transformer(model, "This is", length=100, temperature=0.3, topk=10)
       # print(s)
    #    s +=  tokenizer.sample_transformer(model, "This is the 100th Etext file presented by Project Gutenberg", length=100, temperature=0.3, topk=10)
     #   print(s)
    #    s +=  tokenizer.sample_transformer(model, "A", length=100, temperature=0.3, topk=10)
        f = open("./textsamples/tx.txt", "a")
        f.write(s)
        f.close()
        torch.cuda.empty_cache()
    return model


def main():
    model = transformer.DecoderOnlyTransformer(128,4, 512, 0.1, 6)
    model = transformer.most_recent_model("./models/sspeare_ztransformer_227*.pt")
    batchsize = 2   
    length = 4200
    # for book in parsedfiles.melville_booktensors_list:
    #     model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(book))
    # model = train(model, 50, batchsize, length, get_tokens=get_get_tokens(parsedfiles.moby_tensor))
    # model = train(model, 50, batchsize, length, get_tokens=get_get_tokens(parsedfiles.hhgttg_tensor))
    # model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(parsedfiles.sspeare_tensor))
    # for i in range(5):
    #     for book in parsedfiles.melville_booktensors_list:
    #         model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(book), lr=0.007)
    #     model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(parsedfiles.moby_tensor) , lr=0.007)
    #     model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(parsedfiles.hhgttg_tensor)   , lr=0.007)
    #     model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(parsedfiles.sspeare_tensor)  , lr=0.007)
    # for book in parsedfiles.melville_booktensors_list:
    #     model = train(model, 10, batchsize, length, get_tokens=get_get_tokens(book), lr=0.007)
    # model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(parsedfiles.sspeare_tensor)  , lr=0.007)
    # model = train(model, 100, batchsize, length, get_tokens=get_get_tokens(parsedfiles.moby_tensor) , lr=0.007)
    model = train(model, 10, batchsize, length, get_tokens=get_get_tokens(parsedfiles.hhgttg_tensor) , lr=0.012)
    model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(parsedfiles.sspeare_tensor) , lr=0.012)
    model = train(model, 1, batchsize, length, get_tokens=get_get_tokens(parsedfiles.moby_tensor) , lr=0.012)
    model = train(model, 30, 1, 6000, get_tokens=get_get_tokens(parsedfiles.hhgttg_tensor) , lr=0.008)
    model = train(model, 2, 1, 6000, get_tokens=get_get_tokens(parsedfiles.sspeare_tensor) , lr=0.008)
    model = train(model, 10, batchsize, length, get_tokens=get_get_tokens(parsedfiles.hhgttg_tensor) , lr=0.012)
    model = train(model, 500, batchsize, length, get_tokens=get_get_tokens(parsedfiles.wiki_tensor) , lr=0.012, batches=500)
    s =  tokenizer.sample_transformer(model, "The q", length=100, temperature=0, topk=10 )
    print(s)
    # model = train(model, 5, 5, 1600  )
    s =  tokenizer.sample_transformer(model, "The q", length=100, temperature=0.6, topk=10 )
    print(s) 
    # model = train(model, 100, 16, 100 )
    for i in range(120):
        s =  tokenizer.sample_transformer(model, "The q", length=100, temperature=0.3, topk=10 )
        print(s)
        s =  tokenizer.sample_transformer(model, "This is", length=100, temperature=0.3, topk=10)
        print(s)    
        s =  tokenizer.sample_transformer(model, "This is the 100th Etext file presented by Project Gutenberg", length=100, temperature=0.3, topk=10)
        print(s)
        s =  tokenizer.sample_transformer(model, "A", length=100, temperature=0.3, topk=10)
if __name__ == "__main__":
    main()
    
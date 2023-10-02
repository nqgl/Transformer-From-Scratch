from rnn import RNN, StandardRNN
import torch
import torchvision
import torchvision.transforms as transforms
import tokenizer
device = torch.device('cuda')
import os
import sspear_parse
import torch.nn as nn
import time
import torch.nn.functional as F
# torch.autograd.set_detect_anomaly(True)

def view_norms_data(model, hiddens, losses, loss_per_step):
    pgrads =  [p.grad.norm(2).cpu().detach().numpy() for p in model.parameters()]
    # print("Parameter grad norms: {}".format(pgrads))
    # print("MLP grad norms: {}".format([w.. for w in model.W.mlp]))
    # print("Hidden grad norms: {}".format(hgrads))
    hgrads = [h.grad.norm(1).cpu().detach().numpy() for h in hiddens if h.grad is not None]
    # print("Hidden norms: {}".format([h.norm(2) for h in hiddens]))
    import matplotlib.pyplot as plt
    
    layer_norms = [layer.weight.norm(2, dim=1).cpu().detach().numpy() for layer in model.W.layers if type(layer) == nn.Linear]

    plt.subplot(3, 2, 1)
    plt.title("Hidden norms")
    plt.plot([(h.norm(1, dim=-1).norm(1)/h.shape[0]).cpu().detach().numpy() for h in hiddens])
    plt.subplot(3, 2, 2)
    plt.title("Hidden grad norms")
    plt.plot(hgrads)
    plt.subplot(3, 2, 5)    
    plt.title("Parameter grad norms")
    # plot parameter grad norms
    paramnames = [n for n, p in model.named_parameters()]
    params = [p for n, p in model.named_parameters()]
    pgrads = [p.grad.norm(2).cpu().detach().numpy() for p in params]
    plt.bar(paramnames, pgrads)
    plt.subplot(3, 2, 3)
    plt.title("Layer avg norms")
    # plot layer norms
    plt.plot([layer_norm.mean() for layer_norm in layer_norms])
    plt.subplot(3, 2, 4)
    plt.title("Losses per step")
    plt.plot([loss.cpu().detach().numpy() for loss in loss_per_step])

    # plt.title("Layer grad norms")
    # # plot layer grad norms
    # plt.plot([layer.weight.grad.norm(2, dim=1).mean().cpu().detach().numpy() for layer in model.W.layers if type(layer) == nn.Linear])
    plt.subplot(3, 2, 6)
    plt.title("Losses")
    losses[1:1] = [losses[0] - losses[1]]
    plt.bar(["All", "Total", "Penalty"], [l.cpu().detach().numpy() for l in losses])
    plt.pause(0.001)
    plt.clf()   

    plt.draw()

from typing import List
def push_to_1_penalty(model: RNN, hiddens: List[torch.Tensor]):
    hidden_dist_from_1_penalty = [
        torch.abs(
            h_i.norm(1, dim=-1).norm(1)/h_i.shape[0] - 
            1/(h_i.norm(1, dim=-1).norm(1)/h_i.shape[0]
               )
            ) ** 1 for h_i  in hiddens]
    linear_layers = [layer.weight for layer in model.W.layers]
    # rowsnorm = [torch.abs(layer.norm(2, dim=1) - 1) for layer in linear_layers]
    rowsnorm = []
    penalty = sum(hidden_dist_from_1_penalty) * 0.0001 + sum([0] + [rownorm.norm(1) for rownorm in rowsnorm]) * 0.000
    return penalty 

def criterion(y_pred, y_i):
    # caps_pred = F.sigmoid(y_pred[:, :1])  
    capsloss = F.binary_cross_entropy_with_logits(y_pred[:, :1], y_i[:, :1])
    charloss = F.cross_entropy(y_pred, y_i)
    # print("losses:", charloss, capsloss)
    return charloss + capsloss

def train(model, epochs = 10, batchsize = 40, runlength = 500):
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0, dampening=0.5)
    for epoch in range(epochs):
        #adjust learning rate
        if epoch > 3:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.995
        # decay learning rate   
        running_loss = 0.0
        t0 = time.time()
        for batchnextwords in sspear_parse.nexwords500batched(batchsize, runlength=runlength):
            for _ in range(1):
                optimizer.zero_grad()
                h_i = model.initHidden(batchsize=batchsize)
                losses = []
                hiddens = [h_i]
                total_loss = 0
                for i in range(len(batchnextwords) - 1):
                    x_i =  batchnextwords[i].to(device)
                    y_i =  batchnextwords[i+1].to(device)            #clear gradients
 
                    y_pred, h_i = model(x_i, h_i)
                    h_i = h_i
                    hiddens.append(h_i)
                    #calculate loss
                    loss = criterion(y_pred, y_i)
                    losses.append(loss)
                    total_loss += loss
                #backward pass
                penalty = 0
                penalty = push_to_1_penalty(model, hiddens)
                total_loss += penalty
                # for p in model.parameters():
                #     penalty += p.norm(2) + p.norm(1)
                # total_loss += penalty * 0.000005
                #update parameters
                [h.retain_grad() for h in hiddens]

                total_loss.backward()
                pgrads =  [p.grad.norm(2) for p in model.parameters()]
                hgrads = [h.grad.norm(2) for h in hiddens if h.grad is not None]
                # print("Parameter grad norms: {}".format(pgrads))
                # # print("MLP grad norms: {}".format([w.. for w in model.W.mlp]))
                # print("Hidden grad norms: {}".format(hgrads))
                # print("Hidden norms: {}".format([h.norm(2) for h in hiddens]))
                # print(model.h_0.grad)
                view_norms_data(model, hiddens, [total_loss, penalty], losses)
                optimizer.step()
                s = tokenizer.sample_rnn(model, "This is", length=100, temperature=0.5, topk=5)
                print(s)
                s = tokenizer.sample_rnn(model, "There q", length=100, temperature=0.5, topk=5)
                print(s)

                running_loss += total_loss.item()
                print("Epoch {} - Batch loss: {}".format(epoch,total_loss.item()/(runlength - 1)))
        print("Epoch {} - Training loss: {}  - Time elapsed:    {}".format(epoch, running_loss/(runlength - 1), time.time() - t0))
        import os
        os.makedirs("./models", exist_ok=True)
        k = len(os.listdir("./models"))
        k = "0" * (4 - len(str(k))) + str(k)
        torch.save(model.state_dict(), "./models/sspeare_rnn_{}.{}-{}_{}-{}-{}_epoch{}.pt".format(k, model.depth, model.width, epochs, batchsize, runlength, epoch))

    print("Finished training")
    import os
    os.makedirs("./models", exist_ok=True)
    k = len(os.listdir("./models"))
    k = "0" * (4 - len(str(k))) + str(k)
    torch.save(model.state_dict(), "./models/sspeare_rnn_{}.{}-{}_{}-{}-{}.pt".format(k, model.depth, model.width, epochs, batchsize, runlength))
    s = tokenizer.sample_rnn(model, "T", length=100, temperature=0.5, topk=5)
    model = model.to(torch.device('cpu'))
    torch.cuda.empty_cache()
    return model


if __name__ == "__main__":
    import rnn
    width = 200
    depth = 2
    epochs = 100
    batchsize = 100
    runlength = 600 # 500
    print("width: {}, depth: {}, epochs: {}, batchsize: {}, runlength: {}".format(width, depth, epochs, batchsize, runlength))
    model = RNN(tokenizer.ALPHABET_SIZE, width, tokenizer.ALPHABET_SIZE, hidden_layers=depth)
    # model = rnn.most_recent_model()
    import cProfile
    # cProfile.run("train(model, epochs=epochs, batchsize=batchsize, runlength=runlength)")
    #model.load_state_dict(torch.load("./models/sspeare_rnn_0247.16-1200_100-400-75_epoch5.pt"))
    model = train(model, epochs=epochs, batchsize=batchsize, runlength=runlength)
    model = train(model, epochs=500, batchsize=500, runlength=32)
    model = train(model, epochs=30, batchsize=200, runlength=80)
    model = train(model, epochs=40, batchsize=80, runlength=200)

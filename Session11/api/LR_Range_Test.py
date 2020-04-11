import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt


import copy

Lrtest_train_acc = []
LRtest_Lr = []
def LR_test(max_lr, min_lr,device,epoch,model,criterion,trainloader,momemtum = 0.9,weight_decay=0.05, plot= True ):
    step = (max_lr - min_lr )/epoch
    lr = min_lr
    for e in range(epoch):
        testmodel = copy.deepcopy(model)
        optimizer = optim.SGD(testmodel.parameters(), lr=lr ,momentum=momemtum,weight_decay=weight_decay ) 
        lr += (max_lr - min_lr)/epoch
        testmodel.train()
        pbar = tqdm(trainloader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred =testmodel(data)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(desc= f'epoch = {e+1} Lr = {optimizer.param_groups[0]["lr"]}  Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        Lrtest_train_acc.append(100*correct/processed)
        LRtest_Lr.append(optimizer.param_groups[0]['lr'])

    if(plot):
        plt.plot(LRtest_Lr, Lrtest_train_acc)
        plt.ylabel('train Accuracy')
        plt.xlabel("Learning rate")
        plt.title("Lr v/s accuracy")
        plt.show()



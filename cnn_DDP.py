#!/usr/bin/env python
# coding: utf-8

# ### Classifier tutorial
# March 15, 2021

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# from torchsummary import summary
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import time
from datetime import datetime
import glob
import pickle
import yaml
import logging


def try_barrier(rank):
    """Attempt a barrier but ignore any exceptions"""
    print('BAR %d'%rank)

    try:
        dist.barrier()
    except:
        pass
    

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.to(device)


def f_train_model(net,train_loader,num_epochs):
    
    device = torch.cuda.current_device()
    print(device)
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            print("i",i,device)
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device),data[1].to(device)
            inputs=torch.reshape(inputs,(batch_size,3,32,32))
            labels=labels.long()
#             optimizer.zero_grad()
            print("before zero grad",i)
            # forward + backward + optimize
            net.zero_grad()
            net.train()
            print("before network call",local_rank)
            outputs = net(inputs)
            print("before loss call")
            loss = criterion(outputs, labels)
            loss.backward()
            print("before optimizer call",local_rank)
            optimizer.step()
            print("after optimizer call",local_rank,i)
            
            # print statistics
#             running_loss += loss.item()
#             print("before print",local_rank,i)
#             if i % 2000 == 1999:    # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#             net.train()

    print('Finished Training')
    
## Test model
def f_test(data_loader,net):
    net.eval()
    
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data[0].to(device),data[1].to(device)
            inputs=torch.reshape(inputs,(batch_size,3,32,32))
            labels=labels.long()
            
            outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print('Accuracy of the network on the 10000 test images: %d %%\n' % (100 * correct / total))
    
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

if __name__=="__main__":
    
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size","-b", default=64, type=int)
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--local_rank", default=0, type=int)
    args=parser.parse_args()

    batch_size=args.batch_size
    epochs=args.epochs
#     batch_size=128
#     epochs=10

    ### Distributed data-parallel setup
    local_rank=args.local_rank    
    
    verb=False
    if local_rank==0: verb=True
    
    dist.init_process_group(backend='nccl', init_method="env://")  
    
    world_rank = dist.get_rank() 
    print("Local rank",local_rank)
    print("World rank",world_rank)
    
    world_size, rank = torch.cuda.device_count(), torch.cuda.current_device()

#     raise SystemExit

    #################
    ### Extract data
    data_dir='data/cifar'
    train_x=np.load(data_dir+'/train_x.npy')
    train_y=np.load(data_dir+'/train_y.npy')

    test_x=np.load(data_dir+'/test_x.npy')
    test_y=np.load(data_dir+'/test_y.npy')

    # size=train_x.shape[0]
    # val_size=int(size/100)

    size=50000
    val_size=5000

    dataset=TensorDataset(torch.Tensor(train_x[:(size-val_size)]),torch.Tensor(train_y[:(size-val_size)]))
    train_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)

#     train_sampler=torch.utils.data.distributed.DistributedSampler(dataset,num_replicas=world_size,rank=world_rank)
#     train_loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=1,drop_last=True,sampler=train_sampler)

    dataset=TensorDataset(torch.Tensor(train_x[-val_size:]),torch.Tensor(train_y[-val_size:]))
    val_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1,drop_last=True)

    dataset=TensorDataset(torch.Tensor(test_x),torch.Tensor(test_y))
    test_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=1,drop_last=True)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    #################
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device=torch.cuda.current_device()
    if verb:
        print(device)
        print("Using ",torch.cuda.device_count(), "GPUs")
    
    # Build network
    net = Net().to(device)
    
#     if device=='cuda':
#         torch.cuda.set_device(local_rank)
#         model.cuda(local_rank)
    
#     dist.barrier()
    criterion = nn.CrossEntropyLoss().to(device)
    print("After criterion def",local_rank)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    try_barrier(local_rank)
    print("Before ddp call",local_rank)
    net=nn.parallel.DistributedDataParallel(net,device_ids=[local_rank],output_device=[local_rank])
    print("After ddp call",local_rank)

    try_barrier(local_rank)
    
    t1=time.time()
    f_train_model(net,train_loader,epochs)
    t2=time.time()
    print("Training time",t2-t1)
    
    if local_rank==0: 
        f_test(test_loader,net)
    

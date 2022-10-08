import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
from torch.backends import cudnn
import torchvision.transforms as transforms
import matplotlib.pylab as plt
from torch.utils.data import DataLoader
from alexnet import Alexnet
from dataset_class import Dataset

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

def train_model(model,train_loader,validation_loader,optimizer,n_epochs=4,gpu=False): 
    accuracy_list=[]
    loss_list=[]
    N_test=0
    for epoch in range(n_epochs):
        for x, y in train_loader:
            if gpu:
                # Transfer to GPU
                x, y = x.to(device), y.to(device)
            
            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data)

        correct=0
        N_test=0
        for x_test, y_test in validation_loader:
            if gpu:
                # Transfer to GPU
                x_test, y_test = x_test.to(device), y_test.to(device)
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
            N_test = N_test+1
        accuracy = correct / N_test
        accuracy_list.append(accuracy)

    return accuracy_list, loss_list


data_loc = 'QML' #validation data directory
data_test_loc = 'QML' #test data directory


training_set = Dataset(data_loc)

#Or can use torchvision dataset
#training_set = torchvision.datasets.CIFAR100('.', train=True, download=True)

training_generator = DataLoader(training_set, batch_size = 128)

validation_set = Dataset(data_test_loc)
validation_generator = DataLoader(validation_set, batch_size = 128)


model = Alexnet()

#Data Parallelism, if multiple gpu available
if torch.cuda.device_count() > 1:
	model = nn.DataParallel(model)

#use gpu, if available
if use_cuda:
	model.to(device)


#Loss and hyperparameters set as per the authors of Alexnet
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0005
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)

accuracy_list, loss_list = train_model(model=model,n_epochs=10,train_loader=training_generator,validation_loader=validation_generator,optimizer=optimizer,gpu=True)

plt.plot(np.arange(len(accuracy_list)),accuracy_list)
plt.show()

plt.plot(np.arange(len(loss_list)),loss_list)
plt.show()

'''torch.save(model, PATH)'''
#to save trained model

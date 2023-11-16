import sys
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import datetime
device = "cuda" if torch.cuda.is_available() else "cpu"

def read_mnist(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show_mnist(file_name,mode):
    
    data_set = read_mnist(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
                   
def read_insurability(file_name):
    
    count = 0
    data = []
    with open(file_name,'rt') as f:
        for line in f:
            if count > 0:
                line = line.replace('\n','')
                tokens = line.split(',')
                if len(line) > 10:
                    x1 = float(tokens[0])
                    x2 = float(tokens[1])
                    x3 = float(tokens[2])
                    if tokens[3] == 'Good':
                        cls = 0
                    elif tokens[3] == 'Neutral':
                        cls = 1
                    else:
                        cls = 2
                    data.append([[cls],[x1,x2,x3]])
            count = count + 1
    return(data)


##OUR CODE BELOW##
class CustomInsurDataSet(torch.utils.data.Dataset):
    def __init__(self, file, scaler):
        self.df = read_insurability(file)
        self.sc = scaler

        # Extract features and labels from self.df
        features = [d[1] for d in self.df]
        labels = [d[0] for d in self.df]

        # Scale the features
        self.scaled_features = self.sc.fit_transform(features)

        # Convert labels to the correct format
        self.labels = [label[0] for label in labels] 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        scaled_features = self.scaled_features[idx]
        label = self.labels[idx]

        # Convert to tensors
        data_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return data_tensor, label_tensor

  
class CustomMNISTDataSet(torch.utils.data.Dataset):
    def __init__(self, file, scaler):
        self.df = read_mnist(file)
        self.sc = scaler

        # Extract features and labels from self.df
        features = [d[1] for d in self.df]
        labels = [d[0] for d in self.df]

        # Scale the features
        self.scaled_features = self.sc.fit_transform(features)

        # Convert labels to the correct format
        self.labels = [label[0] for label in labels] 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        scaled_features = self.scaled_features[idx]
        label = self.labels[idx]

        # Convert to tensors
        data_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return data_tensor, label_tensor


#class for the ff nn
class FeedForward(nn.Module):
  def __init__(self):
    super(FeedForward, self).__init__()
    self.linear1 = nn.Linear(3, 32)
    self.relu1 = nn.LeakyReLU()
    self.linear2 = nn.Linear(32, 16)
    self.relu2 = nn.LeakyReLU()
    self.linear_out = nn.Linear(16, 1)

  def forward(self, x):
    x = self.linear1(x)
    x = self.relu1(x)
    x = self.linear2(x)
    x = self.relu2(x)
    x = self.linear_out(x)
    return x
  
def train(dataloader, model, loss_func, optimizer, lamb, ff):
  model.train()
  train_loss = []

  now = datetime.datetime.now()
  for batch, (X, y) in enumerate(dataloader):
    # ignore the first time we see this
    # second time why is gpu better than cpu for this?
    X, y = X.to(device), y.to(device)

    # make some predictions and get the error
    pred = model(X)

    R1 = ff.linear1.weight
    R1 = torch.mul(R1,R1)
    R1 = torch.sum(R1)
    R1 = torch.sqrt(R1)

    R2 = ff.linear2.weight
    R2 = torch.mul(R2,R2)
    R2 = torch.sum(R2)
    R2 = torch.sqrt(R2)

    R3 = ff.linear_out.weight
    R3 = torch.mul(R3,R3)
    R3 = torch.sum(R3)
    R3 = torch.sqrt(R3)

    R = R1 + R2 + R3

    loss = loss_func(pred, y.unsqueeze(1)) + lamb * R

    # where the magic happens
    # backpropogation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch % 10 == 0:
      loss, current = loss.item(), batch * len(X)
      iters = 10 * len(X)
      then = datetime.datetime.now()
      iters /= (then - now).total_seconds()
      print(f"loss: {loss:>6f} |w| {R1:>6f} [{current:>5d}/{17000}] ({iters:.1f} its/sec)")
      now = then
      train_loss.append(loss)
  return train_loss
               
def test(dataloader, model, loss_func):
  size = len(dataloader)
  num_batches = 0
  model.eval()
  test_loss = 0

  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_func(pred, y.unsqueeze(1)).item()
      num_batches = num_batches + 1
  test_loss /= num_batches
  print(f"Avg Loss: {test_loss:>8f}\n")
  return test_loss

def classify_insurability():
    # insert code to train simple FFNN and produce evaluation metrics
    #same arch as slides 12-8, use SGD optimizer(), implement softmax() ourselves
    # train one obsv at a time, exp with diff hypparams 
    #output learning curves, final test res, why this is a bad idea, why hypparm and impact

    sc = MinMaxScaler()
    train_data = CustomInsurDataSet('three_train.csv', sc)
    test_data = CustomInsurDataSet('three_test.csv', sc)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    ff = FeedForward()
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(ff.parameters(), lr=1e-3)
    epochs = 10
    train_loss = []
    test_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n------------------------------- \n")
        losses = train(train_loader, ff, loss_func, optimizer, 0.01, ff)
        train_loss.append(losses)
        test_loss.append(test(test_loader, ff, loss_func))

    # Could add a condition that interrupts training when the loss doesn't change much
    print('Done!')


    
def classify_mnist():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv','pixels')
    
    # insert code to train a neural network with an architecture of your choice
    # (a FFNN is fine) and produce evaluation metrics
    #use same data trans as hw 2
    
def classify_mnist_reg():
    
    train = read_mnist('mnist_train.csv')
    valid = read_mnist('mnist_valid.csv')
    test = read_mnist('mnist_test.csv')
    show_mnist('mnist_test.csv','pixels')
    
    # add a regularizer of your choice to classify_mnist()
    
def classify_insurability_manual():
    
    train = read_insurability('three_train.csv')
    valid = read_insurability('three_valid.csv')
    test = read_insurability('three_test.csv')
    
    # reimplement classify_insurability() without using a PyTorch optimizer.
    # this part may be simpler without using a class for the FFNN
    
    
def main():
    classify_insurability()
    classify_mnist()
    classify_mnist_reg()
    classify_insurability_manual()
    
if __name__ == "__main__":
    main()

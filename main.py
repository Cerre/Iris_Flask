import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from visdom import Visdom
import utils


labels = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(4,32)
        self.linear2 = nn.Linear(32, 10)
        self.linear3 = nn.Linear(10, 3)

    def forward(self, X):
        X = self.linear1(X)
        X = F.relu(X)
        X = self.linear2(X)
        X = F.relu(X)
        X = self.linear3(X)
        return F.log_softmax(X, dim= 1)

        

def fit(model, train_loader, EPOCHS, BATCH_SIZE):


    optimizer = torch.optim.Adam(model.parameters())
    error = nn.CrossEntropyLoss()
    model.train()
    # lr_lambda = lambda epoch: 0.99*epoch 
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    total_loss = []
    total_correct = []

    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1] 
            correct += (predicted == var_y_batch).sum()
            total_correct.append(correct)
            if batch_idx % 50 == 0:
                total_loss.append(loss.item())
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx*len(X_batch), len(train_loader.dataset), 100.*batch_idx / len(train_loader), 
                    loss.item(), float(correct*100) / float(BATCH_SIZE*(batch_idx+1))))
        # scheduler.step()
    # plt.plot(range(1,len(total_loss) + 1),total_loss)
    # plt.plot(range(1,len(total_correct) + 1), total_correct)
    # plt.show()
        



def evaluate(model, test_loader, BATCH_SIZE):
#model = mlp
    correct = 0 
    for test_imgs, test_labels in test_loader:
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)
        predicted = torch.max(output,1)[1]
        
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f}% ".format( float(correct) / (len(test_loader)*BATCH_SIZE)*100))




def read_data():
    iris = pd.read_csv('data/iris.csv')
    # 2 Define X and y
    X = iris.drop('Species', axis=1)
    X = X.drop('Id', axis = 1).values
    y = iris['Species']
    y_numbers = []

    # 3 Binarize the labels
    encoder = LabelBinarizer()
    y = encoder.fit_transform(y)


    for vec in y:
        if vec[0] == 1:
            y_numbers.append(0)
        elif vec[1] == 1:
            y_numbers.append(1)
        else:
            y_numbers.append(2)


    y= np.array(y_numbers)


    # Scale the data
    scaler = MinMaxScaler()
    scaled_X = scaler.fit_transform(X)
    return scaled_X, y


def process_data(X, y, BATCH_SIZE):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)


    torch_X_train = torch.from_numpy(X_train)
    torch_y_train = torch.from_numpy(y_train)
    # create feature and targets tensor for test set.
    torch_X_test = torch.from_numpy(X_test)
    torch_y_test = torch.from_numpy(y_test)
    # print(torch_X_train)
    # Pytorch train and test sets
    train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)
    test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)

    # data loader
    train_loader = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = BATCH_SIZE, shuffle = False)
    return train_loader, test_loader

def run():
    EPOCHS = 20
    BATCH_SIZE = 1
    X, y = read_data()
    train_loader, test_loader = process_data(X, y, BATCH_SIZE)
    model = Model()
    
    fit(model, train_loader, EPOCHS, BATCH_SIZE)
    evaluate(model, test_loader, BATCH_SIZE)
    torch.save(model.state_dict(), 'models/model_v1')


def test(model, X):
    X = torch.tensor(np.array([X]))
    X = Variable(X).float()
    output = model(X)
    predicted = torch.max(output.data, 1)[1]
    return predicted

def single_predict(model, X):
    mean_X = [5.84333333, 3.054, 3.75866667, 1.19866667] 
    sigma = [0.82530129, 0.43214658, 1.75852918, 0.76061262]
    for i in range(len(X)):
        X[i] = (X[i]- mean_X[i]) / sigma[i]
    print(X)
    model.eval()
    predicted = test(model, X)
    return labels[predicted.item()]

if __name__ == "__main__":
    run()


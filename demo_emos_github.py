import numpy as np
from torch.optim import Adam
import torch

from sklearn import metrics
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import scipy.io as scio
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from fastTSNE import TSNE
# from sklearn.manifold import TSNE
import time
import random
from sklearn.metrics import accuracy_score
from Focal_Loss import focal_loss
from scipy.stats import entropy
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)



class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.softmax=nn.Softmax(dim=1)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[1]
        scores = torch.matmul(queries, keys.T) / math.sqrt(d)
        self.attention_weights = self.softmax(scores)
        return torch.matmul(self.dropout(self.attention_weights), values)



#@save
class transformer_nn(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens, dropout, bias=False):
        super(transformer_nn, self).__init__()
        self.relu=nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attention1 = DotProductAttention(dropout)
        self.W_q1 = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k1 = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v1 = nn.Linear(value_size, num_hiddens, bias=bias)
        self.attention2 = DotProductAttention(dropout)
        self.W_q2 = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k2 = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v2 = nn.Linear(value_size, num_hiddens, bias=bias)
        self.attention3 = DotProductAttention(dropout)
        self.W_q3 = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k3 = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v3 = nn.Linear(value_size, num_hiddens, bias=bias)
        self.attention4 = DotProductAttention(dropout)
        self.W_q4 = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k4 = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v4 = nn.Linear(value_size, num_hiddens, bias=bias)


        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, x):
        output1 = self.relu(self.attention1(self.W_q1(x[:, 0, :]), self.W_k1(x[:, 0, :]), self.W_v1(x[:, 0, :])))
        output2 = self.relu(self.attention2(self.W_q2(x[:, 1, :]), self.W_k2(x[:, 1, :]), self.W_v2(x[:, 1, :])))
        output3 = self.relu(self.attention3(self.W_q3(x[:, 2, :]), self.W_k3(x[:, 2, :]), self.W_v3(x[:, 2, :])))
        output4 = self.relu(self.attention4(self.W_q4(x[:, 3, :]), self.W_k4(x[:, 3, :]), self.W_v4(x[:, 3, :])))
        output_concat = output1+output2+output3+output4
        classification=x[:, 0, :].T * torch.sum(output1, dim=1)+x[:, 1, :].T * torch.sum(output2, dim=1)+x[:, 2, :].T * torch.sum(output3, dim=1)+x[:, 3, :].T * torch.sum(output4, dim=1)
        return output_concat,self.softmax(classification.T)



class db_loss(torch.nn.Module):
    def __init__(self):
        super(db_loss, self).__init__()

    def forward(self, feature, label):
        feature_normal = feature[torch.where(label == 0)[0], :]
        feature_anomaly = feature[torch.where(label != 0)[0], :]
        normal_distance = torch.max(torch.cdist(torch.mean(feature_normal, 0).reshape((1,feature_normal.shape[1])), feature_normal ,p=2))
        if feature_anomaly.shape[0] != 0:
            anomaly_distance = torch.max(torch.cdist(torch.mean(feature_anomaly, 0).reshape((1,feature_anomaly.shape[1])), feature_anomaly ,p=2))
            # center_distance=torch.min(torch.cdist(feature_normal,feature_anomaly,p=2))
            center_distance = F.pairwise_distance(torch.mean(feature_normal, 0).reshape(1,feature.shape[1]), torch.mean(feature_anomaly, 0).reshape(1,feature.shape[1]), p=2)
        else:
            anomaly_distance = 0
            center_distance = 0

        # print(normal_distance+anomaly_distance)
        # print(center_distance)
        return (normal_distance + anomaly_distance) / center_distance


def evalution(predict_label, test_y):
    num_acc = 0
    num_seen = 0
    num = 0
    for i in range(predict_label.shape[0]):
        if predict_label[i] == test_y[i, :]:
            num_acc = num_acc + 1
            if predict_label[i] == 0:
                num = num + 1
            if predict_label[i] == 1:
                num_seen = num_seen + 1


    ACC = num_acc / test_y.shape[0]
    precision_seen = num_seen / np.where(predict_label == 1)[0].shape[0]
    recall_seen = num_seen / np.where(test_y == 1)[0].shape[0]
    precision_normal = num / np.where(predict_label == 0)[0].shape[0]
    recall_normal = num / np.where(test_y== 0)[0].shape[0]
    return ACC, precision_seen, recall_seen, precision_normal, recall_normal



if __name__=="__main__":

    # trainticket
    data_x = np.load('x_trainticket.npy')
    data_y = np.load('y_trainticket.npy')
    data_x = preprocessing.scale(data_x)
    data_y = data_y[:, np.newaxis]
    data_y[np.where(data_y!=0), :] = 1


    # model instance
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    model = transformer_nn(key_size=data_x.shape[1], query_size=data_x.shape[1], value_size=data_x.shape[1],
                           num_hiddens=10, dropout=0.5)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)

    # dataloader traindata
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.5, random_state=42)
    # batch_size = 512

    # training subset
    dataset = MyDataset(train_x, train_y)
    traindata_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    # testing subset
    dataset = MyDataset(test_x, test_y)
    testdata_loader = DataLoader(dataset, batch_size, shuffle=True)

    # TrainTicket Masked
    step = 4
    masked_initial = torch.zeros([1, step, data_x.shape[1]]).to(device)
    for i in range(step):
        masked_initial[0,i,i* 41:(i+1)*41] = 1
    masked_train = masked_initial.repeat((batch_size, 1, 1))

    # training
    optimizer1 = Adam(model.parameters(), lr=0.001)
    # optimizer1 = SGD(model.parameters(), lr=0.0001)
    # optimizer1 = Adagrad(model.parameters(), lr=1e-4)
    loss0 = db_loss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.99)
    all_loss1 = []
    model.train()
    for epoch in range(100):
        for batch_idx, batch in enumerate(traindata_loader):
            input_x, input_y = tuple(t.to(device) for t in batch)
            feature, _ = model(input_x.reshape(batch_size,1,data_x.shape[1])*masked_train)
            loss1 = loss0(feature, input_y)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            all_loss1.append(loss1.item())
        lr_scheduler.step()
    plt.plot(all_loss1)
    print("end")

    model.eval()
    masked_train = masked_initial.repeat((train_x.shape[0], 1, 1))
    train_x_cuda = torch.FloatTensor(train_x).to(device)
    masked_test = masked_initial.repeat((test_x.shape[0], 1, 1))
    test_x_cuda = torch.FloatTensor(test_x).to(device)
    feature_test, _ = model(test_x_cuda.reshape(test_x.shape[0],1,data_x.shape[1])*masked_test)
    feature_train, _ = model(train_x_cuda.reshape(train_x.shape[0], 1, data_x.shape[1]) * masked_train)
    feature_normal = feature_train[np.where(train_y == 0)[0], :]
    feature_anomaly = feature_train[np.where(train_y != 0)[0], :]

    dist_temp = torch.cdist(feature_test, feature_train, p=2)
    predict_label = np.ones_like(test_y) * 5

    for i in range(dist_temp.shape[0]):

        dist_detec = dist_temp[i, :]
        detec_values = torch.topk(dist_detec, 11, largest=False).values
        detec_loc = torch.topk(dist_detec, 11, largest=False).indices
        indicator = train_y[detec_loc.cpu().detach().numpy(), :]
        if np.sum(indicator)<6:
            predict_label[i, 0] = 0
        else:
            predict_label[i, 0] = 1

    ACC, precision_seen, recall_seen, precision_normal, recall_normal = evalution(predict_label,test_y)
    print('ACC:', ACC)
    print('precision_seen:', precision_seen)
    print('recall_seen:', recall_seen)
    print('precision_normal:', precision_normal)
    print('recall_normal:', recall_normal)

import numpy as np
from torch.optim import Adam
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
from sklearn.metrics import precision_score,recall_score
from sklearn.model_selection import train_test_split


class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))
    def __len__(self):
        return len(self.data)


class stepwise_attention(nn.Module):
    def __init__(self,sample_dim,num_hiddens):
        super(stepwise_attention, self).__init__()
        self.softmax=nn.Softmax(dim=1)
        self.relu=nn.ReLU()
        self.relu2= nn.ReLU()
        self.drop=nn.Dropout(0.5)
        self.vLinear=nn.Linear(sample_dim,num_hiddens)
        self.kLinear=nn.Linear(sample_dim,num_hiddens)
        self.qLinear=nn.Linear(sample_dim,num_hiddens)
        self.linear1=nn.Linear(sample_dim,sample_dim)
        self.normalize1=nn.LayerNorm(sample_dim)
        self.linear2=nn.Linear(sample_dim,sample_dim)
    def forward(self,x):
        masked_x=self.relu(self.linear2(self.normalize1(self.linear1(x))))
        x=masked_x*x
        key=self.kLinear(x)
        query=self.qLinear(x)
        value=self.vLinear(x)
        scores=torch.matmul(query,key.T)/math.sqrt(x.shape[1])
        z_feature=torch.matmul(self.drop(self.softmax(scores)),value)
        return masked_x,z_feature

class CauseFormer(nn.Module):
    def __init__(self,num_step,sample_dim,num_hiddens):
        super(CauseFormer, self).__init__()
        self.blks_feature = nn.Sequential()
        for i in range(num_step):
            self.blks_feature.add_module("blockfeature"+str(i),stepwise_attention(sample_dim,num_hiddens))
        self.linear1=nn.Linear(num_hiddens,num_hiddens)
        self.normalize1=nn.LayerNorm(num_hiddens)
        self.relu1=nn.ReLU()
        self.drop1=nn.Dropout(0.5)
        self.drop2= nn.Dropout(0.5)
        self.normalize2 = nn.LayerNorm(num_hiddens)
        self.relu2 = nn.ReLU()
        self.linear2=nn.Linear(num_hiddens,num_hiddens)
        self.linear3= nn.Linear(num_hiddens, num_hiddens)
        self.scale=nn.BatchNorm1d(sample_dim)
    def forward(self,x):
        x=self.scale(x)
        z_feature=0
        masked_x=[]
        for i,blks in enumerate(self.blks_feature):
            masked_temp, feature_temp=blks(x)
            masked_x.append(masked_temp)
            z_feature=z_feature+feature_temp
        masked_x=torch.stack(masked_x).transpose(1,0)
        output1=self.normalize1(self.drop1(z_feature+self.relu1(self.linear1(z_feature))))
        output2=self.normalize2(self.drop2(output1+self.relu2(self.linear2(output1))))
        output=self.linear3(output2)
        return masked_x,output

class db_loss(torch.nn.Module):
    def __init__(self):
        super(db_loss, self).__init__()
    def forward(self, feature,label,masked_x):
        feature_normal = feature[torch.where(label == 0)[0], :]
        feature_anomaly = feature[torch.where(label != 0)[0], :]
        normal_distance = torch.max(torch.cdist(torch.mean(feature_normal, 0).reshape((1,feature_normal.shape[1])), feature_normal ,p=2))
        if feature_anomaly.shape[0] != 0:
            anomaly_distance = torch.max(torch.cdist(torch.mean(feature_anomaly, 0).reshape((1,feature_anomaly.shape[1])), feature_anomaly ,p=2))
            center_distance = F.pairwise_distance(torch.mean(feature_normal, 0).reshape(1,feature.shape[1]), torch.mean(feature_anomaly, 0).reshape(1,feature.shape[1]), p=2)
        else:
            anomaly_distance = 0
            center_distance = 0
        ones=torch.tril(torch.ones(masked_x.shape[1],masked_x.shape[1]))-torch.eye(masked_x.shape[1],masked_x.shape[1])
        loss2=torch.mean(torch.matmul(torch.bmm(masked_x,masked_x.transpose(2,1)),ones.cuda()))
        return (normal_distance + anomaly_distance) / center_distance+loss2

def evalution(predict_label, test_y):
    precision_anomaly = precision_score(y_true=test_y,y_pred=predict_label,pos_label=1)
    recall_anomaly = recall_score(y_true=test_y,y_pred=predict_label,pos_label=1)
    precision_normal = precision_score(y_true=test_y,y_pred=predict_label,pos_label=0)
    recall_normal = precision_score(y_true=test_y,y_pred=predict_label,pos_label=0)
    precision=(precision_anomaly+precision_normal)/2
    recall=(recall_anomaly+recall_normal)/2
    f1=2*precision*recall/(precision+recall)
    return precision,recall,recall_anomaly,f1

def decision_label(feature_test, feature_train,test_y,train_y,k):
    dist_temp = torch.cdist(feature_test, feature_train, p=2)
    predict_label = np.ones_like(test_y)
    neighbor=[]
    for i in range(dist_temp.shape[0]):
        dist_detec = dist_temp[i, :]
        detec_values = torch.topk(dist_detec, k, largest=False).values
        detec_loc = torch.topk(dist_detec, k, largest=False).indices
        indicator = train_y[detec_loc.cpu().detach().numpy(), :]
        neighbor.append(detec_loc)
        if np.sum(indicator)<k/2:
            predict_label[i, 0] = 0
        else:
            predict_label[i, 0] = 1
    return predict_label,torch.stack(neighbor)

def hr_k(index,metric):
    count=0
    sum=0
    index, metric=index+1,metric+1
    for i in range(index.shape[0]):
        if index[i] in [j for j in range(15849, 16368)]:
            temp1=np.array([1,9,10,12,13,14,15])
            count=np.intersect1d(temp1,metric[i,:]).shape[0]+count
            sum=sum+7
        if index[i] in [j for j in range(16963, 17517)]:
            temp2 = np.array([1,2,3,4,6,7,9,10,11,12,13,14,15,16,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36])
            count = np.intersect1d(temp2, metric[i, :]).shape[0] + count
            sum = sum + 31
        if index[i] in [j for j in range(18071,18528)]:
            temp3 = np.array([1,2,9,10,12,13,14,15])
            count = np.intersect1d(temp3, metric[i, :]).shape[0] + count
            sum = sum + 8
        if index[i] in [j for j in range(19367,20088)]:
            temp4 = np.array([1,2,3,4,9,10,11,12,13,14,15,16,25,28])
            count = np.intersect1d(temp4, metric[i, :]).shape[0] + count
            sum = sum + 14
        if index[i] in [j for j in range(20786,21195)]:
            temp5 = np.array([1,9,10,12,13,14,15])
            count = np.intersect1d(temp5, metric[i, :]).shape[0] + count
            sum = sum + 7
        if index[i] in [24679,24680,24681,24682,26114,26115,26116,27554,27555,27556]:
            temp6 = np.array([13, 14, 15])
            count = np.intersect1d(temp6, metric[i, :]).shape[0] + count
            sum = sum + 3
    return count/sum

def acc_k(index,metric):
    # count=0
    sum=0
    # index, metric=index+1,metric+1
    count=0
    for i in range(index.shape[0]):
        if index[i] in [j for j in range(15849, 16368)]:
            temp1=np.array([j for j in range(15849, 16368)])
            if np.intersect1d(temp1,metric[i,:]).shape[0]>count:
                sum=sum+1
        if index[i] in [j for j in range(16963, 17517)]:
            temp2 = np.array([j for j in range(16963, 17517)])
            if np.intersect1d(temp2, metric[i, :]).shape[0] > count:
                sum = sum + 1
        if index[i] in [j for j in range(18071,18528)]:
            temp3 = np.array([j for j in range(18071,18528)])
            if np.intersect1d(temp3, metric[i, :]).shape[0] > count:
                sum = sum + 1
        if index[i] in [j for j in range(19367,20088)]:
            temp4 = np.array([j for j in range(19367,20088)])
            if np.intersect1d(temp4, metric[i, :]).shape[0] > count:
                sum = sum + 1
        if index[i] in [j for j in range(20786,21195)]:
            temp5 = np.array([j for j in range(20786,21195)])
            if np.intersect1d(temp5, metric[i, :]).shape[0] > count:
                sum = sum + 1
        if index[i] in [24679,24680,24681,24682,26114,26115,26116,27554,27555,27556]:
            temp6 = np.array([24679,24680,24681,24682,26114,26115,26116,27554,27555,27556])
            if np.intersect1d(temp6, metric[i, :]).shape[0] > count:
                sum = sum + 1
    return sum/(i+1)

if __name__=="__main__":

    # SMD Test Machine 1.1
    smd1 = np.load('smd1.npy')
    data_x=preprocessing.scale(smd1[:,0:-1])
    data_y=smd1[:,-1][:, np.newaxis]

    # model instance
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    step = 5
    model =  CauseFormer(num_step=step,sample_dim=data_x.shape[1],num_hiddens=64)
    model.to(device)

    # dataloader traindata
    x = np.array([i for i in range(data_x.shape[0])])
    data_x = np.hstack((data_x, x[:, np.newaxis]))
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.5)

    # training subset
    dataset = MyDataset(train_x, train_y)
    traindata_loader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    # testing subset
    dataset = MyDataset(test_x, test_y)
    testdata_loader = DataLoader(dataset, batch_size, shuffle=True)

    # training
    optimizer1 = Adam(model.parameters(), lr=0.001)
    loss0 = db_loss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.9)
    model.train()
    for epoch in range(250):
        for batch_idx, batch in enumerate(traindata_loader):
            input_x, input_y = tuple(t.to(device) for t in batch)
            masked_x,feature=model(input_x[:,0:-1])
            loss1 = loss0(feature, input_y,masked_x)
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
        lr_scheduler.step()

    model.eval()
    feature_train = []
    train_y = []
    masked_train=[]
    train_index = []
    with torch.no_grad():
        for batch_idx,batch in enumerate(traindata_loader):
            input_x,input_y=tuple(t.to(device) for t in batch)
            masked_x,feature=model(input_x[:,0:-1])
            train_index.append(input_x[:, -1])
            train_y.append(input_y)
            masked_train.append(masked_x)
            feature_train.append(feature)
    feature_train=torch.cat(feature_train)
    masked_train=torch.cat(masked_train)
    train_y = torch.cat(train_y).cpu().detach().numpy()
    train_index = torch.cat(train_index).cpu().detach().numpy()

    # testing
    test_y = []
    masked_test = []
    feature_test = []
    index_all = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(testdata_loader):
            input_x, input_y = tuple(t.to(device) for t in batch)
            masked_x,feature = model(input_x[:,0:-1])
            feature_test.append(feature)
            masked_test.append(masked_x)
            test_y.append(input_y)
            index_all.append(input_x[:, -1])
    feature_test = torch.cat(feature_test)
    masked_test=torch.cat(masked_test)
    test_y = torch.cat(test_y).cpu().detach().numpy()
    index_all = torch.cat(index_all).cpu().detach().numpy()

    # decision classification
    predict_part=0
    k=5
    annotation=[]
    neighbor=[]
    for i in range(step):
        temp,temp_neighbor=decision_label(masked_test[:,i,:],masked_train[:,i,:],test_y,train_y,k)
        predict_part = predict_part +temp
        annotation.append(torch.tensor(temp))
        neighbor.append(temp_neighbor)
    neighbor=torch.stack(neighbor).transpose(1,0)
    predict_label=predict_part
    predict_label[np.where(predict_label!=0)]=1
    precision,recall,recall_anomaly,f1 = evalution(predict_label,test_y)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)

    annotation = torch.stack(annotation).transpose(1, 0).cuda()
    weight = torch.sum(annotation * masked_test, dim=1)
    value, indices = torch.topk(weight, k=20, dim=1)
    # hr
    positive_identify = test_y + predict_label
    loc = np.where(positive_identify == 2)[0]
    hr = hr_k(index_all[loc], indices[loc, :].cpu().detach().numpy())
    # Acc
    loc_add = annotation * neighbor
    loc_add = loc_add.reshape(loc_add.shape[0], loc_add.shape[1] * loc_add.shape[2])
    loc_add = loc_add[loc, :]
    neighbor_index = train_index[loc_add.cpu().detach().numpy()]
    Acck = acc_k(index_all[loc], neighbor_index)
    print('HR@20:', hr*recall_anomaly)
    print('ACC_K:', Acck)


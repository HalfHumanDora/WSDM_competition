import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

from sklearn.model_selection import train_test_split

import re
import os
from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')

# from model import BERT_Classifier
from dataset import *
from collections import defaultdict
from sklearn.model_selection import KFold
import random


class BERT_Classifier(nn.Module):
    def __init__(self,target_size=3):
        super(BERT_Classifier, self).__init__()

        self.fc1 = nn.Linear(768, 768)
        self.fc1_bn = nn.BatchNorm1d(768)
        self.fc1_drop = nn.Dropout(p=0.3, inplace=False)
        self.fc2 = nn.Linear(768, target_size)

    def forward(self, last_encoder_layer):#, input_ids, input_mask):

        #last_encoder_layer, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)


        #print(last_encoder_layer.size())
        embedding = torch.sum(last_encoder_layer, 1)
        #print("embedding", embedding.size())

        fc1 = self.fc1_drop(F.relu(self.fc1_bn(self.fc1(embedding))))
        fc2 = self.fc2(fc1)

        return fc2





EMBEDDING_DIM = 512
HIDDEN_DIM = 256
max_seq_en = 50
max_seq_zh = 100
EPOCH=10

batch=32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)


train_df = pd.read_csv("data/train.csv")

train_df.replace('unrelated', 0, inplace=True)
train_df.replace('agreed', 1, inplace=True)
train_df.replace('disagreed', 2, inplace=True)


X = pd.read_pickle("save/features.pickle")
print("X:", X.shape)
y = list(train_df["label"])


p = list(zip(X, y))
random.shuffle(p)
X, y = zip(*p)
X = np.array(X)
y = np.array(y)


# K-Fold Cross validation
fold_num = 5
kf = KFold(n_splits=fold_num)
kf.get_n_splits(X, y)

train_data_list = []
val_data_list = []
fold=1
for train_index, val_index in kf.split(X):
    X_train = X[train_index]
    X_val = X[val_index]
    y_train = y[train_index]
    y_val = y[val_index]


    print("{}/{} fold :".format(fold, fold_num))
    print("train length:{}, val length:{}".format(len(X_train), len(X_val)))


    c = Counter(y_train)
    class_weight = []
    for label, num in sorted(c.items()):
        print(label, num)
        class_weight.append(len(y_train)/(3*num))
    class_weight = torch.FloatTensor(class_weight).to(device)




    model = BERT_Classifier()
    model.to(device)
    loss_function = nn.CrossEntropyLoss()#weight=class_weight)
    weighted_loss_function = nn.CrossEntropyLoss(weight=class_weight)#weight=class_weight)

    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    #ミニバッチ内のクラス比を揃える.
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False, sampler=sampler)#, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    def train(epoch):
        model.train()

        for batch_idx, sample_batch in enumerate(tqdm(train_loader)):
            inputs, y = sample_batch

            inputs = inputs.to(device)
            y = y.to(device)


            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_function(outputs, y)
            loss.backward()
            optimizer.step()

        print("epoch:{},train_loss:{:.4f}".format(epoch+1 ,loss))
        #print("train data all :", (batch_idx+1)*batch)

        return model



    def test():
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0

            for batch_idx, sample_batch in enumerate(val_loader):
                inputs, y = sample_batch
                inputs = inputs.to(device)
                y = y.to(device)


                output = model(inputs)
                # sum up batch loss
                test_loss += weighted_loss_function(output, y).item()
                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()

            #test_loss /= len(val_loader.dataset)
            test_loss /= batch_idx+1
            #accuracy = 100. * correct / len(val_loader.dataset)

            accuracy = weighted_accuracy(pred, y)

            print('Validation set: Weighted loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
                  .format(test_loss, correct, len(val_loader.dataset),
                          accuracy))

            return test_loss, accuracy


    def weighted_accuracy(pred, true):
        true = true.cpu().numpy()
        pred = pred.cpu().numpy()

        class_weight = [1/16, 1/15, 1/5]
        score = 0
        perfect_score = 0

        for p, t in zip(true, pred):
            if p == t:
                if t == 0:
                    score += 1/16
                    perfect_score += 1/16
                elif t == 1:
                    score += 1/15
                    perfect_score += 1/15
                elif t == 2:
                    score += 1/5
                    perfect_score += 1/5
            else:
                if t == 0:
                    perfect_score += 1/16
                elif t == 1:
                    perfect_score += 1/15
                elif t == 2:
                    perfect_score += 1/5
        #print("score:{}, ideal:{}".format(score, perfect_score))
        return 100 * score/perfect_score




    def save_model(model, val_accuracy, save_path="model/BERT/"):
        # if os.path.exists(path + "*.model"):
        #     os.remove(path + "*.model")
        name = "{}fold_mlp.model".format(fold)
        PATH = os.path.join(save_path, name)
        torch.save(model, PATH)

    lowest_loss = 1000000000
    highest_accuracy = 0
    for epoch in range(EPOCH):
        #print(epoch+1)
        model = train(epoch)
        val_loss, accuracy = test()

    #     if val_loss < lowest_loss:
    #         lowest_loss = val_loss
    #         save_model(model)

        if accuracy > highest_accuracy:
            #print("saving model...")
            highest_accuracy = accuracy
            #save_model(model, highest_accuracy)
        print("highest_accuracy:{:.2f}% \n".format(highest_accuracy))

    fold+=1

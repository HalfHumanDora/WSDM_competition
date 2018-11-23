import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import re

from nltk.corpus import stopwords
import nltk

from model import *
from dataset import TitleDataset, Toidx
from preprocess import preprocess_




EMBEDDING_DIM = 512
HIDDEN_DIM = 256
max_seq_en = 50
max_seq_zh = 100

batch=1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

(id1_train, id1_val, train1_en, val1_en, train1_zh, val1_zh, id2_train, id2_val, train2_en, val2_en,train2_zh, val2_zh, y_train, y_val), word_to_ix_en, word_to_ix_zh = preprocess_()

# Class weight gan be got as : n_samples / (n_classes * np.bincount(y))
# 不均衡データなので
c = Counter(y_train)
class_weight = []
for label, num in c.items():
    class_weight.append(len(y_train)/(3*num))
class_weight = torch.FloatTensor(class_weight).to(device)
print("class weight:", class_weight)


#model = LSTM_Classifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix_en), len(word_to_ix_zh), target_size=3, seq_length_en=max_seq_en,seq_length_zh=max_seq_zh)
#model = MLP_Classifier(EMBEDDING_DIM, len(word_to_ix_en), target_size=3, seq_length=max_seq_length)
#model = Text_CNN_Classifier(EMBEDDING_DIM, len(word_to_ix_en), target_size=3, seq_length=max_seq_length)
model = MLP_Classifier_Twolang(EMBEDDING_DIM, len(word_to_ix_en),len(word_to_ix_zh), target_size=3, seq_length=max_seq_length)

model.to(device)

loss_function = nn.CrossEntropyLoss(weight=class_weight)
#optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)


train_dataset = TitleDataset(train1_en, train2_en, train1_zh, train2_zh, y_train,
                             dic_en=word_to_ix_en, dic_zh=word_to_ix_zh, transform=Toidx(),
                             seq_length_en=max_seq_en, seq_length_zh=max_seq_zh)

val_dataset = TitleDataset(val1_en, val2_en, val1_zh, val2_zh, y_val,
                           dic_en=word_to_ix_en, dic_zh=word_to_ix_zh, transform=Toidx(),
                           seq_length_en=max_seq_en, seq_length_zh=max_seq_zh)



train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)#, sampler = sampler, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


def train(epoch):
    model.train()

    for batch_idx, sample_batch in enumerate(train_loader):
        #print("batch_idx:",batch_idx)
        en_title1 = sample_batch["t1_en"].to(device)
        en_title2 = sample_batch["t2_en"].to(device)
        zh_title1 = sample_batch["t1_zh"].to(device)
        zh_title2 = sample_batch["t2_zh"].to(device)
        y = sample_batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(en_title1, en_title2, zh_title1, zh_title2)

        loss = loss_function(outputs, y)
        loss.backward()
        optimizer.step()


        #optimizer.zero_grad()
        #outputs = model(en_title2, en_title1)

        #loss = loss_function(outputs, y)
        #loss.backward()
        #optimizer.step()

    print("epoch:{},train_loss:{:.4f}".format(epoch+1 ,loss))

    return model



def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0

        for batch_idx, sample_batch in enumerate(val_loader):
            en_title1 = sample_batch["t1_en"].to(device)
            en_title2 = sample_batch["t2_en"].to(device)
            zh_title1 = sample_batch["t1_zh"].to(device)
            zh_title2 = sample_batch["t2_zh"].to(device)
            y = sample_batch["label"].to(device)

            output = model(en_title1, en_title2, zh_title1, zh_title2)

            # sum up batch loss
            test_loss += loss_function(output, y).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

        #test_loss /= len(val_loader.dataset)
        test_loss /= batch_idx+1
        accuracy = 100. * correct / len(val_loader.dataset)
        print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
              .format(test_loss, correct, len(val_loader.dataset),
                      accuracy))

        return test_loss, accuracy


def save_model(model, path="model/LSTM.model"):
    torch.save(model, path)

lowest_loss = 1000000000
highest_accuracy = 0
for epoch in range(100):
    #print(epoch+1)
    model = train(epoch)
    val_loss, accuracy = test()

#     if val_loss < lowest_loss:
#         lowest_loss = val_loss
#         save_model(model)


    if accuracy > highest_accuracy:
        #print("saving model...")
        highest_accuracy = accuracy
        save_model(model)

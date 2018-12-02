import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from sklearn.model_selection import train_test_split

import re
import os
import pickle

from nltk.corpus import stopwords
import nltk

from model import *
from dataset import TitleDataset, Toidx
from preprocess import preprocess_, make_new_data

from sklearn.model_selection import KFold


EMBEDDING_DIM = 512
HIDDEN_DIM = 256
max_seq_en = 50
max_seq_zh = 100
EPOCH=100

batch=1024

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)



# with open('save/word_to_ix_en.pickle', mode='rb') as f:
#      word_to_ix_en = pickle.load(f)
# with open('save/word_to_ix_zh.pickle', mode='rb') as f:
#      word_to_ix_zh = pickle.load(f)

print("@preprocessing..")
#_ = preprocess_()

# Data loading
with open('save/word_to_ix_en.pickle', mode='rb') as f:
     word_to_ix_en = pickle.load(f)
with open('save/word_to_ix_zh.pickle', mode='rb') as f:
     word_to_ix_zh = pickle.load(f)
with open('save/train_df.pickle', mode='rb') as f:
     train_df = pickle.load(f)
with open('save/test_df.pickle', mode='rb') as f:
     test_df = pickle.load(f)
train_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)

# K-Fold Cross validation
fold_num = 5
kf = KFold(n_splits=fold_num)
kf.get_n_splits(train_df)

train_data_list = []
val_data_list = []
#
# for train_index, val_index in kf.split(train_df):
#     training_df = train_df.iloc[train_index]
#     val_df = train_df.iloc[val_index]
#
#     new_data, _, _, _ = make_new_data(training_df)
#     train1_en, train2_en = [],[]
#     train1_zh, train2_zh = [],[]
#     y_train = []
#     for text1_en, text2_en, text1_zh, text2_zh,label in new_data:
#             train1_en.append(text1_en)
#             train2_en.append(text2_en)
#             train1_zh.append(text1_zh)
#             train2_zh.append(text2_zh)
#             y_train.append(label)
#     val1_en, val2_en = list(val_df["title1_en"]), list(val_df["title2_en"])
#     val1_zh, val2_zh = list(val_df["title1_zh"]), list(val_df["title2_zh"])
#     y_val = list(val_df["label"])
#
#
#     train_data_list.append((train1_en,train2_en,train1_zh,train2_zh,y_train))
#     val_data_list.append((val1_en, val2_en,val1_zh, val2_zh,y_val))
#
# with open('save/kfold_train_data.pickle', mode='wb') as f:
#     pickle.dump(train_data_list, f)
# with open('save/kfold_val_data.pickle', mode='wb') as f:
#     pickle.dump(val_data_list, f)
#

with open('save/kfold_train_data.pickle', mode='rb') as f:
     train_data_list = pickle.load(f)
with open('save/kfold_val_data.pickle', mode='rb') as f:
     val_data_list = pickle.load(f)


fold=1
for train_fold, val_fold in zip(train_data_list,val_data_list):
    print("{}/{} fold :".format(fold, fold_num))
    print("train length:{}, val length:{}".format(len(train_fold[0]), len(val_fold[0])))

    (train1_en,train2_en,train1_zh,train2_zh,y_train) = train_fold
    (val1_en, val2_en,val1_zh, val2_zh,y_val) = val_fold


    # Class weight gan be got as : n_samples / (n_classes * np.bincount(y))
    # 不均衡データなので
    c = Counter(y_train)
    class_weight = []
    for label, num in sorted(c.items()):
        print(label, num)
        class_weight.append(len(y_train)/(3*num))
    class_weight = torch.FloatTensor(class_weight).to(device)
    print("class weight:", class_weight)


    #model = LSTM_Classifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix_en), len(word_to_ix_zh), target_size=3, seq_length_en=max_seq_en,seq_length_zh=max_seq_zh)
    #model = MLP_Classifier(EMBEDDING_DIM, len(word_to_ix_en), target_size=3, seq_length=max_seq_en)
    #model = Text_CNN_Classifier(EMBEDDING_DIM, len(word_to_ix_en), target_size=3, seq_length=max_seq_length)
    model = Twolang_Classifier(EMBEDDING_DIM, len(word_to_ix_en),len(word_to_ix_zh), target_size=3, kernel_num=64)

    model.to(device)

    loss_function = nn.CrossEntropyLoss()#weight=class_weight)
    weighted_loss_function = nn.CrossEntropyLoss(weight=class_weight)#weight=class_weight)

    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    train_dataset = TitleDataset(train1_en, train2_en, train1_zh, train2_zh, y_train,
                                 dic_en=word_to_ix_en, dic_zh=word_to_ix_zh, transform=Toidx(),
                                 seq_length_en=max_seq_en, seq_length_zh=max_seq_zh)

    val_dataset = TitleDataset(val1_en, val2_en, val1_zh, val2_zh, y_val,
                               dic_en=word_to_ix_en, dic_zh=word_to_ix_zh, transform=Toidx(),
                               seq_length_en=max_seq_en, seq_length_zh=max_seq_zh)


    #ミニバッチ内のクラス比を揃える.
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False, sampler=sampler)#, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)


    def train(epoch):
        model.train()

        for batch_idx, sample_batch in enumerate(tqdm(train_loader)):
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
        #print("train data all :", (batch_idx+1)*batch)

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




    def save_model(model, val_accuracy, save_path="model"):
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
            save_model(model, highest_accuracy)
        print("highest_accuracy:{:.2f}% \n".format(highest_accuracy))

    fold+=1

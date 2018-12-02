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

import re
import os

from model import *
from dataset import TitleDataset, Toidx
from preprocess import preprocess_, make_new_data

import pickle
from collections import defaultdict



# _ = preprocess_()


with open('save/word_to_ix_en.pickle', mode='rb') as f:
     word_to_ix_en = pickle.load(f)
with open('save/word_to_ix_zh.pickle', mode='rb') as f:
     word_to_ix_zh = pickle.load(f)
with open('save/train_df.pickle', mode='rb') as f:
     train_df = pickle.load(f)
with open('save/test_df.pickle', mode='rb') as f:
     test_df = pickle.load(f)






#_,given_dic,fixed_dic,forecast_dic = make_new_data(train_df)

with open('save/fixed_dic.pickle', mode='rb') as f:
    fixed_dic = pickle.load(f)


#推論
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


EMBEDDING_DIM = 512
HIDDEN_DIM = 128
max_seq_en = 50
max_seq_zh = 100

#model = LSTM_Classifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), target_size=3, seq_length=max_seq_length)
#model = MLP_Classifier(EMBEDDING_DIM, len(word_to_ix), target_size=3, seq_length=max_seq_length)
model = Twolang_Classifier(EMBEDDING_DIM, len(word_to_ix_en),len(word_to_ix_zh), target_size=3)

#PATH = "model/LSTM.model"

# PATH = "model/MLP.model"





title1_en_test = list(test_df["title1_en"])
title2_en_test = list(test_df["title2_en"])
title1_zh_test = list(test_df["title1_zh"])
title2_zh_test = list(test_df["title2_zh"])
test_tid1 = list(test_df["tid1"])
test_tid2 = list(test_df["tid2"])

id_ = test_df["id"]


preded_id_label = []

given, not_given = 0, 0
# 予測できるラベルもあるよね.
#
# fixed_dicの枝張りを収束まで行う.

agree_dic = defaultdict(list)
disagree_dic = defaultdict(list)

for id1, id_label_list in fixed_dic.items():
    if len(id_label_list) == 0:
        continue
    id_list = np.array(id_label_list)[:,0]
    label_list = np.array(id_label_list)[:,1]
    for id2, label in zip(id_list, label_list):

        if label == 1:
            agree_dic[id1].append(id2)
        elif label == 2:
            disagree_dic[id1].append(id2)

#番兵
#agreeのagreeはagree. agreeのdisagreeはdisagree.
print("二部グラフの作成...")
change=0
while True:
    for tid1, agree_id_list in agree_dic.items():
        for tid2 in agree_id_list:
            disagree_to_tid2 = disagree_dic[tid2]
            for dis in disagree_to_tid2:
                if not dis in disagree_dic[tid1]:
                    disagree_dic[tid1].append(dis)
                    change+=1

                if not tid1 in disagree_dic[dis]:
                    disagree_dic[dis].append(tid1)
                    change+=1

            agree_to_tid2 = agree_dic[tid2]
            for dis in agree_to_tid2:
                if not dis in agree_dic[tid1]:
                    agree_dic[tid1].append(dis)
                    change+=1

                if not tid1 in agree_dic[dis]:
                    agree_dic[dis].append(tid1)
                    change+=1
    for tid1, disagree_id_list in disagree_dic.items():
        for tid2 in disagree_id_list:

            agree_to_tid2 = agree_dic[tid2]
            for dis in agree_to_tid2:
                if not dis in disagree_dic[tid1]:
                    disagree_dic[tid1].append(dis)
                    change+=1

                if not tid1 in disagree_dic[dis]:
                    disagree_dic[dis].append(tid1)
                    change+=1

    print("change number: ", change)
    if change == 0:
        break
    else:
        change = 0

mujun = 0

for id1, id2, each_id in zip(test_tid1, test_tid2, id_):
    if id2 in disagree_dic[id1]:
        #check
        if id1 in disagree_dic[id2]:
            preded_id_label.append((each_id, 2))
        else:
            mujun+=1

    elif id2 in agree_dic[id1]:
        #check
        if id1 in agree_dic[id2]:
            preded_id_label.append((each_id, 1))
        else:
            mujun+=1


preded_id_label = []
print("予測できたもの:{}, 矛盾してたもの:{}, total:{}".format(len(preded_id_label), mujun, len(test_df)))




#
# for id1, id2, each_id in zip(test_tid1, test_tid2, id_):
#     if not id1 in forecast_dic.keys():
#         #print("label cannot be predicted")
#         not_given+=1
#         pass
#     else:
#         forecast_data_label = np.array(forecast_dic[id1])
#         if len(forecast_data_label) == 0:
#             continue
#
#         forecast_id = forecast_data_label[:,0]
#         forecast_label = forecast_data_label[:,1]
#
#         if id2 in forecast_id:
#             idx = list(forecast_id).index(id2)
#             label = forecast_label[idx]
#              given+=1
#             # preded_id_label.append((each_id, label))
#         else:
#             #print("label not given")
#             not_given+=1
#             pass
# print("予測可能セット:{}, わからないセット:{}".format(given, not_given))


PATH = "model/MLP.model"
PATH_list = ["model/{}fold_mlp.model".format(fold) for fold in range(1,6,1)]


average_prediction = []
for PATH in PATH_list:

    model = torch.load(PATH)
    print("model loaded:{}".format(PATH))


    # test dataset. label is None.
    test_dataset = TitleDataset(title1_en_test, title2_en_test, title1_zh_test, title2_zh_test, None,
                                dic_en=word_to_ix_en, dic_zh=word_to_ix_zh, transform=Toidx(),
                                seq_length_en=max_seq_en, seq_length_zh=max_seq_zh, if_test=True)


    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    with torch.no_grad():
        model.eval()
        predictions = []
        for batch_idx, sample_batch in enumerate(tqdm(test_loader)):
            en_title1 = sample_batch["t1_en"].to(device)
            en_title2 = sample_batch["t2_en"].to(device)
            zh_title1 = sample_batch["t1_zh"].to(device)
            zh_title2 = sample_batch["t2_zh"].to(device)
            output = model(en_title1, en_title2, zh_title1, zh_title2)

            # pred = output.max(1, keepdim=True)[1].cpu()
            #print("model out :",output.size())
            #predictions.extend(list(pred.numpy()))
            output = output.cpu().numpy()
            #print("model out:",output.shape)

            if batch_idx == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))

    average_prediction.append(predictions)

average_prediction = np.array(average_prediction)
# print("total pred:", average_prediction.shape)
average_prediction = np.mean(average_prediction, axis=0)
# print("total pred:", average_prediction.shape)

predictions = np.argmax(average_prediction, axis=1)
print("predictions:", predictions.shape)

#'unrelated', 0
#'agreed', 1
#'disagreed', 2


if len(preded_id_label) == 0:
    preded_labels = []
    preded_id = []
else:
    preded_id = np.array(preded_id_label)[:, 0]
    preded_labels = np.array(preded_id_label)[:, 1]
print("directly preded label:", len(preded_id))


fixed_predictions = []
for each_id, p in zip(id_, predictions):
    if each_id in preded_id:
        #trainの中に現れてたやつ
        idx = list(preded_id).index(each_id)
        fixed_predictions.append(preded_labels[idx])
    else:
        fixed_predictions.append(p)


new_predictions = []
for p in fixed_predictions:
    if p == 0:
        new_predictions.append("unrelated")
    elif p==1:
        new_predictions.append("agreed")
    elif p==2:
        new_predictions.append("disagreed")


#
# c = Counter(list(predictions))
# print("original",c)
#
# c = Counter(fixed_predictions)
# print("fixed", c)


submit_csv = pd.concat([id_, pd.Series(new_predictions)], axis=1)
#display(submit_csv)

submit_csv.columns = ["Id", "Category"]
submit_csv.to_csv("submit.csv", header=True, index=False)
submit = pd.read_csv("submit.csv")

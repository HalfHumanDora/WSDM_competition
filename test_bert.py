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
import argparse

from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')
import copy
# from model import BERT_Classifier
from dataset import *
from collections import defaultdict
from sklearn.model_selection import KFold
import pickle
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam



# from train_bert import BERT_Classifier, chinese_clean_series, english_clean_series
test_df = pd.read_csv("data/test.csv")#.iloc[:300,:]

with open('save/fixed_dic.pickle', mode='rb') as f:
    fixed_dic = pickle.load(f)
print("load fixed dic.", len(fixed_dic))
#
# test_tid1 = list(test_df["tid1"])
# test_tid2 = list(test_df["tid2"])
#
# id_ = test_df["id"]
#
#
# preded_id_label = []
#
# given, not_given = 0, 0
# # 予測できるラベルもあるよね.
# #
# # fixed_dicの枝張りを収束まで行う.
#
# agree_dic = defaultdict(list)
# disagree_dic = defaultdict(list)
#
# for id1, id_label_list in fixed_dic.items():
#     if len(id_label_list) == 0:
#         continue
#     id_list = np.array(id_label_list)[:,0]
#     label_list = np.array(id_label_list)[:,1]
#     for id2, label in zip(id_list, label_list):
#         if label == 1:
#             agree_dic[id1].append(id2)
#         elif label == 2:
#             disagree_dic[id1].append(id2)
#
# print(len(agree_dic), len(disagree_dic))
#
# #番兵
# #agreeのagreeはagree. agreeのdisagreeはdisagree.
# print("二部グラフの作成...")
# change=0
# while True:
#     for tid1, agree_id_list in agree_dic.items():
#         for tid2 in agree_id_list:
#             disagree_to_tid2 = disagree_dic[tid2]
#             for dis in disagree_to_tid2:
#                 if not dis in disagree_dic[tid1]:
#                     disagree_dic[tid1].append(dis)
#                     change+=1
#
#                 if not tid1 in disagree_dic[dis]:
#                     disagree_dic[dis].append(tid1)
#                     change+=1
#
#             agree_to_tid2 = agree_dic[tid2]
#             for dis in agree_to_tid2:
#                 if not dis in agree_dic[tid1]:
#                     agree_dic[tid1].append(dis)
#                     change+=1
#
#                 if not tid1 in agree_dic[dis]:
#                     agree_dic[dis].append(tid1)
#                     change+=1
#     for tid1, disagree_id_list in disagree_dic.items():
#         for tid2 in disagree_id_list:
#
#             agree_to_tid2 = agree_dic[tid2]
#             for dis in agree_to_tid2:
#                 if not dis in disagree_dic[tid1]:
#                     disagree_dic[tid1].append(dis)
#                     change+=1
#
#                 if not tid1 in disagree_dic[dis]:
#                     disagree_dic[dis].append(tid1)
#                     change+=1
#
#     print("change number: ", change)
#     if change == 0:
#         break
#     else:
#         break
#         change = 0
#
# mujun = 0
#
# for id1, id2, each_id in zip(test_tid1, test_tid2, id_):
#     if id1 == id2:
#         preded_id_label.append((each_id, 0))
#         continue
#     if id2 in disagree_dic[id1]:
#         #check
#         if id1 in disagree_dic[id2]:
#             preded_id_label.append((each_id, 2))
#             continue
#         else:
#             mujun+=1
#
#     elif id2 in agree_dic[id1]:
#         #check
#         if id1 in agree_dic[id2]:
#             preded_id_label.append((each_id, 1))
#         else:
#             mujun+=1
#
# del agree_dic, disagree_dic
# # preded_id_label = []
# with open('save/preded_id_label.pickle', mode='wb') as f:
#     pickle.dump(preded_id_label, f)

with open('save/preded_id_label.pickle', mode='rb') as f:
    preded_id_label = pickle.load(f)
print("予測できたもの:{}, total:{}".format(len(preded_id_label), len(test_df)))


class BERT_Classifier(nn.Module):
    def __init__(self, bert_model, target_size=3):
        super(BERT_Classifier, self).__init__()

        self.embedding_dim=768
        kernel_num=256
        self.seq_length_en=50

        self.bert_model = bert_model

        self.fc1 = nn.Linear(768, 768)
        #self.fc1_bn = nn.BatchNorm1d(300)
        self.fc1_drop = nn.Dropout(p=0.5, inplace=False)
        #self.activation = F.tanh()
        self.fc2 = nn.Linear(768, target_size)

    def forward(self, input_ids, input_mask):
        batch = len(input_ids)

        last_encoder_layer, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)

        first_token_tensor = last_encoder_layer[:, 0]

        fc1 = self.fc1_drop(F.relu(self.fc1(first_token_tensor)))
        #fc1 = self.fc1_drop(self.activation(self.fc1(first_token_tensor)))
        fc2 = self.fc2(fc1)

        return fc2


#
# def chinese_clean_series(series):
#     def clean_seq(seq):
#         seq = str(seq)
#         ori = copy.copy(seq)
#
#         seq = seq.replace("< i >", "")
#         seq = seq.replace("< / i >", "")
#         seq = seq.replace("\n", "")
#         seq = re.sub(r'[,."''“”。、#()→⇒←↓↑:;_㊙️【《》=|/<>]+', '', seq)
#         seq = re.sub(r'[!！？?-]+', ' ', seq)
#         seq = re.sub(r'[$]+', '$ ', seq)
#         seq = re.sub(r'[0-9]+', '<NUM>', seq)
#
#         if len(seq)==0:
#             print("0 lengrh assert!!,",ori, seq)
#
#         return seq
#
#     series = series.apply(clean_seq)
#     return series
#
# def english_clean_series(series):
#     # 大文字--->小文字
#     series = series.str.lower()
#
#     def clean_seq(seq):
#         ori = copy.copy(seq)
#
#         seq = seq.replace("it's", "it is")
#         seq = seq.replace("he's", "he is")
#         seq = seq.replace("she's", "she is")
#         seq = seq.replace("you're", "you are")
#         seq = seq.replace("we're", "we are")
#         seq = seq.replace("they're", "they are")
#         seq = seq.replace("i'm", "i am")
#         seq = seq.replace("don't", "do not")
#         seq = seq.replace("does't", "does not")
#         seq = seq.replace("didn't", "did not")
#         seq = seq.replace("aren't", "are not")
#         seq = seq.replace("weren't", "were not")
#         seq = seq.replace("isn't", "is not")
#         seq = seq.replace("wasn't", "was not")
#         seq = seq.replace("haven't", "have not")
#         seq = seq.replace("hasn't", "has not")
#         seq = seq.replace("can't", "can not")
#         seq = seq.replace("cannot", "can not")
#
#         seq = seq.replace("shouldn't", "should not")
#         seq = seq.replace("wouldn't", "would not")
#         seq = seq.replace("couldn't", "could not")
#         seq = seq.replace("mightn't", "might not")
#         seq = seq.replace("mustn't", "must not")
#         seq = seq.replace("needn't", "need not")
#         seq = seq.replace("won't", "will not")
#
#         seq = seq.replace("\n", "")
#
#         seq = seq.replace("< i >", "")
#         seq = seq.replace("< / i >", "")
#
#         seq = re.sub(r'[,."''“”。、#()→⇒←↓↑:;㊙️【《》=|/+<>]+', '', seq)
#         seq = re.sub(r'[!?]+', ' ', seq)
#         # seq = re.sub(r'[$]+', '$ ', seq)
#         # seq = re.sub(r'[0-9]+', '<NUM>', seq)
#
#         if len(seq)==0:
#             print("0 lengrh assert!!,",ori, seq)
#         return seq
#
#     series = series.apply(clean_seq)
#     return series

from preprocess import english_clean_series, chinese_clean_series

#推論
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print("device:", device)

max_seq_en = 50
max_seq_zh = 50
batch = 128

print("batch", batch)



test_df["title1_zh"] =  chinese_clean_series(test_df["title1_zh"])
test_df["title2_zh"] =  chinese_clean_series(test_df["title2_zh"])
test_df["title1_en"] = english_clean_series(test_df["title1_en"])
test_df["title2_en"] = english_clean_series(test_df["title2_en"])
id_ = test_df["id"]

test1_en, test2_en = list(test_df["title1_en"]), list(test_df["title2_en"])
test1_zh, test2_zh = list(test_df["title1_zh"]), list(test_df["title2_zh"])

model_dir_en = "model/BERT/balanced_8fold/en/"
model_dir_zh = "model/BERT/balanced_8fold/zh/"

MAX_fold = 8
PATH_list_en = [os.path.join(model_dir_en, "{}fold_bert.model".format(fold)) for fold in range(3,MAX_fold+1,1)]
PATH_list_zh = [os.path.join(model_dir_zh, "{}fold_bert.model".format(fold)) for fold in range(3,MAX_fold+1,1)]

PATH_list_en = [os.path.join(model_dir_en, "3fold_bert.model"),
                os.path.join(model_dir_en, "4fold_bert.model"),
                os.path.join(model_dir_en, "5fold_bert.model"),
                os.path.join(model_dir_en, "6fold_bert.model"),
                os.path.join(model_dir_en, "7fold_bert.model"),
                os.path.join(model_dir_en, "8fold_bert.model")]

PATH_list_zh = [os.path.join(model_dir_zh, "3fold_bert.model"),
                os.path.join(model_dir_zh, "4fold_bert.model"),
                os.path.join(model_dir_zh, "5fold_bert.model"),
                os.path.join(model_dir_zh, "6fold_bert.model"),
                os.path.join(model_dir_zh, "7fold_bert.model"),
                os.path.join(model_dir_zh, "8fold_bert.model")]


y_dummy = torch.empty(len(test1_en), dtype=torch.long).random_(5)

tokenizer_en = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_zh = BertTokenizer.from_pretrained('bert-base-chinese')

test_dataset_en = BERTDataset(test1_en, test2_en, y_dummy, tokenizer_en, seq_length=max_seq_en)
test_dataset_zh = BERTDataset(test1_zh, test2_zh, y_dummy, tokenizer_zh, seq_length=max_seq_zh)

test_loader_en = DataLoader(test_dataset_en, batch_size=batch, shuffle=False)
test_loader_zh = DataLoader(test_dataset_zh, batch_size=batch, shuffle=False)


average_prediction = []
#inference english
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BERT_Classifier(bert_model)#.to(device)
print("inference english...")
for PATH in PATH_list_en:
    # model.load_state_dict(torch.load(PATH, map_location="cuda:7"))
    model.load_state_dict(torch.load(PATH))
    model = model.to(device)

    # model = model.load_state_dict(torch.load(PATH)).to(device)
    print("model loaded:{}".format(PATH))

    with torch.no_grad():
        model.eval()
        predictions = []
        for batch_idx, sample_batch in enumerate(tqdm(test_loader_en)):
            input_ids = sample_batch["input_ids"].to(device)
            input_mask = sample_batch["input_mask"].to(device)
            input_type_ids = sample_batch["input_type_ids"].to(device)

            output = model(input_ids, input_mask)
            output = output.cpu().numpy()
            #print("model out:",output.shape)

            if batch_idx == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))

    average_prediction.append(predictions)



#inference english
print("inference chinese...")
bert_model = BertModel.from_pretrained('bert-base-chinese')
model = BERT_Classifier(bert_model)#.to(device)
for PATH in PATH_list_zh:
    print(PATH)
    model.load_state_dict(torch.load(PATH)) #, map_location=device))
    model.to(device)
    #model.load_state_dict(torch.load(PATH)).to(device)

    print("model loaded:{}".format(PATH))

    with torch.no_grad():
        model.eval()
        predictions = []
        for batch_idx, sample_batch in enumerate(tqdm(test_loader_zh)):
            input_ids = sample_batch["input_ids"].to(device)
            input_mask = sample_batch["input_mask"].to(device)
            input_type_ids = sample_batch["input_type_ids"].to(device)

            output = model(input_ids, input_mask)
            output = output.cpu().numpy()
            #print("model out:",output.shape)

            if batch_idx == 0:
                predictions = output
            else:
                predictions = np.vstack((predictions, output))

    average_prediction.append(predictions)

average_prediction = np.array(average_prediction)
print("total prediction:", average_prediction.shape)
average_prediction = np.mean(average_prediction, axis=0)
# print("total pred:", average_prediction.shape)

predictions = np.argmax(average_prediction, axis=1)
print("predictions:", predictions.shape)

# preded_id_label = []


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


#'unrelated', 0
#'agreed', 1
#'disagreed', 2

new_predictions = []
for p in fixed_predictions:
    if p == 0:
        new_predictions.append("unrelated")
    elif p==1:
        new_predictions.append("agreed")
    elif p==2:
        new_predictions.append("disagreed")


submit_csv = pd.concat([id_, pd.Series(new_predictions)], axis=1)
#display(submit_csv)

submit_csv.columns = ["Id", "Category"]
submit_csv.to_csv("result/balanced_8fold/submit.csv", header=True, index=False)

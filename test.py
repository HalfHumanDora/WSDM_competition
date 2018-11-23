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


from nltk.corpus import stopwords
import nltk

from model import *
from dataset import TitleDataset, Toidx
from preprocess import preprocess_


(id1_train, id1_val, train1_en, val1_en, train1_zh, val1_zh, id2_train, id2_val, train2_en, val2_en,train2_zh, val2_zh, y_train, y_val), word_to_ix_en, word_to_ix_zh, test_df = preprocess_()


#推論
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


EMBEDDING_DIM = 512
HIDDEN_DIM = 128
max_seq_en = 50
max_seq_zh = 100

#model = LSTM_Classifier(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), target_size=3, seq_length=max_seq_length)
#model = MLP_Classifier(EMBEDDING_DIM, len(word_to_ix), target_size=3, seq_length=max_seq_length)
model = MLP_Classifier_Twolang(EMBEDDING_DIM, len(word_to_ix_en),len(word_to_ix_zh), target_size=3, seq_length=max_seq_en)

#PATH = "model/LSTM.model"
PATH = "model/MLP.model"
model = torch.load(PATH)
print("model loaded.")


title1_en_test = list(test_df["title1_en"])
title2_en_test = list(test_df["title2_en"])
title1_zh_test = list(test_df["title1_zh"])
title2_zh_test = list(test_df["title2_zh"])
id_ = test_df["id"]

# test dataset. label is None.
test_dataset = TitleDataset(title1_en_test, title2_en_test, title1_zh_test, title2_zh_test, None,
                            dic_en=word_to_ix_en, dic_zh=word_to_ix_zh, transform=Toidx(),
                            seq_length_en=max_seq_en, seq_length_zh=max_seq_zh, if_test=True)


test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

with torch.no_grad():
    model.eval()
    predictions = []
    for sample_batch in tqdm(test_loader):
        en_title1 = sample_batch["t1_en"].to(device)
        en_title2 = sample_batch["t2_en"].to(device)
        zh_title1 = sample_batch["t1_zh"].to(device)
        zh_title2 = sample_batch["t2_zh"].to(device)
        output = model(en_title1, en_title2, zh_title1, zh_title2)

        pred = output.max(1, keepdim=True)[1].cpu()
        #print(output.cpu(), pred)
        predictions.extend(list(pred.numpy()))

#'unrelated', 0
#'agreed', 1
#'disagreed', 2

new_predictions = []
for p in predictions:
    if p == 0:
        new_predictions.append("unrelated")
    elif p==1:
        new_predictions.append("agreed")
    elif p==2:
        new_predictions.append("disagreed")





c = Counter(new_predictions)
print(c)


submit_csv = pd.concat([id_, pd.Series(new_predictions)], axis=1)
#display(submit_csv)

submit_csv.columns = ["Id", "Category"]
submit_csv.to_csv("submit.csv", header=True, index=False)
submit = pd.read_csv("submit.csv")

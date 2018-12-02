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

from nltk.corpus import stopwords
import nltk
#nltk.download('stopwords')

# from model import BERT_Classifier
from dataset import *
from collections import defaultdict
from sklearn.model_selection import KFold


train_df = pd.read_csv("data/train.csv")
train_df.replace('unrelated', 0, inplace=True)
train_df.replace('agreed', 1, inplace=True)
train_df.replace('disagreed', 2, inplace=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def english_clean_series(series):
    # 大文字--->小文字
    series = series.str.lower()

    def clean_seq(seq):
        seq = seq.replace("it's", "it is")
        seq = seq.replace("he's", "he is")
        seq = seq.replace("she's", "she is")
        seq = seq.replace("you're", "you are")
        seq = seq.replace("we're", "we are")
        seq = seq.replace("they're", "they are")
        seq = seq.replace("i'm", "i am")
        seq = seq.replace("don't", "do not")
        seq = seq.replace("does't", "does not")
        seq = seq.replace("didn't", "did not")
        seq = seq.replace("aren't", "are not")
        seq = seq.replace("weren't", "were not")
        seq = seq.replace("isn't", "is not")
        seq = seq.replace("wasn't", "was not")
        seq = seq.replace("haven't", "have not")
        seq = seq.replace("hasn't", "has not")
        seq = seq.replace("can't", "can not")
        seq = seq.replace("cannot", "can not")

        seq = seq.replace("shouldn't", "should not")
        seq = seq.replace("wouldn't", "would not")
        seq = seq.replace("couldn't", "could not")
        seq = seq.replace("mightn't", "might not")
        seq = seq.replace("mustn't", "must not")
        seq = seq.replace("needn't", "need not")
        seq = seq.replace("won't", "will not")

        seq = seq.replace("\n", "")



        seq = seq.replace("< i >", "")
        seq = seq.replace("< / i >", "")

        seq = re.sub(r'[,."''“”]+', '', seq)

        return seq



    series = series.apply(clean_seq)
    return series


train_df["title1_en"] = english_clean_series(train_df["title1_en"])
train_df["title2_en"] = english_clean_series(train_df["title2_en"])




train1_en, train2_en = list(train_df["title1_en"]), list(train_df["title2_en"])
y_train = list(train_df["label"])



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)

bert_model.eval()


dataset = BERTDataset(train1_en, train2_en, y_train, tokenizer)
loader = DataLoader(dataset, batch_size=1, shuffle=False)#, pin_memory=True)



features = []
for batch_idx, sample_batch in enumerate(tqdm(loader)):
    input_ids = sample_batch["input_ids"].to(device)
    input_mask = sample_batch["input_mask"].to(device)
    input_type_ids = sample_batch["input_type_ids"].to(device)
    y = sample_batch["label"].to(device)
    last_encoder_layer, _ = bert_model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)

    embedding = last_encoder_layer[0].detach().cpu().numpy()

    features.append(embedding)

    


features = np.asarray(features)
print(features.shape)

pd.to_pickle(features, "save/features.pickle")

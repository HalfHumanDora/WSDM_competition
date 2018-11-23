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
#nltk.download('stopwords')
stopwords = stopwords.words('english')


def preprocess_():

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    sub = pd.read_csv("data/sample_submission.csv")

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
            seq = seq.replace("'s", "")
            seq = seq.replace("\n", "")



            seq = seq.replace("< i >", "")
            seq = seq.replace("< / i >", "")

            seq = re.sub(r'[,."''“”。、#()→⇒←↓↑:;_㊙️【《》]+', '', seq)
            seq = re.sub(r'[-!?]+', ' ', seq)
            seq = re.sub(r'[$]+', '$ ', seq)
            seq = re.sub(r'[0-9]+', '<NUM>', seq)

            return seq

        series = series.apply(clean_seq)
        return series

    def chinese_clean_series(series):
        def clean_seq(seq):
            seq = str(seq)
            seq = seq.replace("< i >", "")
            seq = seq.replace("< / i >", "")
            seq = seq.replace("\n", "")
            seq = re.sub(r'[,."''“”。、#()→⇒←↓↑:;_㊙️【《》]+', '', seq)
            seq = re.sub(r'[!！？?-]+', ' ', seq)
            seq = re.sub(r'[$]+', '$ ', seq)
            seq = re.sub(r'[0-9]+', '<NUM>', seq)

            return seq
        series = series.apply(clean_seq)
        return series

    train_df["title1_en"] = english_clean_series(train_df["title1_en"])
    train_df["title2_en"] = english_clean_series(train_df["title2_en"])
    train_df["title1_zh"] =  chinese_clean_series(train_df["title1_zh"])
    train_df["title2_zh"] =  chinese_clean_series(train_df["title2_zh"])

    test_df["title1_en"] = english_clean_series(test_df["title1_en"])
    test_df["title2_en"] = english_clean_series(test_df["title2_en"])
    test_df["title1_zh"] =  chinese_clean_series(test_df["title1_zh"])
    test_df["title2_zh"] =  chinese_clean_series(test_df["title2_zh"])

    train_df.replace('unrelated', 0, inplace=True)
    train_df.replace('agreed', 1, inplace=True)
    train_df.replace('disagreed', 2, inplace=True)

    y = list(train_df["label"])


    #単語辞書の作成

    train_t1_en = train_df["title1_en"]
    train_t2_en = train_df["title2_en"]

    test_t1_en = test_df["title1_en"]
    test_t2_en = test_df["title2_en"]

    train_t1_zh = train_df["title1_zh"]
    train_t2_zh = train_df["title2_zh"]
    test_t1_zh = test_df["title1_zh"]
    test_t2_zh = test_df["title2_zh"]

    label = train_df["label"]


    word_to_ix_en = {}
    for title1, title2 in zip(tqdm(train_t1_en), train_t2_en):
        for word in title1.split():
            if word not in word_to_ix_en.keys():
                word_to_ix_en[word] = len(word_to_ix_en)+1
        for word in title2.split():
            if word not in word_to_ix_en.keys():
                word_to_ix_en[word] = len(word_to_ix_en)+1

    for title1, title2 in zip(tqdm(test_t1_en), test_t2_en):
        for word in title1.split():
            if word not in word_to_ix_en.keys():
                word_to_ix_en[word] = len(word_to_ix_en)+1
        for word in title2.split():
            if word not in word_to_ix_en.keys():
                word_to_ix_en[word] = len(word_to_ix_en)+1

    #Chinese
    word_to_ix_zh = {}
    for title1, title2 in zip(tqdm(train_t1_zh), train_t2_zh):
        for word in title1.split():
            if word not in word_to_ix_zh.keys():
                word_to_ix_zh[word] = len(word_to_ix_zh)+1
        for word in title2.split():
            if word not in word_to_ix_zh.keys():
                word_to_ix_zh[word] = len(word_to_ix_zh)+1

    for title1, title2 in zip(tqdm(test_t1_zh), test_t2_zh):
        for word in title1.split():
            if word not in word_to_ix_zh.keys():
                word_to_ix_zh[word] = len(word_to_ix_zh)+1
        for word in title2.split():
            if word not in word_to_ix_zh.keys():
                word_to_ix_zh[word] = len(word_to_ix_zh)+1

    print("the number of english words:{}, chinese words:{}".format(len(word_to_ix_en), len(word_to_ix_zh)))



    # Aにagreeな記事たちは、Aにdisagreeな記事たちとdisagreeな関係にあるかも?

    def make_new_data(id1, id2, title1, title2, labels):
    #     title1_en = list(train_df["title1_en"])
    #     title2_en = list(train_df["title2_en"])
    #     labels = list(train_df["label"])
    #     id1_train = list(train_df["tid1"])
    #     id2_train = list(train_df["tid2"])

        title1_en = list(title1)
        title2_en = list(title2)
        id1_train = list(id1)
        id2_train = list(id2)

        # id-text dictionary
        id_to_text = {}
        for idx, id1 in enumerate(id1_train):
            if not id1 in id_to_text.keys():
                id_to_text[id1] = title1_en[idx]
        for idx, id2 in enumerate(id2_train):
            if not id2 in id_to_text.keys():
                id_to_text[id2] = title2_en[idx]

        # key : id,
        # value : id of agreed text or diagreed text.
        agree_dic = {}
        disagree_dic = {}
        #すでにラベルが与えられているid-id
        given_dic = {}

        #initialize dic
        for id1 in id1_train:
            agree_dic[id1] = []
            disagree_dic[id1] = []
            given_dic[id1] = []
        for id2 in id2_train:
            agree_dic[id2] = []
            disagree_dic[id2] = []
            given_dic[id2] = []


        # agree, disagree辞書の作成
        for idx, id1 in enumerate(id1_train):
            label = labels[idx]
            id2 = id2_train[idx]

            given_dic[id1].append([id2, label])
            given_dic[id2].append([id1, label])


            if label == 1:
                agree_dic[id1].append(id2)
                agree_dic[id2].append(id1)
            elif label == 2:
                disagree_dic[id1].append(id2)
                disagree_dic[id2].append(id1)

        print("creating new data.")
        new_data = []
        given_label = []

        for id_, agree_ids in agree_dic.items():
            disagree_ids = disagree_dic[id_]

            for agree_id in agree_ids:
                given_ids_labels= given_dic[agree_id]
                given_ids = np.array(given_ids_labels)[:, 0]
                given_labels = np.array(given_ids_labels)[:, 1]
                assert given_ids.shape == given_labels.shape

                # new 'disagree data'
                for disagree_id in disagree_ids:
                    if disagree_id in given_ids:
                        #すでにラベルが与えられている時.
                        idx = list(given_ids).index(disagree_id)
                        label = given_labels[idx]
                        #given_label.append(label)
                        pass
                    else:
                        new_data.append((id_to_text[agree_id], id_to_text[disagree_id], 2))

                # new 'agree data'
    #             for agree_id2 in agree_ids:
    #                 if agree_id == agree_id2:
    #                     continue
    #                 else:
    #                     if agree_id2 in given_ids:
    #                         #すでにラベルが与えられている時.
    #                         idx = list(given_ids).index(agree_id2)
    #                         label = given_labels[idx]
    #                         given_label.append(label)
    #                         pass
    #                     else:
    #                         new_data.append((id_to_text[agree_id], id_to_text[agree_id2], 1))

        #c = Counter(given_label)
        #print(c)
        return new_data


    title1_en = list(train_df["title1_en"])
    title2_en = list(train_df["title2_en"])
    title1_zh = list(train_df["title1_zh"])
    title2_zh = list(train_df["title2_zh"])
    labels = list(train_df["label"])

    id1 = list(train_df["tid1"])
    id2 = list(train_df["tid2"])

    id1_train, id1_val, train1_en, val1_en, train1_zh, val1_zh, id2_train, id2_val, train2_en, val2_en,train2_zh, val2_zh, y_train, y_val = train_test_split(id1, title1_en, title1_zh, id2, title2_en, title2_zh, labels, test_size=0.2, random_state=0)

    new_data_en = make_new_data(id1_train, id2_train, train1_en, train2_en, y_train)
    print(len(new_data_en))
    for text1, text2, label in new_data_en:
            train1_en.append(text1)
            train2_en.append(text2)
            y_train.append(label)

    new_data_zh = make_new_data(id1_train, id2_train, train1_zh, train2_zh, y_train)
    print(len(new_data_zh))
    for text1, text2, label in new_data_zh:
            train1_zh.append(text1)
            train2_zh.append(text2)

    assert len(train1_zh)==len(train1_en)  and len(y_train)==len(train1_zh)

    print("training data:{}, validation data:{}".format(len(y_train), len(y_val)))


    return (id1_train, id1_val, train1_en, val1_en, train1_zh, val1_zh, id2_train, id2_val, train2_en, val2_en,train2_zh, val2_zh, y_train, y_val), word_to_ix_en, word_to_ix_zh

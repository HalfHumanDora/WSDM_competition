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
import copy
# from model import BERT_Classifier
from dataset import *
from collections import defaultdict
from sklearn.model_selection import KFold

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


import requests
def line(Me):
    line_token = "2rSRaP9FPrDCgIPNGuY3LRJ73jRhjq5xgbLDovGMzHi"
    line_token = "8G6JxWLK0dh3KjqEFgdcQVNr2EZKPOdD59o9cHsdhCC"
    line_notify_token = line_token #先程発行したコードを貼ります
    line_notify_api = 'https://notify-api.line.me/api/notify'
    message = '\n' + Me
    #変数messageに文字列をいれて送信します トークン名の隣に文字が来てしまうので最初に改>行しました
    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)

    #return 0

class BERT_Classifier(nn.Module):
    def __init__(self, bert_model, target_size=3):
        super(BERT_Classifier, self).__init__()

        self.embedding_dim=768
        kernel_num=256
        self.seq_length_en=100

        self.bert_model = bert_model
        # self.conv2_en = nn.Conv2d(1, kernel_num, (2, self.embedding_dim))
        # self.conv3_en = nn.Conv2d(1, kernel_num, (3, self.embedding_dim))
        # self.conv4_en = nn.Conv2d(1, kernel_num, (4, self.embedding_dim))
        # self.Max2_pool_en = nn.MaxPool2d((self.seq_length_en-2+1, 1))
        # self.Max3_pool_en = nn.MaxPool2d((self.seq_length_en-3+1, 1))
        # self.Max4_pool_en = nn.MaxPool2d((self.seq_length_en-4+1, 1))


        # self.fc1 = nn.Linear(kernel_num*3, 300)
        # self.fc1_bn = nn.BatchNorm1d(300)
        # self.fc1_drop = nn.Dropout(p=0.3, inplace=False)
        # self.fc2 = nn.Linear(300, target_size)

        self.fc1 = nn.Linear(768, 768)
        #self.fc1_bn = nn.BatchNorm1d(300)
        self.fc1_drop = nn.Dropout(p=0.3, inplace=False)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(768, target_size)

    def forward(self, input_ids, input_mask):
        batch = len(input_ids)

        last_encoder_layer, _ = self.bert_model(input_ids, token_type_ids=None, attention_mask=input_mask, output_all_encoded_layers=False)



        first_token_tensor = last_encoder_layer[:, 0]

        # fc1 = self.fc1_drop(F.relu(self.fc1(first_token_tensor)))
        fc1 = self.fc1_drop(self.activation(self.fc1(first_token_tensor)))
        fc2 = self.fc2(fc1)

        return fc2


line("train_bert_en.py starting...")


EMBEDDING_DIM = 512
HIDDEN_DIM = 256
max_seq_en = 50
max_seq_zh = 100
EPOCH=50

batch=32

gradient_accumulation_steps=1

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("device:",device)


train_df = pd.read_csv("data/train.csv")
# test_df = pd.read_csv("data/test.csv")

train_df.replace('unrelated', 0, inplace=True)
train_df.replace('agreed', 1, inplace=True)
train_df.replace('disagreed', 2, inplace=True)

def english_clean_series(series):
    # 大文字--->小文字
    series = series.str.lower()

    def clean_seq(seq):
        ori = copy.copy(seq)

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

        seq = re.sub(r'[,."''“”。、#()→⇒←↓↑:;㊙️【《》=|/+<>]+', '', seq)
        seq = re.sub(r'[!?]+', ' ', seq)
        # seq = re.sub(r'[$]+', '$ ', seq)
        # seq = re.sub(r'[0-9]+', '<NUM>', seq)

        if len(seq)==0:
            print("0 lengrh assert!!,",ori, seq)
        return seq

    series = series.apply(clean_seq)
    return series

train_df["title1_en"] = english_clean_series(train_df["title1_en"])
train_df["title2_en"] = english_clean_series(train_df["title2_en"])


train_df = train_df.sample(frac=1, random_state=0).reset_index(drop=True)#.iloc[:300, :]


# K-Fold Cross validation
fold_num = 5
kf = KFold(n_splits=fold_num)
kf.get_n_splits(train_df)

# kf.get_n_splits(X, y)

train_data_list = []
val_data_list = []
for train_index, val_index in kf.split(train_df):
#for train_index, val_index in kf.split(X):
    training_df = train_df.iloc[train_index]
    val_df = train_df.iloc[val_index]

    train1_en, train2_en = list(training_df["title1_en"]), list(training_df["title2_en"])
    y_train = list(training_df["label"])

    val1_en, val2_en = list(val_df["title1_en"]), list(val_df["title2_en"])
    #val1_zh, val2_zh = list(val_df["title1_zh"]), list(val_df["title2_zh"])
    y_val = list(val_df["label"])

    train_data_list.append((train1_en,train2_en, y_train))#train1_zh,train2_zh,y_train))
    val_data_list.append((val1_en, val2_en, y_val))# val1_zh, val2_zh,y_val))
#
# with open('save/kfold_train_data.pickle', mode='wb') as f:
#     pickle.dump(train_data_list, f)
# with open('save/kfold_val_data.pickle', mode='wb') as f:
#     pickle.dump(val_data_list, f)



fold=1
for train_fold, val_fold in zip(train_data_list,val_data_list):
    print("{}/{} fold :".format(fold, fold_num))
    print("train length:{}, val length:{}".format(len(train_fold[0]), len(val_fold[0])))

    (train1_en,train2_en,y_train) = train_fold
    (val1_en, val2_en,y_val) = val_fold

    c = Counter(y_train)
    class_weight = []
    for label, num in sorted(c.items()):
        print(label, num)
        class_weight.append(len(y_train)/(3*num))
    class_weight = torch.FloatTensor(class_weight).to(device)

    num_train_steps = int(len(train_fold[0]) / batch / gradient_accumulation_steps * EPOCH)



    # model = BERT_Classifier(bert_model)
    # model.to(device)
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model = BERT_Classifier(bert_model).to(device)

    loss_function = nn.CrossEntropyLoss()#weight=class_weight)
    weighted_loss_function = nn.CrossEntropyLoss(weight=class_weight)#weight=class_weight)

    #optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    print("t_total", t_total)

    local_rank=-1
    learning_rate=5e-5
    warmup_proportion=0.1

    if local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=learning_rate,
                         warmup=warmup_proportion,
                         t_total=t_total)


    train_dataset = BERTDataset(train1_en, train2_en, y_train, tokenizer, seq_length=max_seq_en)
    val_dataset = BERTDataset(val1_en, val2_en, y_val, tokenizer, seq_length=max_seq_en)

    #ミニバッチ内のクラス比を揃える.
    class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in y_train])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=False, sampler=sampler)#, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    def train(epoch, global_step):
        model.train()
        tr_loss = 0

        for batch_idx, sample_batch in enumerate(tqdm(train_loader)):
            input_ids = sample_batch["input_ids"].to(device)
            input_mask = sample_batch["input_mask"].to(device)
            input_type_ids = sample_batch["input_type_ids"].to(device)
            y = sample_batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, input_mask)

            loss = loss_function(outputs, y)
            tr_loss += loss.item()
            #loss = model(input_ids, input_type_ids, input_mask, y)
            loss.backward()
            optimizer.step()

            global_step+=1

            if batch_idx%100==0:
                print("epoch:{},train_loss:{:.4f}".format(epoch+1 ,loss))


        print("epoch:{},train_loss:{:.4f}".format(epoch+1 ,loss))
        line("EN, fold:{}, epoch:{},train_loss:{:.4f}".format(fold, epoch+1 ,loss))

        #print("train data all :", (batch_idx+1)*batch)
        tr_loss /= (batch_idx+1)

        return model, tr_loss, global_step



    def test(tr_loss):
        with torch.no_grad():
            model.eval()
            test_loss = 0
            correct = 0

            for batch_idx, sample_batch in enumerate(val_loader):
                input_ids = sample_batch["input_ids"].to(device)
                input_mask = sample_batch["input_mask"].to(device)
                input_type_ids = sample_batch["input_type_ids"].to(device)
                y = sample_batch["label"].to(device)

                output = model(input_ids, input_mask)
                # sum up batch loss
                test_loss += weighted_loss_function(output, y).item()
                #test_loss += loss_function(output, y).item()
                #total_loss += model(input_ids, input_type_ids, input_mask, y)

                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()

            #test_loss /= len(val_loader.dataset)
            test_loss /= batch_idx+1
            #accuracy = 100. * correct / len(val_loader.dataset)

            accuracy = weighted_accuracy(pred, y)

            print('Validation set: Weighted loss: {:.4f}, Weighted Accuracy: {}/{} ({:.2f}%)'
                  .format(test_loss, correct, len(val_loader.dataset),
                          accuracy))

            line('tori,EN : Validation set: Weighted loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'
                    .format(test_loss, correct, len(val_loader.dataset),
                            accuracy))

            result = {'val_loss': test_loss,
                      'eval_accuracy': accuracy,
                      'global_step': global_step,
                      'train_loss': tr_loss}

            output_eval_file = os.path.join("result/en/", "{}_fold_eval_results.txt".format(fold))
            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                writer.write("\n" )

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




    def save_model(model, val_accuracy, save_path="model/BERT/en"):
        # if os.path.exists(path + "*.model"):
        #     os.remove(path + "*.model")
        name = "{}fold_mlp.model".format(fold)
        PATH = os.path.join(save_path, name)
        torch.save(model, PATH)

    lowest_loss = 1000000000
    highest_accuracy = 0
    global_step=0

    for epoch in range(EPOCH):
        #print(epoch+1)
        model, tr_loss, global_step = train(epoch, global_step)
        val_loss, accuracy = test(tr_loss)

    #     if val_loss < lowest_loss:
    #         lowest_loss = val_loss
    #         save_model(model)

        if accuracy > highest_accuracy:
            #print("saving model...")
            highest_accuracy = accuracy
            save_model(model, highest_accuracy)
        print("highest_accuracy:{:.2f}% \n".format(highest_accuracy))

    fold+=1

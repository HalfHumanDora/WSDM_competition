import pandas as pd
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Dataset
class TitleDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, titles1_en, titles2_en, titles1_zh, titles2_zh, labels, dic_en=None, dic_zh=None, transform=None, seq_length_en=50, seq_length_zh=140, if_test=False):

        self.titles1_en = titles1_en
        self.titles2_en = titles2_en
        self.titles1_zh = titles1_zh
        self.titles2_zh = titles2_zh

        self.labels = labels
        self.transform = transform
        self.dic_en=dic_en
        self.dic_zh=dic_zh

        self.seq_length_en=seq_length_en
        self.seq_length_zh=seq_length_zh

        self.if_test=if_test

    def __len__(self):
        return len(self.titles1_en)

    def __getitem__(self, idx):
        title1_en = self.titles1_en[idx]
        title2_en = self.titles2_en[idx]
        title1_zh = self.titles1_zh[idx]
        title2_zh = self.titles2_zh[idx]

        if self.if_test:
            # dummy label
            label = title1_en
        else:
            label = torch.tensor(self.labels[idx])

        sample = {'t1_en': title1_en, 't2_en': title2_en, 't1_zh': title1_zh, 't2_zh': title2_zh, 'label': label}

        if self.transform:
            sample = self.transform(sample, self.dic_en, self.dic_zh, self.seq_length_en, self.seq_length_zh)

        return sample


class Toidx(object):
    def __call__(self, sample, word_to_idx_en, word_to_idx_zh, max_seq_length_en, max_seq_length_zh):

        def prepare_sequence(seq, to_ix, max_seq_length):
            #zero padding and word--->ix in seq.
            idxs = [to_ix[w] for w in seq.split()]
            if len(idxs) > max_seq_length:
                idxs = idxs[:max_seq_length]
            else:
                idxs += [0] * (max_seq_length - len(idxs))
            return torch.tensor(idxs, dtype=torch.long)

        t1_en, t2_en, t1_zh, t2_zh, label = sample['t1_en'], sample['t2_en'], sample['t1_zh'], sample['t2_zh'], sample["label"]
        return {'t1_en': prepare_sequence(t1_en, word_to_idx_en, max_seq_length_en),
                    't2_en': prepare_sequence(t2_en, word_to_idx_en, max_seq_length_en),
                    't1_zh': prepare_sequence(t1_zh, word_to_idx_zh, max_seq_length_zh),
                    't2_zh': prepare_sequence(t2_zh, word_to_idx_zh, max_seq_length_zh),
                    'label': label}

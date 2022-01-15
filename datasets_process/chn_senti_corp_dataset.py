#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial

import torch
from torch.utils.data import DataLoader

from datasets_process.chinese_bert_dataset import ChineseBertDataset
from datasets_process.collate_functions import collate_to_max_length


class ChnSentCorpDataset(ChineseBertDataset):

    def get_lines(self):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        return lines[1:]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        try:
            label, sentence = line.split('\t', 1)
        except:
            print(line)
        sentence = sentence[:self.max_length - 2]
        # convert sentence to ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # assert
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # convert list to tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor([int(label)])
        return input_ids, pinyin_ids, label



#!/usr/bin/env python
# -*- coding: utf-8 -*-


from functools import partial

import torch
from torch.utils.data import DataLoader

from datasets_process.chinese_bert_dataset import ChineseBertDataset
from datasets_process.collate_functions import collate_to_max_length


class ThuCNewsDataset(ChineseBertDataset):

    def __init__(self, data_path, chinese_bert_path, max_length: int = 512):
        super().__init__(data_path, chinese_bert_path, max_length)
        self.lable_map = {"体育": 0, "娱乐": 1, "家居": 2, "房产": 3, "教育": 4,
                          "时尚": 5, "时政": 6, "游戏": 7, "科技": 8, "财经": 9}

    def get_lines(self):
        with open(self.data_path, 'r') as f:
            lines = f.readlines()
        return lines[1:]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        label, sentence = line.split('\t', 1)
        sentence = sentence[:self.max_length - 2]
        # 将句子转为ids
        tokenizer_output = self.tokenizer.encode(sentence)
        bert_tokens = tokenizer_output.ids
        pinyin_tokens = self.convert_sentence_to_pinyin_ids(sentence, tokenizer_output)
        # 验证正确性，id个数应该相同
        assert len(bert_tokens) <= self.max_length
        assert len(bert_tokens) == len(pinyin_tokens)
        # 转化list为tensor
        input_ids = torch.LongTensor(bert_tokens)
        pinyin_ids = torch.LongTensor(pinyin_tokens).view(-1)
        label = torch.LongTensor([self.lable_map[label]])
        return input_ids, pinyin_ids, label



import OpenAttack
import datasets
from ChnSetiCorp_trainer import ChnSentiClassificationTask
import json
import argparse
import torch
import pytorch_lightning as pl
import os
from datasets_process.bert_dataset import BertDataset
from pypinyin import pinyin, Style
import numpy as np
import pandas
from OpenAttack.substitutes.chinese_glyph_pron_char import ChineseGlyphPronSubstitute
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset_mapping(x):
    return {
        "x": x["text_a"],
        "y": x["label"],
    }

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
    
def main():
    res=[]
    path='adv_result_new/textbugger_attack/'
    samples=[]
    with open(os.path.join(path, 'DMSC_chinesebert_1000_textbugger_representation_result.txt'), 'r',encoding='utf8') as fout:
        for line in fout.readlines():
            line_=line.strip().split(' ',1)
            #print(line_)
            samples.append(line_[1])
    print(samples)
    print("Building model")
    #clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")
    path='trained_models/checkpoint/DMSC_chinesebert.ckpt'
    with open("trained_models/checkpoint/chinesebert_args.json",'r') as load_f:
        dict = json.load(load_f)
        print(dict)
    args = argparse.Namespace(**dict)
    chinesebert = ChnSentiClassificationTask(args)
    checkpoint = torch.load(path)#, map_location=lambda storage, loc: storage.cuda(0))
    chinesebert.load_state_dict(checkpoint['state_dict'])
    # clsf = BaseBert_model(model,args)
    #sent="于我而言，新宿站的繁华和湖边小镇的恬静都是如此真实，又何尝不是一场梦"
    chinesebert=chinesebert.to(device)
    chinesebert.eval()
    for sent in tqdm(samples):
        tokenizer = BertDataset(args.bert_path)
        input_ids, pinyin_ids = tokenizer.tokenize_sentence(sent)
        length = input_ids.shape[0]
        input_ids = input_ids.view(1, length)
        pinyin_ids = pinyin_ids.view(1, length, 8)
        if torch.cuda.is_available():  # GPU available
            pinyin_ids = pinyin_ids.to(torch.device('cuda'))
            input_ids = input_ids.to(torch.device('cuda'))
        #self.model.
        # y_output = self.model(input_ids, pinyin_ids)
        outputs=chinesebert.model.bert(input_ids, pinyin_ids)
        pooled_output = outputs[1]
        pooled_output =chinesebert.model.dropout(pooled_output)
        # hidden_states=clsf.get_representation(sent)
        representation=pooled_output.cpu().detach().numpy().tolist()
        res.append(representation[0])
    print(len(res))
    res=np.array(res)
    np.save('adv_result_new/textbugger_attack/DMSC_chinesebert_1000_textbugger_representation_result.npy',res) 



if __name__ == "__main__":
    main()

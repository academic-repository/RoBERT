import OpenAttack
import datasets
from ChnSetiCorp_trainer_basebert import ChnSentiClassificationTask_BaseBERT
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
    savepath='adv_result_new/textbugger_attack/'
    samples=[]
    with open(os.path.join(savepath, 'DMSC_transformer6_normal_representation_result.txt'), 'r',encoding='utf8') as fout:
        for line in fout.readlines():
            line_=line.strip().split(' ',1)
            #print(line_)
            samples.append(line_[1])
    print(samples)
    print("Building model")
    #clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")
    path='trained_models/checkpoint/DMSC_basebert.ckpt'
    with open("trained_models/checkpoint/basebert_args.json",'r') as load_f:
        dict = json.load(load_f)
        print(dict)
    args = argparse.Namespace(**dict)
    basebert = ChnSentiClassificationTask_BaseBERT(args)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    basebert.load_state_dict(checkpoint['state_dict'])
    # clsf = BaseBert_model(model,args)
    #sent="于我而言，新宿站的繁华和湖边小镇的恬静都是如此真实，又何尝不是一场梦"
    basebert=basebert.to(device)
    basebert.eval()
    for sent in tqdm(samples):
        input_ids,token_type_ids,attention_mask = basebert.encode_fn(sent)
        outputs=basebert.model.bert(input_ids.to(device),token_type_ids.to(device),attention_mask.to(device))
        pooled_output = outputs[1]
        pooled_output =basebert.model.dropout(pooled_output)
        # hidden_states=clsf.get_representation(sent)
        representation=pooled_output.cpu().detach().numpy().tolist()
        res.append(representation[0])
    print(len(res))
    res=np.array(res)
    np.save(os.path.join(savepath, 'DMSC_transformer6_normal_representation_result.npy'),res) 



if __name__ == "__main__":
    main()

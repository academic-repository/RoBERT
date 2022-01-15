import OpenAttack
import datasets
from ChnSetiCorp_trainer_basebert_transformer import ChnSentiClassificationTask_BaseBERT_node2vec
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
    path='adv_result_roberta/textbugger/'
    samples=[]
    with open(os.path.join(path, 'roberta_wwm_dmsc_transformer6_adv_representation_result.txt'), 'r',encoding='utf8') as fout:
        for line in fout.readlines():
            line_=line.strip().split(' ',1)
            #print(line_)
            samples.append(line_[1])
    print(samples)
    print("Building model")
    #clsf = OpenAttack.loadVictim("BERT.AMAZON_ZH").to("cuda:0")
    path='trained_models/checkpoint/roberta_wwm_dmsc_transformer6_textbuggeradv.ckpt'
    print("model path",path)
    with open("trained_models/checkpoint/args_node2vec_attn.json",'r') as load_f:
        dict = json.load(load_f)
        print(dict)
    args = argparse.Namespace(**dict)
    robert = ChnSentiClassificationTask_BaseBERT_node2vec(args)
    checkpoint = torch.load(path)
    robert.load_state_dict(checkpoint['state_dict'])
    # clsf = BaseBert_model(model,args)
    #sent="于我而言，新宿站的繁华和湖边小镇的恬静都是如此真实，又何尝不是一场梦"
    robert=robert.to(device)
    robert.eval()
    for sent in tqdm(samples):
        sentence = [sent]
        input_ids,token_type_ids,attention_mask = robert.encode_fn(sentence)
        node_vec=robert.node2vec(sentence)
        node_vec=torch.tensor(node_vec)
        y_output = robert.forward(input_ids.to(device),token_type_ids.to(device),attention_mask.to(device),node_vec.to(device))
        represent =robert.representation
        # hidden_states=clsf.get_representation(sent)
        represent=represent.cpu().detach().numpy().tolist()
        res.append(represent[0])
    print(len(res))
    res=np.array(res)
    np.save('adv_result_roberta/textbugger/roberta_wwm_dmsc_transformer6_adv_representation_result.npy',res) 



if __name__ == "__main__":
    main()

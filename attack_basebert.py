import OpenAttack
import datasets
from trainer_basebert import BaseBERT
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseBert_model(OpenAttack.Classifier):
    def __init__(self,model,args):
        # nltk.sentiment.vader.SentimentIntensityAnalyzer is a traditional sentiment classification model.
        # self.model = SentimentIntensityAnalyzer()
        # path='trained_models/checkpoint/best_model.ckpt'
        # with open("trained_models/checkpoint/args.json",'r') as load_f:
        #     dict = json.load(load_f)
        #     print(dict)
        self.args = args
        # self.model = ChnSentiClassificationTask(self.args)
        # checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        # self.model.load_state_dict(checkpoint['state_dict'])
        self.model=model.to(device)
        self.model.eval()
    
    # access to the classification probability scores with respect input sentences
    def get_prob(self, input_):
        ret = []
        for sent in input_:
            input_ids,token_type_ids,attention_mask = self.model.encode_fn(sent)
            y_output = self.model(input_ids=input_ids.to(device),token_type_ids=token_type_ids.to(device),attention_mask=attention_mask.to(device))
            y_output = torch.cuda.FloatTensor(y_output[0])
            s = torch.exp(y_output)
            y_hat=s / torch.sum(s, dim=1, keepdim=True)
            output=y_hat.cpu().detach().numpy().tolist()[0]
            ret.append(np.array(output))

        return np.array(ret)



def dataset_mapping(x):
    return {
        "x": x["text_a"],
        "y": x["label"],
    }

import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)


def get_parser():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--save_path", required=True, type=str, help="save path of results")
    parser.add_argument("--attacker", default="pwws", type=str, help="attack algorithms, pwws or textbugger or random")
    parser.add_argument("--data_num", default=1000, type=int, help="path of datasets")    
    parser.add_argument("--model_path", required=True, type=str, help="path of model")
    parser.add_argument("--data_path", required=True, type=str, help="path of datasets")
    parser.add_argument("--mode", default="attack", type=str, help="attack or augmentation or representation")
    
    return parser
    
def main():
    parser = get_parser()
    args = parser.parse_args()
    print("Loading chinese processor and substitute")
    chinese_processor = OpenAttack.text_processors.ChineseTextProcessor()
    chinese_substitute = ChineseGlyphPronSubstitute()

    print("New Attacker")
    if args.attacker=="pwws":
        attacker = OpenAttack.attackers.PWWSAttacker(processor=chinese_processor, substitute=chinese_substitute,path=args.save_path,  threshold=40,mode=args.mode)
    elif args.attacker=="textbugger":
        attacker = OpenAttack.attackers.TextBuggerAttacker(processor=chinese_processor, substitute=chinese_substitute,path=args.save_path,  threshold=40,mode=args.mode)
    else:
        attacker = OpenAttack.attackers.RandomAttacker(processor=chinese_processor, substitute=chinese_substitute,path=args.save_path,  threshold=40,mode=args.mode)

    print("Building model")
    with open("trained_models/checkpoint/args_basebert.json",'r') as load_f:
        dict = json.load(load_f)
        print(dict)
    model_args = argparse.Namespace(**dict)
    model = BaseBERT(model_args)
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    clsf = BaseBert_model(model,model_args)
    split_num="train[:"+str(args.data_num)+"]"


    print("Loading dataset")
    dataset = datasets.load_dataset('csv', data_files=args.data_path,delimiter='\t',split=split_num).map(function=dataset_mapping)
    print(dataset)
    correct_samples = [
        inst for inst in dataset if (len(inst["x"])<512 and clsf.get_pred( [inst["x"]] )[0] == inst["y"])
    ]
    #print(correct_samples)
    print(len(correct_samples))

    print("Start attack")
    options = {
        "success_rate": True,
        "fluency": True,
        "mistake": True,
        "semantic": True,
        "levenstein": True,
        "word_distance": True,
        "modification_rate": True,
        "running_time": True,
    }
    attack_eval = OpenAttack.attack_evals.ChineseAttackEval(attacker, clsf, **options, num_process=1)
    attack_eval.eval(correct_samples, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()

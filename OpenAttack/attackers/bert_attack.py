import copy
import numpy as np
from ..utils import check_parameters
from ..attacker import Attacker
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification, BertForMaskedLM
from ..exceptions import WordNotInDictionaryException
from ..substitutes import CounterFittedSubstitute

DEFAULT_CONFIG = {
    "token_unk": "<UNK>",
    "mlm_path": 'bert-base-uncased',
    "k": 36,
    "use_bpe": 1,
    "use_sim_mat": 1,
    "threshold_pred_score": 0.3,
    "max_length": 512,
    "batch_size": 32,
    "device": None
}

filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)

class Feature(object):
    def __init__(self, seq_a, label):
        self.label = label
        self.seq = seq_a
        self.final_adverse = seq_a
        self.query = 0
        self.change = 0
        self.success = 0
        self.sim = 0.0
        self.changes = []

class BERTAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param str token_unk: A token which means "unknown token" in Classifier's vocabulary.
        :param: str mlm_path: the path to the masked language model. **Default:** 'bert-base-uncased'
        :param: int k: the k most important words / sub-words to substitute for. **Default:** 36
        :param: int use_bpe:  whether use bpe. **Default:** 1
        :param: int use_sim_mat: whether use cosine_similarity to filter out atonyms. **Default:** 1
        :param float threshold_pred_score: Threshold used in substitute module. **Default:** 0.3
        :param: int max_length: the maximum length of an input sentence. **Default:** 512
        :param int batch_size: the size of a batch of input sentences. **Default:** 32
        
        :Package Requirements:
            * torch
        :Classifier Capacity: Probability

        BERT-ATTACK: Adversarial Attack Against BERT Using BERT, Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue, Xipeng Qiu, EMNLP2020
        `[pdf] <https://arxiv.org/abs/2004.09984>`__
        `[code] <https://github.com/LinyangLee/BERT-Attack>`__
        """
        from transformers import BertConfig, BertTokenizer, BertForMaskedLM
        import torch

        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        check_parameters(self.config.keys(), DEFAULT_CONFIG)

        self.tokenizer_mlm = BertTokenizer.from_pretrained(self.config['mlm_path'], do_lower_case=True)
        if self.config["device"] is not None:
            self.device = self.config["device"]
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config_atk = BertConfig.from_pretrained(self.config['mlm_path'])
        self.mlm_model = BertForMaskedLM.from_pretrained(self.config['mlm_path'], config=config_atk).to(self.device)
        self.k = self.config['k']
        self.use_bpe = self.config['use_bpe']
        self.threshold_pred_score = self.config['threshold_pred_score']
        self.max_length = self.config['max_length']
        self.batch_size = self.config['batch_size']
        # if self.config['use_sim_mat'] == 1:
        #     # data counter fit vectors
        #     self.cos_mat, self.w2i, self.i2w = self.get_sim_embed('data_defense/counter-fitted-vectors.txt', 'data_defense/cos_sim_counter_fitting.npy')
        # else:        
        #     self.cos_mat, self.w2i, self.i2w = None, {}, {}

    def __call__(self, clsf, x_orig, target=None):
        import torch
        x_orig = x_orig.lower()
        if target is None:
            # targeted
            targeted = False
            target = clsf.get_pred([x_orig])[0]  # calc x_orig's prediction
        else:
            targeted = True
        
        # return None
        tokenizer = self.tokenizer_mlm
        # MLM-process
        feature = Feature(x_orig, target)
        words, sub_words, keys = self._tokenize(feature.seq, tokenizer)
        max_length = self.max_length
        # original label
        inputs = tokenizer.encode_plus(feature.seq, None, add_special_tokens=True, max_length=max_length, )
        input_ids, token_type_ids = torch.tensor(inputs["input_ids"]), torch.tensor(inputs["token_type_ids"])
        attention_mask = torch.tensor([1] * len(input_ids))
        seq_len = input_ids.size(0)

        orig_probs = torch.Tensor(clsf.get_prob([feature.seq]))
        orig_probs = orig_probs[0].squeeze()
        orig_probs = torch.softmax(orig_probs, -1)
       
        current_prob = orig_probs.max()

        sub_words = ['[CLS]'] + sub_words[:2] + sub_words[2:max_length - 2] + ['[SEP]']
       
        input_ids_ = torch.tensor([tokenizer.convert_tokens_to_ids(sub_words)])
        word_predictions = self.mlm_model(input_ids_.to(self.device))[0].squeeze()  # seq-len(sub) vocab
        word_pred_scores_all, word_predictions = torch.topk(word_predictions, self.k, -1)  # seq-len k

        word_predictions = word_predictions[1:len(sub_words) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1:len(sub_words) + 1, :]

        important_scores = self.get_important_scores(words, clsf, current_prob, target, orig_probs,
                                                tokenizer, self.batch_size, max_length)
        feature.query += int(len(words))
        list_of_index = sorted(enumerate(important_scores), key=lambda x: x[1], reverse=True)
        final_words = copy.deepcopy(words)

        for top_index in list_of_index:
            if feature.change > int(0.2 * (len(words))):
                feature.success = 1  # exceed
                return None

            tgt_word = words[top_index[0]]
            if tgt_word in filter_words:
                continue
            if keys[top_index[0]][0] > max_length - 2:
                continue

            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

            substitutes = self.get_substitues(substitutes, tokenizer, self.mlm_model, self.use_bpe, word_pred_scores, self.threshold_pred_score)

            if self.config['use_sim_mat']:
                CFS = CounterFittedSubstitute(cosine=True)
                try:
                    cfs_output = CFS(tgt_word, threshold=0.4)
                    cos_sim_subtitutes = [elem[0] for elem in cfs_output]
                    substitutes = list(set(substitutes) & set(cos_sim_subtitutes))
                except WordNotInDictionaryException:
                    pass
                    # print("The target word is not representable by counter fitted vectors. Keeping the substitutes output by the MLM model.")
            most_gap = 0.0
            candidate = None
            
            for substitute in substitutes:               
                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word

                if substitute in filter_words:
                    continue
                # if substitute in self.w2i and tgt_word in self.w2i:
                #     if self.cos_mat[self.w2i[substitute]][self.w2i[tgt_word]] < 0.4:
                #         continue
                
                temp_replace = final_words
                temp_replace[top_index[0]] = substitute
                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, )
                input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
                seq_len = input_ids.size(1)
                
                temp_prob = torch.Tensor(clsf.get_prob([temp_text]))[0].squeeze()
                feature.query += 1
                temp_prob = torch.softmax(temp_prob, -1)
                temp_label = torch.argmax(temp_prob)

                if (not targeted and temp_label) != target or (targeted and temp_label == target):
                    feature.change += 1
                    final_words[top_index[0]] = substitute
                    feature.changes.append([keys[top_index[0]][0], substitute, tgt_word])
                    feature.final_adverse = temp_text
                    feature.success = 4
                    return feature.final_adverse, temp_label
                else:
                    label_prob = temp_prob[target]
                    gap = current_prob - label_prob
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute

            if most_gap > 0:
                feature.change += 1
                feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
                current_prob = current_prob - most_gap
                final_words[top_index[0]] = candidate

        feature.final_adverse = (tokenizer.convert_tokens_to_string(final_words))
        feature.success = 2
        return None


    def _tokenize(self, seq, tokenizer):
        seq = seq.replace('\n', '').lower()
        words = seq.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, words):
        len_text = len(words)
        masked_words = []
        for i in range(len_text - 1):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words
    
    def _get_masked_insert(self, words):
        len_text = len(words)
        masked_words = []
        for i in range(len_text - 1):
            masked_words.append(words[0:i + 1] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words
    
    def get_important_scores(self, words, tgt_model, orig_prob, orig_label, orig_probs, tokenizer, batch_size, max_length):
        import torch
        masked_words = self._get_masked(words)
        texts = [' '.join(words) for words in masked_words]  # list of text of masked words
        leave_1_probs = torch.Tensor(tgt_model.get_prob(texts))
        leave_1_probs = torch.softmax(leave_1_probs, -1)  
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

        import_scores = (orig_prob
                        - leave_1_probs[:, orig_label]
                        +
                        (leave_1_probs_argmax != orig_label).float()
                        * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                        ).data.cpu().numpy()

        return import_scores

    def get_bpe_substitues(self, substitutes, tokenizer, mlm_model):
        import torch
        # substitutes L, k

        substitutes = substitutes[0:12, 0:4] # maximum BPE candidates

        # find all possible candidates 
        all_substitutes = []
        for i in range(substitutes.size(0)):
            if len(all_substitutes) == 0:
                lev_i = substitutes[i]
                all_substitutes = [[int(c)] for c in lev_i]
            else:
                lev_i = []
                for all_sub in all_substitutes:
                    for j in substitutes[i]:
                        lev_i.append(all_sub + [int(j)])
                all_substitutes = lev_i

        # all substitutes  list of list of token-id (all candidates)
        c_loss = torch.nn.CrossEntropyLoss(reduction='none')
        word_list = []
        # all_substitutes = all_substitutes[:24]
        all_substitutes = torch.tensor(all_substitutes) # [ N, L ]
        all_substitutes = all_substitutes[:24].to(self.device)
        # print(substitutes.size(), all_substitutes.size())
        N, L = all_substitutes.size()
        word_predictions = mlm_model(all_substitutes)[0] # N L vocab-size

        ppl = c_loss(word_predictions.view(N*L, -1), all_substitutes.view(-1)) # [ N*L ] 
        ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1)) # N  
        _, word_list = torch.sort(ppl)
        word_list = [all_substitutes[i] for i in word_list]
        final_words = []
        for word in word_list:
            tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
            text = tokenizer.convert_tokens_to_string(tokens)
            final_words.append(text)
        return final_words

    def get_substitues(self, substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
        # substitues L,k
        # from this matrix to recover a word
        words = []
        sub_len, k = substitutes.size()  # sub-len, k

        if sub_len == 0:
            return words
            
        elif sub_len == 1:
            for (i,j) in zip(substitutes[0], substitutes_score[0]):
                if threshold != 0 and j < threshold:
                    break
                words.append(tokenizer._convert_id_to_token(int(i)))
        else:
            if use_bpe == 1:
                words = self.get_bpe_substitues(substitutes, tokenizer, mlm_model)
            else:
                return words
        return words
    
    def get_sim_embed(self, embed_path, sim_path):
        id2word = {}
        word2id = {}

        with open(embed_path, 'r', encoding='utf-8') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in id2word:
                    id2word[len(id2word)] = word
                    word2id[word] = len(id2word) - 1

        cos_sim = np.load(sim_path)
        return cos_sim, word2id, id2word


import copy
import random
import numpy as np
from ..utils import check_parameters
from ..data_manager import DataManager
from ..attacker import Attacker
from ..metric import UniversalSentenceEncoder

DEFAULT_CONFIG = {
    "token_unk": "<UNK>",
    "mlm_path": 'bert-base-uncased',
    "k": 50,
    "threshold_pred_score": 0.3,
    "max_length": 512,
    "batch_size": 32,
    "replace_rate": 1.0,
    "insert_rate": 0.0,
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

class BAEAttacker(Attacker):
    def __init__(self, **kwargs):
        """
        :param str token_unk: A token which means "unknown token" in Classifier's vocabulary.
        :param: str mlm_path: the path to the masked language model. **Default:** 'bert-base-uncased'
        :param: int k: the k most important words / sub-words to substitute for. **Default:** 50
        :param: int use_sim_mat: whether use cosine_similarity to filter out atonyms. **Default:** 0
        :param float threshold_pred_score: Threshold used in substitute module. **Default:** 0.3
        :param: int max_length: the maximum length of an input sentence. **Default:** 512
        :param int batch_size: the size of a batch of input sentences. **Default:** 32
        
        :Package Requirements:
            * torch
        :Data Requirements: :py:data:`.TProcess.NLTKPerceptronPosTagger`
        :Classifier Capacity: Probability

        BAE: BERT-based Adversarial Examples for Text Classification. Siddhant Garg, Goutham Ramakrishnan. EMNLP 2020. 
        `[pdf] <https://arxiv.org/abs/2004.01970>`__
        `[code] <https://github.com/QData/TextAttack/blob/master/textattack/attack_recipes/bae_garg_2019.py>`__
        This script is adapted from <https://github.com/LinyangLee/BERT-Attack> given the high similarity between the two attack methods.
        This attacker supports the 4 attack methods (BAE-R, BAE-I, BAE-R/I, BAE-R+I) in the paper. 
        """
        from transformers import BertConfig, BertTokenizer, BertForMaskedLM
        import torch

        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
        self.encoder = UniversalSentenceEncoder()
        check_parameters(self.config.keys(), DEFAULT_CONFIG)

        self.tokenizer_mlm = BertTokenizer.from_pretrained(self.config['mlm_path'], do_lower_case=True)
        if self.config["device"] is not None:
            self.device = self.config['device']
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        config_atk = BertConfig.from_pretrained(self.config['mlm_path'])
        self.mlm_model = BertForMaskedLM.from_pretrained(self.config['mlm_path'], config=config_atk).to(self.device)
        self.k = self.config['k']
        self.threshold_pred_score = self.config['threshold_pred_score']
        self.max_length = self.config['max_length']
        self.batch_size = self.config['batch_size']

        self.replace_rate = self.config['replace_rate']
        self.insert_rate = self.config['insert_rate']
        if self.replace_rate == 1.0 and self.insert_rate == 0.0:
            self.sub_mode = 0 # only using replacement
        elif self.replace_rate == 0.0 and self.insert_rate == 1.0:
            self.sub_mode = 1 # only using insertion
        elif self.replace_rate + self.insert_rate == 1.0:
            self.sub_mode = 2 # replacement OR insertion for each token / subword
        elif self.replace_rate == 1.0 and self.insert_rate == 1.0:
            self.sub_mode = 3 # first replacement AND then insertion for each token / subword
        else:
            raise NotImplementedError

    def __call__(self, clsf, x_orig, target=None):
        import torch

        x_orig = x_orig.lower()
        if target is None:
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

        sub_words = ['[CLS]'] + sub_words[:max_length - 2] + ['[SEP]']
       
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

        offset = 0
        for top_index in list_of_index:
            if feature.change > int(0.2 * (len(words))):
                feature.success = 1  # exceed
                return None

            tgt_word = words[top_index[0]]
            if tgt_word in filter_words:
                continue

            substitutes = word_predictions[keys[top_index[0]][0]:keys[top_index[0]][1]]  # L, k
            word_pred_scores = word_pred_scores_all[keys[top_index[0]][0]:keys[top_index[0]][1]]

            # in the substitute function, masked_index = top_index[0] + 1, because "[CLS]" has been inserted into sub_words
            replace_sub_len, insert_sub_len = 0, 0
            temp_sub_mode = -1
            if self.sub_mode == 0:
                substitutes = self.get_substitues(top_index[0] + 1, sub_words, tokenizer, self.mlm_model, 'r', self.k, self.threshold_pred_score)
            elif self.sub_mode == 1:
                substitutes = self.get_substitues(top_index[0] + 1, sub_words, tokenizer, self.mlm_model, 'i', self.k, self.threshold_pred_score)
            elif self.sub_mode == 2:
                rand_num = random.random()
                if rand_num < self.replace_rate:
                    substitutes = self.get_substitues(top_index[0] + 1, sub_words, tokenizer, self.mlm_model, 'r', self.k, self.threshold_pred_score)
                    temp_sub_mode = 0
                else:
                    substitutes = self.get_substitues(top_index[0] + 1, sub_words, tokenizer, self.mlm_model, 'i', self.k, self.threshold_pred_score)
                    temp_sub_mode = 1
            elif self.sub_mode == 3:
                substitutes_replace = self.get_substitues(top_index[0] + 1, sub_words, tokenizer, self.mlm_model, 'r', self.k / 2, self.threshold_pred_score) 
                substitutes_insert = self.get_substitues(top_index[0] + 1, sub_words, tokenizer, self.mlm_model, 'i', self.k - self.k / 2, self.threshold_pred_score) 
                replace_sub_len, insert_sub_len = len(substitutes_replace), len(substitutes_insert)
                substitutes = substitutes_replace + substitutes_insert
            else:
                raise NotImplementedError

            most_gap = 0.0
            candidate = None
            
            for i, substitute in enumerate(substitutes):
                if substitute == tgt_word:
                    continue  # filter out original word
                if '##' in substitute:
                    continue  # filter out sub-word
                if substitute in filter_words:
                    continue

                if self.sub_mode == 3:
                    if i < replace_sub_len:
                        temp_sub_mode = 0
                    else:
                        temp_sub_mode = 1
                temp_replace = final_words
                
                # Check if we should REPLACE or INSERT the substitute into the orignal word list 
                is_replace = self.sub_mode == 0 or temp_sub_mode == 0 
                is_insert = self.sub_mode == 1 or temp_sub_mode == 1 
                if is_replace:
                    orig_word = temp_replace[top_index[0]]
                    pos_tagger = DataManager.load("TProcess.NLTKPerceptronPosTagger")
                    pos_tag_list_before = [elem[1] for elem in pos_tagger(temp_replace)]
                    temp_replace[top_index[0]] = substitute
                    pos_tag_list_after = [elem[1] for elem in pos_tagger(temp_replace)]
                    # reverse temp_replace back to its original if pos_tag changes, and continue
                    # searching for the next best substitue
                    if pos_tag_list_after != pos_tag_list_before:
                        temp_replace[top_index[0]] = orig_word
                        continue
                elif is_insert:
                    temp_replace.insert(top_index[0] + offset, substitute)
                else:
                   raise NotImplementedError
    
                temp_text = tokenizer.convert_tokens_to_string(temp_replace)
                
                
                use_score = self.encoder(temp_text, x_orig)
                
                # From TextAttack's implementation: Finally, since the BAE code is based on the TextFooler code, we need to
                # adjust the threshold to account for the missing / pi in the cosine
                # similarity comparison. So the final threshold is 1 - (1 - 0.8) / pi
                # = 1 - (0.2 / pi) = 0.936338023.
                if use_score < 0.936:
                    continue
                inputs = tokenizer.encode_plus(temp_text, None, add_special_tokens=True, max_length=max_length, )
                input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
                seq_len = input_ids.size(1)
                
                temp_prob = torch.Tensor(clsf.get_prob([temp_text]))[0].squeeze()
                feature.query += 1
                temp_prob = torch.softmax(temp_prob, -1)
                temp_label = torch.argmax(temp_prob)

                if (not targeted and temp_label != target) or (targeted and temp_label == target):
                    feature.change += 1
                    if is_replace:
                        final_words[top_index[0]] = substitute
                    elif is_insert:
                        final_words.insert(top_index[0] + offset, substitute)
                    else:
                        raise NotImplementedError
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
                if is_insert:
                    final_words.pop(top_index[0] + offset)

            if most_gap > 0:
                feature.change += 1
                feature.changes.append([keys[top_index[0]][0], candidate, tgt_word])
                current_prob = current_prob - most_gap
                if is_replace:
                    final_words[top_index[0]] = candidate
                elif is_insert:
                    final_words.insert(top_index[0] + offset, candidate) 
                    offset += 1
                else:
                    raise NotImplementedError
            
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
        # masked_words = self._get_masked(words)
        masked_words = self._get_masked_insert(words)
        texts = [' '.join(words) for words in masked_words]  # list of text of masked words
        leave_1_probs = torch.Tensor(tgt_model.get_prob(texts))
        leave_1_probs = torch.softmax(leave_1_probs, -1)  #
        leave_1_probs_argmax = torch.argmax(leave_1_probs, dim=-1)

        import_scores = (orig_prob
                        - leave_1_probs[:, orig_label]
                        +
                        (leave_1_probs_argmax != orig_label).float()
                        * (leave_1_probs.max(dim=-1)[0] - torch.index_select(orig_probs, 0, leave_1_probs_argmax))
                        ).data.cpu().numpy()

        return import_scores

    ##### TODO: make this one of the substitute unit under ./substitures #####
    def get_substitues(self, masked_index, tokens, tokenizer, model, sub_mode, k, threshold=3.0):
        import torch
        masked_tokens = copy.deepcopy(tokens)

        if sub_mode == "r":
            masked_tokens[masked_index] = '[MASK]'
            
        elif sub_mode == "i":
            masked_tokens.insert(masked_index, '[MASK]')
        else:
            raise NotImplementedError
        
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(masked_tokens)
        # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
        segments_ids = [0] * len(indexed_tokens)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)
        segments_tensors = torch.tensor([segments_ids]).to(self.device)
      
        model.eval()

        # Predict all tokens
        with torch.no_grad():
            outputs = model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = outputs[0]

        predicted_indices = torch.topk(predictions[0, masked_index], self.k)[1]
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)
        return predicted_tokens
    
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


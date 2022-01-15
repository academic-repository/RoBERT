import random
from .base import WordSubstitute
from ..data_manager import DataManager
from ..exceptions import WordNotInDictionaryException
import json
from pypinyin import lazy_pinyin


class ChineseGlyphPronSubstitute(WordSubstitute):
    """
    :Data Requirements: :py:data:`.AttackAssist.CiLin`

    An implementation of :py:class:`.WordSubstitute`.

    Chinese Sememe-based word substitute based CiLin.
    """

    def __init__(self):
        super().__init__()
        #self.cilin_dict = DataManager.load("AttackAssist.CiLin")
        with open('/root/robust-ChineseBERT/process_dict/glyph_sim_chars_confirmed.json','r') as glyph_file:
            self.glyph_dict=json.load(glyph_file)
            
        with open('/root/robust-ChineseBERT/process_dict/pronunciation_dict_non_tone.json','r') as pron_file:
            self.pron_dict=json.load(pron_file)

    def __call__(self, word, pos_tag, threshold=10):
        """
        :param word: the raw word; pos_tag: part of speech of the word, threshold: return top k words.
        :return: The result is a list of tuples, *(substitute, 1)*.
        :rtype: list of tuple
        """
        print(word)
        char_candidate=[]
        # for char in word:
        #     if char not in self.glyph_pron_dict:
        #         print(1111111)
        #         raise WordNotInDictionaryException
        #     sym_char=self.glyph_pron_dict[char]
        #     char_candidate.append(sym_char)
        for char in word:
            if char in self.glyph_dict:
                glyph_can=self.glyph_dict[char]
            else:
                glyph_can=[char]
            try:
                pinyin=lazy_pinyin(char)[0]
                #print(pinyin)
                pron_can=self.pron_dict[pinyin][:10]
            except:
                pron_can=[]
            all_can=glyph_can+pron_can
            all_can="".join(list(set(all_can)))
            #print(all_can)
            char_candidate.append(all_can)

        
        #char_candidate = ['123','78','few']
        sym_words=['']
        for i in range(0, len(char_candidate)):
            tmp=[]
            for j in range(0,len(char_candidate[i])):
                for k in range(0,len(sym_words)):
                    tmp.append(sym_words[k]+char_candidate[i][j])
            sym_words=tmp

        # if word not in self.cilin_dict:
        #     raise WordNotInDictionaryException()
        # sym_words = self.cilin_dict[word]
        
        ret = []
        #ret [('另一方面', 1), ('单方面', 1), ('一端', 1), ('一边', 1), ('一派', 1), ('单', 1), ('一头', 1), ('单向', 1), ('一面', 1)]
        for sym_word in sym_words:
            ret.append((sym_word, 1))
        if threshold is not None:
            ret = ret[:threshold]
        #print(word,ret)
        return ret

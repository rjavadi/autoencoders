import numpy as np
import pandas as pd
from configparser import ConfigParser
import pickle, csv
import string
import os

config_file_name = '../config.ini'

class TextPreprocess(object):
    def __init__(self, inputs_list, swap_model_category=False):
        self.__captions = inputs_list['captions']
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "UNK"}
        self.n_words = 1

    def add_caption(self, caption):
        for word in caption:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] +=1


def remove_punc_list(cap_list):
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    cap_list = [x for x in cap_list if x not in string.punctuation]
    for i in range(len(cap_list)):
        cap_list[i].lower()
        if cap_list[i] in string.digits and int(cap_list[i]) < 10:
            cap_list[i] = digits[int(cap_list[i])]
    return cap_list

def remove_punc_str(cap: str):
    import re
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    translate_dict = dict.fromkeys(string.punctuation, ' ')
    translate_dict.update(dict(zip(string.digits, digits)))
    cap = cap.translate(str.maketrans(translate_dict))
    cap = re.sub(' +', ' ', cap)
    return cap


def clean_text():
    config = ConfigParser()
    config.read(config_file_name)
    print(config.sections())
    attr_file = config.get('data','captions_attr_path')
    file = open(attr_file, 'rb')
    captions_list = pickle.load(file)
    data_frame = pd.DataFrame.from_records(captions_list,
                                        columns=['model_id', 'raw_caption', 'tokenized_caption', 'top_synset',
                                                 'sub_synset', 'top_synset_id', 'sub_synset_id', 'model_attr',
                                                 'text_attr'])
    data_frame['tokenized_caption'] = data_frame['tokenized_caption'].apply(remove_punc_list)
    data_frame.to_csv('captions_processed.csv')
    file.close()




if __name__ == "__main__":
    # config = ConfigParser()
    # config.read(config_file_name)
    # cap_file = config.get('data', 'captions_path')
    print(remove_punc_str("2 apples, 3 bananas, 4 birds...!"))
    cap_csv = pd.read_csv(os.path.join('../data', 'captions_processed.csv'))
    # cap_csv['raw_caption'] = cap_csv['raw_caption'].apply(remove_punc_str)
    cap_csv['raw_label'] = cap_csv['raw_caption']
    cap_csv.to_csv('../data/captions_processed.csv')

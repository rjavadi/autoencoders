import numpy as np
import pandas as pd
from configparser import ConfigParser
import pickle, csv
import string

config_file_name = '../config.ini'

class TextPreprocess(object):
    def __init__(self, inputs_list, swap_model_category=False):
        self.__captions = inputs_list['captions']
        self.__word_to_idx = inputs_list['word_to_idx']
        self.__idx_to_word = inputs_list['idx_to_word']
        self.__max_caption_length = (inputs_list['max_length'] if inputs_list['max_length'] != 0
                                     else self.compute_max_caption_length())
        self.__vocab_size = len(self.__word_to_idx) + 1
        self.__input_shape = (self.__max_caption_length)
        self.__dtype = np.int32
        self.__swap_model_category = swap_model_category

        self.print_dataset_info()

def remove_punc(cap_list):
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    cap_list = [x for x in cap_list if x not in string.punctuation]
    for i in range(len(cap_list)):
        cap_list[i].lower()
        if cap_list[i] in string.digits and int(cap_list[i]) < 10:
            cap_list[i] = digits[int(cap_list[i])]
    return cap_list

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
    data_frame['tokenized_caption'] = data_frame['tokenized_caption'].apply(remove_punc)
    data_frame.to_csv('captions_processed.csv')
    file.close()


if __name__ == "__main__":
    clean_text()

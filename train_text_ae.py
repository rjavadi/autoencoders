import os
import math
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.optim as optim
import numpy as np
import argparse
from configparser import ConfigParser
from model.pointnet_encoder import PCAE
import spacy
from torchtext import data
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch import nn
from model.text_ae import Encoder, Decoder, Seq2Seq, Attention

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
MAX_LENGTH = 56
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-4

def caption_file():
    config = ConfigParser()
    config.read(config_file_name)
    return config.get('data', 'captions_path')

def remove_zero_len_char(cap_list):
    cap_list = [x for x in cap_list if len(x) > 0]
    return cap_list

config_file_name = 'config.ini'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create Field object
TEXT = data.Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',
            eos_token = '<eos>')
LABEL = data.Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',
            eos_token = '<eos>', is_target=True)


# df = pd.read_csv(os.path.join('data', 'captions_processed.csv'))
# train, test = train_test_split(df, test_size=0.25, random_state=42)
# fig = plt.figure(figsize=(8, 4))
# sns.barplot(x = train['top_synset'].unique(), y = train['top_synset'].value_counts())
# plt.show()
# nlp = spacy.load('en_core_web_sm')




# Specify Fields in our dataset
fields = [('model_id', None), ('raw_caption', TEXT), ('tokenized_caption', None),
          ('top_syset', None), ('sub_synset', None), ('top_syset_id', None), ('sub_synset_id', None),
          ('model_attr', None), ('text_attr', None), ('raw_label', LABEL)]
caption_data = data.TabularDataset(path=caption_file(), format='csv', fields=fields)
TEXT.build_vocab(caption_data,
                max_size=30_000,
                min_freq=1)

LABEL.build_vocab(caption_data,
                max_size=30_000,
                min_freq=1)
train_data, test_data = caption_data.split()
train_data, val_data = train_data.split()

# Create a set of iterators for each split
train_iterator, val_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x:len(x.input_sequence),
    device = device)

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
#TODO: change to 1024
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ATTN_DIM = 64
# SET TO ZERO FOR NOW
ENC_DROPOUT = 0.0
DEC_DROPOUT = 0.0


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, drop_out_rate=ENC_DROPOUT)
attention = Attention(ENC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, attention, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

def init_weights(module: nn.Module):
    for name, param in module.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


model.apply(init_weights)
optimizer = optim.Adam(model.parameters())

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')
PAD_IDX = LABEL.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# def load_dataset():
#     spacy_en = spacy.load('en')

# def train(input_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len=MAX_LENGTH):
#     encoder_hidden = encoder.initHidden()
#
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#
#     input_len = input_tensor.size(0)
#     target_len = input_len
#
#     encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)
#     loss = 0
#
#     for cap in range(input_len):
#         encoder_output, encoder_hidden = encoder(input_tensor[cap], encoder_hidden)
#         encoder_outputs[cap] = encoder_output[0, 0]


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import torch
import torch.optim as optim
import numpy as np
import argparse
from configparser import ConfigParser
from model.pointnet_encoder import PCAE
from torchtext import data
from tqdm import tqdm

MAX_LENGTH = 56
BATCH_SIZE = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create Field object
TEXT = data.Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',
            eos_token = '<eos>')
config_file_name = 'config.ini'

def caption_file():
    config = ConfigParser()
    config.read(config_file_name)
    return config.get('data', 'captions_path')



# Specify Fields in our dataset
fields = [('model_id', TEXT), ('synset_id', TEXT), ('description', TEXT)]
caption_data = data.TabularDataset(path=caption_file(), format='csv', fields=fields)
TEXT.build_vocab(caption_data,
                max_size=30_000,
                min_freq=2,
                vectors='glove.6B.300d',
                unk_init=torch.Tensor.normal_)
train_data, test_data = caption_data.split()
train_data, val_data = train_data.split()

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x:len(x.input_sequence),
    device = device)
print(caption_data)

def train(input_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_len=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_len = input_tensor.size(0)
    target_len = input_len

    encoder_outputs = torch.zeros(max_len, encoder.hidden_size, device=device)
    loss = 0

    for cap in range(input_len):
        encoder_output, encoder_hidden = encoder(input_tensor[cap], encoder_hidden)
        encoder_outputs[cap] = encoder_output[0, 0]


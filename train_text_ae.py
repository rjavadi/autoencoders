import datetime
import math
import os
import time
from configparser import ConfigParser

import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, RawField, BucketIterator, TabularDataset

from model.text_ae import Encoder, Decoder, Seq2Seq, Attention

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
MAX_LENGTH = 60
BATCH_SIZE = 128
LR = 5e-4

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
LABEL = Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',
            eos_token = '<eos>', is_target=True)

TEXT = Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',
            eos_token = '<eos>')
MODEL_ID = RawField()



# Specify Fields in our dataset
fields = [('model_id', MODEL_ID), ('raw_caption', TEXT),     ('top_syset', None), ('sub_synset', None), ('top_syset_id', None), ('sub_synset_id', None),
          ('raw_label', LABEL)]
caption_data = TabularDataset(path=caption_file(), format='csv', fields=fields, skip_header=True)
TEXT.build_vocab(caption_data,
                max_size=30_000,
                min_freq=1)

LABEL.build_vocab(caption_data,
                max_size=30_000,
                min_freq=1)


# with open('vocab.json', 'w') as fp:
#     json.dump(TEXT.vocab.stoi, fp)
train_data, test_data = caption_data.split(split_ratio=0.90)
train_data, val_data = train_data.split()

# Create a set of iterators for each split
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x:len(x.raw_caption),
    device = device)

INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(LABEL.vocab)
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
#TODO: change to 1024
ENC_HID_DIM = 64
DEC_HID_DIM = 64
ATTN_DIM = 8
ENC_DROPOUT = 0.3
DEC_DROPOUT = 0.3


enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, drop_out_rate=ENC_DROPOUT)
attention = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, attention, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)

# we will initialize all biases to zero and all weights from N(0, 0.01).
def init_weights(module: nn.Module):
    for name, param in module.named_parameters():
        # if name in ['weight']:
        nn.init.normal_(param.data, mean=0, std=0.11)
        # else:
        #     nn.init.constant_(param.data, 0)


model.apply(init_weights)
print("weights: ", list(model.encoder.named_parameters()))

optimizer = optim.Adam(model.parameters())

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')
print(model)

# when scoring the performance of a language translation model in particular, we have to tell the
# nn.CrossEntropyLoss function to ignore the indices where the target is simply padding.
PAD_IDX = LABEL.vocab.stoi['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H:%M:%S")
summary_writer = SummaryWriter(log_dir='logs/text-ae-v2/' + current_time)


def train(model: nn.Module, iterator: BucketIterator, optimizer, criterion: nn.Module, clip: float, current_epoch: int):
    model.train()
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.raw_caption[0]
        trg = batch.raw_label[0]

        optimizer.zero_grad()

        output = model(src, trg)

        # trg = [trg sent len, batch size]
        # output = [trg sent len, batch size, output dim]

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        # trg = [(trg sent len - 1) * batch size]
        # output = [(trg sent len - 1) * batch size, output dim]
        loss = criterion(output, trg)
        step = i + current_epoch * len(iterator)
        print('training epoch: ', epoch, ', step: ', step , '  *****')
        epoch_loss += loss.item()
        summary_writer.add_scalar('train-loss', epoch_loss/(step+1), step+1)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return epoch_loss / len(iterator)

def eval(model: Seq2Seq, iterator: BucketIterator, criterion: nn.Module, current_epoch: int, is_test=False):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.raw_caption[0]
            trg = batch.raw_label[0]

            output = model(src, trg, 0) # turn off teacher forcing

            if is_test:
                emb = model.encoder.forward(src)
                # print("***EMB***:  ", emb)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

            step = i + current_epoch * len(iterator)
            print('validation epoch: ', epoch_loss / (step+1), ', step: ', step+1, '  *****', 'val loss: ', loss)

            if not is_test:
                summary_writer.add_scalar('val-loss', epoch_loss / (step+1), step+1)

    return epoch_loss / len(iterator)

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = elapsed_time // 60
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

MODEL_PATH = 'seq2seq-model.pt'

N_EPOCHS = 8
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP, epoch)
    valid_loss = eval(model, val_iterator, criterion, epoch)

    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_PATH)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    summary_writer.add_scalar('epoch avg. train loss', train_loss, epoch)
    summary_writer.add_scalar('epoch avg. valid loss', valid_loss, epoch)
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}')

test_loss = eval(model, test_iterator, criterion, 0, is_test=True)
summary_writer.add_scalar('test_loss', test_loss)

print(f'| Test Loss: {test_loss:.3f} |')



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


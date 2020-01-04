
from __future__ import unicode_literals, print_function, division

import random
from configparser import ConfigParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, RawField, BucketIterator, TabularDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 2
EOS_token = 3

config_file_name = 'config.ini'

def caption_file():
    config = ConfigParser()
    config.read(config_file_name)
    return config.get('data', 'captions_path')





MAX_LENGTH = 60
BATCH_SIZE = 128
def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]





LABEL = Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',
            eos_token = '<eos>', is_target=True, batch_first=True)

TEXT = Field(tokenize = 'spacy', lower=True, include_lengths = True, init_token = '<sos>',
            eos_token = '<eos>', batch_first=True)
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
train_data, test_data = caption_data.split(split_ratio=0.75)
train_data, val_data = train_data.split()

# Create a set of iterators for each split
train_iterator, val_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size = BATCH_SIZE,
    sort_within_batch = True,
    sort_key = lambda x:len(x.raw_caption),
    device = device)



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # embedding_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        # input = [batch size]
        embedded = self.embedding(input).view(BATCH_SIZE, -1, hidden_size)
        # embedded = [batch size, 1,emb dim]

        output = embedded
        output, hidden = self.gru(output, hidden)
        # outputs = [src sent len, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)



class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(BATCH_SIZE, 1, -1)
        # input = [B, 1, 1]
        # embedded = [batch size, 1, hid_size]
        # hidden = [batch size, 1, hid_size]
        embedded = self.dropout(embedded)

        # TODO Not sure how this concat works
        # TODO: Fix this exception: Sizes of tensors must match except in dimension 1. Got 128 and 1 in dimension 0
        # hidden = [B, 1, hid_size]
        # embedded = [B, 1, hid_size] TODO: squeeze embed and hidden
        hidden = hidden.squeeze()
        embedded = embedded.squeeze()
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden), 1)), dim=1) #attn_weights = [B, Max_len]

        # encoder_outputs = torch.Size([B, Max_len, 320])


        attn_applied = torch.bmm(attn_weights.unsqueeze(1), #attn_weights = [B, 1, Max_len]
                                 encoder_outputs) # encoder_outputs = torch.Size([B, Max_len, 320])
        # attn_applied = [B, 1, 320]
        attn_applied = attn_applied.squeeze()
        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_combine(output)
        #output[0] = [-3.3558e-01, -3.9008e-01, -3.4977e-01, -1.7140e-03,  2.9517e-01,
        output = F.relu(output) # inaja matrix output kheili sparse mishe :(

        # TODO: TA inja dorost kardam. edame bede!!!!!!!
        # Expected hidden size (1, 1, 320), got (128, 1, 320)
        output, hidden = self.gru(output.unsqueeze(1), hidden.unsqueeze(0))
        # output[0] = [-4.0288e-02, -1.5680e-01, -5.6202e-02,  ..., -1.4169e-01,
        #            1.9721e-02, -2.9662e-01]
        output = F.log_softmax(self.out(output), dim=2) #output=[B, 1, hid] - #self.out(output[0]) = [B, 1, Vocab]
        return output.squeeze(), hidden, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)




def indexesFromSentence(field, sentence):
    return [field.stoi[word] for word in sentence.split(' ')]


def tensorFromSentence(field, sentence):
    indexes = indexesFromSentence(field, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(TEXT, pair[0])
    target_tensor = tensorFromSentence(LABEL, pair[1])
    return (input_tensor, target_tensor)



teacher_forcing_ratio = 0.5

import datetime
current_time = str(datetime.datetime.now())
summary_writer = SummaryWriter(log_dir='logs/' + current_time)

def train(iterator: BucketIterator, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder.train()
    decoder.train()
    epoch_loss = 0
    count = 0
    encoder_hidden = encoder.initHidden(BATCH_SIZE)
    for i, batch in enumerate(iterator):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        src = batch.raw_caption[0] #[B, src_len]
        trg = batch.raw_label[0] #[B, trg_len]

        input_length = src.size(1)
        target_length = trg.size(1)

        encoder_outputs = torch.zeros(BATCH_SIZE, max_length, encoder.hidden_size, device=device) #[B, 60, hid]
        encoder_output = torch.zeros(BATCH_SIZE, 1, encoder.hidden_size, device=device)
        loss = 0
        # TODO: moshkel ine ke input_len 70 hast, vali ma goftim max_length=60. bayad voroud ha ro cut va filter konim.
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                src[:, ei], encoder_hidden)
            encoder_outputs[:, ei] = encoder_output[:, 0] #[128, 60, 320]

        decoder_input = torch.tensor([[SOS_token]], device=device).repeat(128, 1, 1)
        # encoder_hidden = [B, 1, hid_size]
        decoder_hidden = encoder_output #[B, 1, hid_size]


        for di in range(target_length):
            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if use_teacher_forcing:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, trg[:, di]) #trg=[B, len] , decoder_output = [B, 1, vocab]
                decoder_input = trg[:, di]  # Teacher forcing
            else:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # topv = [[-9.0280],[-9.0581],[-9.0583],[-9.0091],[-9.0417],[-9.0071],....]
                #         [-9.0256],....] [128, 1]
                topv, topi = decoder_output.topk(1) #topi = [1306, 8360, 2019, 2019, 2019, 2019, 5037, 5037,...]
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, trg[:, di])
                # if decoder_input.item() == EOS_token:
                #     break
                print('******', count)

        count += 1

        loss.backward(retain_graph=True)
        encoder_optimizer.step()
        decoder_optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def eval(iterator: BucketIterator, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, is_test=False):
    encoder.eval()
    decoder.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.raw_caption[0]
            trg = batch.raw_label[0]

            input_length = src.size(1)
            target_length = trg.size(1)

            encoder_outputs = torch.zeros(BATCH_SIZE, max_length, encoder.hidden_size, device=device)
            encoder_output = torch.zeros(BATCH_SIZE, 1, encoder.hidden_size, device=device)

            loss = 0
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(
                    src[:, ei], encoder_hidden)
                encoder_outputs[:, ei] = encoder_output[:, 0]

            decoder_input = torch.tensor([[SOS_token]], device=device).repeat(128, 1, 1)

            decoder_hidden = encoder_output
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, trg[:, di])
                # if decoder_input.item() == EOS_token:
                #     break

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def epoch_time(start_time: int, end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = elapsed_time // 60
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.

N_EPOCHS = 8

def trainIters(encoder, decoder, n_epochs = N_EPOCHS, print_every=1000, plot_every=100, learning_rate=0.01):

    ENCODER_MODEL = 'seq2seq-enc.pt'
    DECODER_MODEL = 'seq2seq-dec.pt'

    best_valid_loss = float('inf')

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    for epoch in range(n_epochs):
        start_time = time.time()

        criterion = nn.NLLLoss()
        train_loss = train(train_iterator, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        valid_loss = eval(val_iterator, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        summary_writer.add_scalar('train-loss', train_loss, epoch)
        summary_writer.add_scalar('val-loss', valid_loss, epoch)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(encoder.state_dict(), ENCODER_MODEL)
            torch.save(encoder.state_dict(), DECODER_MODEL)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')




######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#



def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(TEXT, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden(BATCH_SIZE)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(LABEL.vocab.itos[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, n=10):
    pairs = test_iterator.data
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################
# Training and Evaluating
# =======================
#
# With all these helper functions in place (it looks like extra work, but
# it makes it easier to run multiple experiments) we can actually
# initialize a network and start training.
#
# Remember that the input sentences were heavily filtered. For this small
# dataset we can use relatively small networks of 256 hidden nodes and a
# single GRU layer. After about 40 minutes on a MacBook CPU we'll get some
# reasonable results.
#
# .. Note::
#    If you run this notebook you can train, interrupt the kernel,
#    evaluate, and continue training later. Comment out the lines where the
#    encoder and decoder are initialized and run ``trainIters`` again.
#

hidden_size = 320
encoder1 = EncoderRNN(len(TEXT.vocab), hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, len(LABEL.vocab), dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1)

######################################################################
#

evaluateRandomly(encoder1, attn_decoder1)


######################################################################
# Visualizing Attention
# ---------------------
#
# A useful property of the attention mechanism is its highly interpretable
# outputs. Because it is used to weight specific encoder outputs of the
# input sequence, we can imagine looking where the network is focused most
# at each time step.
#
# You could simply run ``plt.matshow(attentions)`` to see attention output
# displayed as a matrix, with the columns being input steps and rows being
# output steps:
#

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "an orange colored chair with a black frame and arm rests")
plt.matshow(attentions.numpy())


######################################################################
# For a better viewing experience we will do the extra work of adding axes
# and labels:
#

# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)
#
#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)
#
#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
#     plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    # showAttention(input_sentence, output_words, attentions)


evaluateAndShowAttention("half egg chair red in color with black base ")

evaluateAndShowAttention("cabinet is tall made of wood")

evaluateAndShowAttention("this is a gray lamp with four bulbs")

evaluateAndShowAttention("a grey chair with one cushion and wooden arm rests that wall off into solid grey squares particularly box like in appearance")


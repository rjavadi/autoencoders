import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, drop_out_rate=0.0):

        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.drop_out_rate = drop_out_rate
        # Embedding layer that will be shared with Decoder
        self.embedding = nn.Embedding(input_size, embedding_size)
        # Bidirectional GRU
        self.gru = nn.GRU(embedding_size, hidden_size, dropout=drop_out_rate, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.drop_out = nn.Dropout(drop_out_rate)

    def forward(self, input):
        # input = [input sent len, batch size]
        word_embeddings = self.drop_out(self.embedding(input))
        # word_embeddings = [src sent len, batch size, emb dim]

        outputs, hidden = self.gru(word_embeddings)
        # outputs = [src sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        # Note: torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        # is of shape [batch_size, enc_hid_dim * 2]
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        # outputs = [src sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        return outputs, hidden



class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim, attn_dim):
        super(Attention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, dec_hidden, encoder_outputs):
        # dec_hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        # repeat decoder hidden state src_len times
        repeated_decoder_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # decoder_hidden = [batch size, src sent len, dec hid dim]
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]

        # Step 1: to enable feeding through "self.attn" pink box above, concatenate
        # `repeated_decoder_hidden` and `encoder_outputs`:
        # torch.cat((hidden, encoder_outputs), dim = 2) has shape
        # [batch_size, seq_len, enc_hid_dim * 2 + dec_hid_dim]

        # Step 2: feed through self.attn to end up with:
        # [batch_size, seq_len, attn_dim]

        # Step 3: feed through tanh
        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim=2)))
        # energy = [batch size, src sent len, attn_dim]

        attention = torch.sum(energy, dim=2)
        # attention= [batch size, src len]

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_size, emb_dim, enc_hid_dim, dec_hid_dim, attention, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_size = output_size
        self.attention = attention
        # Note: from Attention: self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, attn_dim)

        # Note: `output_dim` same as `vocab_size`
        self.embedding = nn.Embedding(output_size, emb_dim)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_size)
        self.dropout = nn.Dropout(dropout_rate)


    def _weighted_encoder_rep(self, decoder_hidden: Tensor, encoder_outputs: Tensor):
        # Attention, at a high level, takes in:
        # The decoder hidden state
        # All the "seq_len" encoder outputs
        # Outputs a vector summing to 1 of length seq_len for each observation

        a = self.attention(decoder_hidden, encoder_outputs)
        # a = [batch size, src len]
        a = a.unsqueeze(1)
        # a = [batch size, 1, src len]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        # weighted_encoder_rep = [batch size, 1, enc hid dim * 2]
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep



    # returns a tuple of tensors
    def forward(self, input, decoder_hidden, encoder_outputs):
        # input = [batch size] Note: "one character at a time"
        # decoder_hidden = [batch size, dec hid dim]
        # encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        weighted_enc_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
        # Then, the input to the decoder _for this character_ is a concatenation of:
        # This weighted attention
        # The embedding itself
        rnn_input = torch.cat((embedded, weighted_enc_rep), dim=2)
        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        # output = [sent len, batch size, dec hid dim * n directions]
        # decoder_hidden = [n layers * n directions, batch size, dec hid dim]

        # sent len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_enc_rep = weighted_enc_rep.squeeze(0)
        output = self.out(torch.cat((output, weighted_enc_rep, embedded), dim=1))
        # output = [batch, output dim]
        return output, decoder_hidden.squeeze(0)



"""encoder returns both the final hidden state (which is the final hidden state from both the forward and backward encoder RNNs passed through a linear layer) to be used as the initial hidden state for the encoder, as well as every hidden state (which are the forward and backward hidden states stacked on top of each other). We also need to ensure that hidden and encoder_outputs are passed to the decoder."""
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        # Embedding layer shared by encoder and decoder

        # Encoder network
        self.encoder = encoder

        # Decoder network
        self.decoder = decoder
        self.device = device

        # Indices of special tokens and hardware device


    def forward(self, input_sequence, output_sequence, teacher_forcing_ratio=0.5):
        # input_seq = [input_seq sent len, batch size]
        # output_seq = [output_seq sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        # Unpack input_sequence tuple
        batch_size = input_sequence.shape[1]
        max_len = output_sequence.shape[0]

        vocab_size = self.decoder.output_size
        # tensor to store decoder outputs
        dec_outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(input_sequence)
        # first input to the decoder is the <sos> token
        output = output_sequence[0, :]

        #TODO: I don't understand
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            dec_outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (output_sequence[t] if teacher_force else top1)

        return dec_outputs


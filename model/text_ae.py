import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# https://github.com/keon/seq2seq/blob/master/model.py

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

    def forward(self, input, input_lengths):
        word_embeddings = self.drop_out(self.embedding(input))

        outputs, hidden = self.gru(word_embeddings)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        return outputs, hidden



class Attention(nn.Module):
    def __init__(self, hidden_size, attn_dim):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn_in = hidden_size * 3
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def dot_score(self, hidden_state, encoder_state):
        return torch.sum(hidden_state * encoder_state, dim=2)

    def forward(self, dec_hidden, encoder_outputs, mask):
        src_len = encoder_outputs.shape[0]

        repeated_decoder_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((
            repeated_decoder_hidden,
            encoder_outputs),
            dim=2)))

        attention = torch.sum(energy, dim=2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_size, embedding_size, enc_hid_dim, dec_hid_dim, attention, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_size = output_size
        self.attention = attention
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.GRU((enc_hid_dim * 2) + embedding_size, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + embedding_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)


    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs):

        a = self.attention(decoder_hidden, encoder_outputs)

        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted_encoder_rep = torch.bmm(a, encoder_outputs)

        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)

        return weighted_encoder_rep



    # returns a tuple of tensors
    def forward(self, input, decoder_hidden, encoder_outputs):

        embedded = self.dropout(self.embedding(input))
        weighted_enc_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted_enc_rep), dim=2)
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted_enc_rep = weighted_enc_rep.squeeze(0)
        output = self.out(torch.cat((output, weighted_enc_rep, embedded), dim=1))
        return output, decoder_hidden.squeeze(0)


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
    def create_mask(self, input_sequence):
        # Permute the dimensions of this tensor.
        return (input_sequence != self.pad_idx).permute(1, 0)

    def forward(self, input_sequence, output_sequence, teacher_forcing_ratio=0.5):
        # Unpack input_sequence tuple
        batch_size = input_sequence.shape[1]
        max_len = output_sequence.shape[0]

        vocab_size = self.decoder.output_size
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(input_sequence)
        hidden = hidden[:self.decoder.n_layers]
        # first input to the decoder is the <sos> token
        output = output_sequence[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            output = (output_sequence[t] if teacher_force else top1)

        return outputs


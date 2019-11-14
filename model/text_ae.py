import torch
import torch.nn as nn
import torch.nn.functional as F


#https://medium.com/@adam.wearne/seq2seq-with-pytorch-46dc00ff5164

class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, embedding, num_layers=2, drop_out=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.drop_out = drop_out
        self.embedding = embedding
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, dropout=drop_out, bidirectional=True)


    def forward(self, input, input_lengths):
        word_embeddings = self.embedding(input)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(word_embeddings, input_lengths)
        outputs, hidden = self.gru(packed_embeddings)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # The output of a GRU has shape (seq_len, batch, hidden_size * num_directions)
        # Because the Encoder is bidirectional, combine the results from the
        # forward and reversed sequence by simply adding them together.
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        return outputs, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size


    def dot_score(self, hidden_state, encoder_state):
        return torch.sum(hidden_state * encoder_state, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        attn_scores = self.dot_score(hidden, encoder_outputs)
        # Transpose max_length and batch_size dimensions
        attn_scores = attn_scores.t()
        # Apply mask so network does not attend <pad> tokens
        attn_scores = attn_scores.masked_fill(mask == 0, -1e10)
        # Return softmax over attention scores
        return F.softmax(attn_scores, dim=1).unsqueeze(1)




class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


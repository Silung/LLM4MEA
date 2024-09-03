from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class GRU4REC(nn.Module):
    def __init__(self, args):
        super(GRU4REC, self).__init__()
        self.args = args
        self.input_size = args.num_items + 1
        self.hidden_size = 2 * args.bert_hidden_units
        self.output_size = self.input_size
        self.num_layers = 1
        self.dropout_hidden = 0.5
        self.dropout_input = 0
        self.embedding_dim = args.bert_hidden_units
        # self.embedding_dim = -1
        self.device = args.device
        self.onehot_buffer = self.init_emb()
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        self.create_final_activation('tanh')
        if self.embedding_dim != -1:
            self.look_up = nn.Embedding(self.input_size, self.embedding_dim)
            self.gru = nn.GRU(self.embedding_dim, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        else:
            self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout_hidden)
        self = self.to(self.device)

    def create_final_activation(self, final_act):
        if final_act == 'tanh':
            self.final_activation = nn.Tanh()
        elif final_act == 'relu':
            self.final_activation = nn.ReLU()
        elif final_act == 'softmax':
            self.final_activation = nn.Softmax()
        elif final_act == 'softmax_logit':
            self.final_activation = nn.LogSoftmax()
        elif final_act.startswith('elu-'):
            self.final_activation = nn.ELU(alpha=float(final_act.split('-')[1]))
        elif final_act.startswith('leaky-'):
            self.final_activation = nn.LeakyReLU(negative_slope=float(final_act.split('-')[1]))

    # def forward(self, input, hidden):
    def forward(self, x, lengths):
        x = self.look_up(x)
            
        x = pack_padded_sequence(x, lengths.to('cpu'), batch_first=True, enforce_sorted=False)
        output, hidden = self.gru(x) #(num_layer, B, H)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = output[range(len(lengths)), lengths-1]
        # output = output.view(-1, output.size(-1))  #(B,H)
        logit = self.final_activation(self.h2o(output))

        return logit

    def init_emb(self):
        '''
        Initialize the one_hot embedding buffer, which will be used for producing the one-hot embeddings efficiently
        '''
        onehot_buffer = torch.FloatTensor(self.args.batch_size, self.args.bert_max_len)
        onehot_buffer = onehot_buffer.to(self.device)
        return onehot_buffer

    def embedding_dropout(self, input):
        p_drop = torch.Tensor(input.size(0), 1).fill_(1 - self.dropout_input)
        mask = torch.bernoulli(p_drop).expand_as(input) / (1 - self.dropout_input)
        mask = mask.to(self.device)
        input = input * mask
        return input

    def init_hidden(self):
        '''
        Initialize the hidden state of the GRU
        '''
        try:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
        except:
            self.device = 'cpu'
            h0 = torch.zeros(self.num_layers, self.args.batch_size, self.hidden_size).to(self.device)
        return h0
import torch
import torch.nn as nn


class rnn(nn.Module):
    def __init__(self, hidden_dim):
        super(rnn, self).__init__()
        self.w1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.active = nn.Tanh()

    def forward(self, input, hidden):
        for idx in range(input.size(0)):
            hidden = self.active(self.w1(torch.cat((input[idx, :, :], hidden), dim=1))) #배치사이즈, 히든사이즈*2 --> 배치사이즈 히든사이즈
            if idx == 0:
                output = hidden
                output = output.unsqueeze(0)
            else:
                output = torch.cat((output, hidden.unsqueeze(0)), dim=0)
        return output, hidden


class rnn_model(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(rnn_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = rnn(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, batch_size):
        encoded = self.encoder(input)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        output, hidden = self.rnn(encoded, hidden)
        decode = self.decoder(output)
        decode = decode.view(-1, decode.size(-1))
        return decode, hidden


class lstm(nn.Module):
    def __init__(self, hidden_dim):
        super(lstm, self).__init__()
        self.w1 = nn.Linear(hidden_dim * 2, hidden_dim*4)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.hidden_dim = hidden_dim

    def forward(self, input, hidden,cell):
        for idx in range(input.size(0)):
            hidden =self.w1(torch.cat((input[idx, :, :], hidden), dim=1))
            forget_gate = self.sigmoid(hidden[:,:self.hidden_dim])
            input_gate = self.sigmoid(hidden[:,self.hidden_dim:self.hidden_dim*2])
            gate_gate = self.tanh(hidden[:,self.hidden_dim*2:self.hidden_dim*3])
            output_gate = self.sigmoid(hidden[:,self.hidden_dim*3:])
            cell = cell *  forget_gate + gate_gate * input_gate
            hidden = cell * output_gate
            if idx == 0:
                output = hidden
                output = output.unsqueeze(0)
            else:
                output = torch.cat((output, hidden.unsqueeze(0)), dim=0)
        return output, cell


class lstm_model(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(lstm_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = lstm(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, batch_size):
        encoded = self.encoder(input)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        cell = torch.zeros(batch_size, self.hidden_dim).to(device)
        output, hidden = self.lstm(encoded, hidden,cell)
        decode = self.decoder(output)
        decode = decode.view(-1, decode.size(-1))
        return decode, hidden

class gru(nn.Module):
    def __init__(self, hidden_dim):
        super(gru, self).__init__()
        self.w1 = nn.Linear(hidden_dim * 2, hidden_dim*2)
        self.h_weight = nn.Linear(hidden_dim * 2, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.hidden_dim = hidden_dim

    def forward(self, input, hidden):
        for idx in range(input.size(0)):
            gates =self.w1(torch.cat((input[idx, :, :], hidden), dim=1))
            z_gate = self.sigmoid(gates[:,:self.hidden_dim])
            r_gate = self.sigmoid(gates[:,self.hidden_dim:self.hidden_dim*2])
            h_gate = self.tanh(self.h_weight(torch.cat((r_gate*hidden,input[idx, :, :]),dim=1)))
            hidden = (1 - z_gate) * hidden + z_gate * h_gate
            if idx == 0:
                output = hidden
                output = output.unsqueeze(0)
            else:
                output = torch.cat((output, hidden.unsqueeze(0)), dim=0)
        return output, hidden


class gru_model(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(lstm_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Embedding(vocab_size, hidden_dim)
        self.gru = gru(hidden_dim)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, batch_size):
        encoded = self.encoder(input)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(device)
        output, hidden = self.gru(encoded, hidden)
        decode = self.decoder(output)
        decode = decode.view(-1, decode.size(-1))
        return decode, hidden


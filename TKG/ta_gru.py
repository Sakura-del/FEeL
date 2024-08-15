import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size, device):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.device = device

    def forward(self, ts, hiddens, mask):
        ts = -1 * torch.tensor(ts).to(self.device)
        ts = ts.masked_fill(mask==0,1e10)

        attn_energies = self.score(h, encoder_outputs)
        attn_energies = attn_energies.masked_fill(mask == 0, -1e10)  # Apply mask
        return F.softmax(attn_energies, dim=1)
    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(2, 1)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)


class AttentionGRU(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1,device=0):
        super(AttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(hidden_size * 2, output_size).to(self.device)  # Adjusted size to match concatenated input

    def forward(self, ts, input_lengths, encoder_outputs):
        packed = nn.utils.rnn.pack_padded_sequence(encoder_outputs, input_lengths, batch_first=True, enforce_sorted=False).to(self.device)
        gru_output, hidden = self.gru(packed)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)

        mask = ts != -1
        ts = -ts.masked_fill(mask==0, 1e10)
        attn_weights = F.softmax(ts.float(),dim=-1)

        # attn_weights = self.attention(ts, encoder_outputs, mask)
        context = attn_weights.unsqueeze(1).bmm(gru_output)

        # Concatenate GRU output and context vector, and pass through a linear layer
        output = torch.cat((hidden.squeeze(), context.squeeze(1)), 1)
        output = self.fc(output)

        return output, hidden.squeeze()

    # def init_hidden(self, batch_size):
    #     return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(next(self.parameters()).device)


# # Example usage
# input_size = 10  # Size of the input vocabulary
# hidden_size = 20  # Size of the GRU hidden layer
# output_size = 5  # Size of the output layer
# seq_len = 6  # Length of the input sequences
# batch_size = 3  # Batch size
#
# model = AttentionGRU(hidden_size, hidden_size,device=0)
# input_seq = [[2, 3],[2, 3, 4],[1, 2, 3, 4, 5, 6]]  # Padded sequences
#
#
# input_lengths = torch.tensor([2, 3, 6])  # Actual lengths of the sequences
# encoder_outputs = torch.randn(batch_size, seq_len, hidden_size)
#
# input_seq = [torch.tensor(t) for t in input_seq]
# pad_ts = nn.utils.rnn.pad_sequence(input_seq, batch_first=True, padding_value=-1)
#
# output, hidden = model(pad_ts, input_lengths, encoder_outputs)
#
# print("Output shape:", output.shape)
# print("Hidden shape:", hidden.shape)

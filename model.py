import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import *
import math
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(64, 1, kernel_size=1)
        self.fc1 = nn.Linear(2700,2700)
        self.fc2 = nn.Linear(2700,2700)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x=x.reshape(x.size(0), x.size(1), -1)
        return x

class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class LSTMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMGenerator, self).__init__()

        self.LSTM = nn.LSTM(input_dim, hidden_dim, batch_first=True,dtype=torch.double)#bidirectional=True,hidden_dim要*2
 
        # 添加全连接层
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 128).double()
        self.bn = nn.BatchNorm1d(128).double()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(128, output_dim).double()

    def forward(self, x, h=None):
        # x: (batch, seq, input_dim)

        x, h = self.LSTM(x, h)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.linear(x)
        x = torch.clamp(x, max=2500)
        y_fake = x.reshape(x.size(0), x.size(1), -1)

        return y_fake

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128).double(),
            nn.BatchNorm1d(128, dtype=torch.double),#终于不过拟合了
            nn.LeakyReLU(0.5, inplace=True),
            nn.Linear(128, 1).double(),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, seq)
        return self.model(x)


class LSTMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.LSTM = nn.LSTM(input_dim, hidden_dim, batch_first=True,dtype=torch.double)

        # 添加全连接层
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_dim, 128).double()
        self.bn = nn.BatchNorm1d(128).double()
        self.relu = nn.ReLU()
        self.linear = nn.Linear(128, output_dim).double()

    def forward(self, x, h=None):
        # x: (batch, seq, input_dim)

        x, h = self.LSTM(x, h)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.linear(x)
        y_fake = x.reshape(x.size(0), x.size(1), -1)

        return y_fake

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.LSTM = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                            dtype=torch.double)
        self.linear = nn.Linear(hidden_dim, output_dim).double()

    def forward(self, x, h=None):
        # x: (batch, seq, input_dim)
        x, h = self.LSTM(x, h)
        x = self.linear(x)
        x = torch.clamp(x, max=2500)
        y = x.reshape(x.size(0), x.size(1), -1)

        return y

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True,
                            dtype=torch.double)
        self.linear = nn.Linear(hidden_dim*2, output_dim).double()

    def forward(self, x, h=None):
        # x: (batch, seq, input_dim)
        x, h = self.lstm(x, h)
        x = self.linear(x)
        y = x.reshape(x.size(0), x.size(1), -1)

        return y

class modifiedLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=1).double()
        self.fc1 = nn.Linear(input_dim, input_dim).double()
        self.relu = nn.LeakyReLU(0.5, inplace=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,bidirectional=True,
                            dtype=torch.double)  # bidirectional=True,hidden_dim要*2
        self.bn = nn.BatchNorm1d(2700, dtype=torch.double)

        self.fc2 = nn.Linear(hidden_dim*2, output_dim).double()



    def forward(self, x):
        residual = x

        out=x.permute(0, 2, 1)
        out= self.conv1(out)
        out = out.permute(0, 2, 1)
        out = self.fc1(out)

        out = self.relu(out).clone()

        out += residual
        out, _ = self.lstm(out)
        out = self.bn(out)
        # out = self.ln(out)
        out = self.fc2(out)
        out= out.reshape(x.size(0), x.size(1), -1)

        return out



class LSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=1,
                                               dtype=torch.double)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                            dtype=torch.double)

        self.linear = nn.Linear(hidden_dim, output_dim).double()

    def forward(self, x):
        # Attention
        attn_out, attn_weights = self.attention(x, x, x)
        # LSTM
        lstm_out, _ = self.lstm(attn_out)
        pred= self.linear(lstm_out)


        return  pred, attn_weights



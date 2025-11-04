import torch
import torch.nn as nn
import torch.nn.functional as F

class ParallelCNNBiGRU(nn.Module):
    def __init__(self, num_classes, n_mels=256, rnn_proj=128, rnn_hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 1), padding=(1,0)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 32, kernel_size=(3, 1), padding=(1,0)), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1,0)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding=(1,0)), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 1), padding=(1,0)), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.cnn_out = 128

        self.prepool = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.proj = nn.Linear(n_mels, rnn_proj)
        self.bigru = nn.GRU(
            input_size=rnn_proj, hidden_size=rnn_hidden,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.rnn_out = rnn_hidden * 2  # bidirection


        fused = self.cnn_out + self.rnn_out
        self.head = nn.Sequential(
            nn.Linear(fused, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)  # logits
        )

    def forward(self, x):

        # CNN path
        z_cnn = self.cnn(x)
        z_cnn = z_cnn.view(x.size(0), -1)

        # RNN path
        x_pooled = self.prepool(x)
        x_seq = x_pooled.squeeze(1).transpose(1, 2)
        x_seq = self.proj(x_seq)
        y_seq, _ = self.bigru(x_seq)

        y_mean = y_seq.mean(dim=1)
        y_max  = y_seq.max(dim=1).values
        z_rnn = 0.5 * (y_mean + y_max)

        # Fusion
        z = torch.cat([z_cnn, z_rnn], dim=1)
        logits = self.head(z)
        return logits

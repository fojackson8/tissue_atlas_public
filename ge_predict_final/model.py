import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=(3, 3)):
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer1(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ConvNet(nn.Module):
    def __init__(self, n_epigenetic_cols, n_1kb_bins):
        super(ConvNet, self).__init__()
        self.layer1 = ResidualBlock(1, 32, kernel_size=(3, 5))
        self.layer2 = ResidualBlock(32, 64, kernel_size=(3, 5))
        self.layer3 = ResidualBlock(64, 128, kernel_size=(3, 5))
        self.layer4 = ResidualBlock(128, 256, kernel_size=(3, 5))
        self.fc1 = nn.Linear(256 * n_epigenetic_cols * n_1kb_bins, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class SimpleDNN(nn.Module):
    def __init__(self, input_size, output_size=1):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class HybridConvNetTransformer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, n_epigenetic_cols, n_1kb_bins):
        super(HybridConvNetTransformer, self).__init__()
        self.layer1 = ResidualBlock(1, 32, kernel_size=(3, 5))
        self.layer2 = ResidualBlock(32, 64, kernel_size=(3, 5))
        self.layer3 = ResidualBlock(64, 128, kernel_size=(3, 5))
        self.layer4 = ResidualBlock(128, 256, kernel_size=(3, 5))
        self.fc1 = nn.Linear(256 * n_epigenetic_cols * n_1kb_bins, d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = out.view(out.size(0), -1, self.fc1.out_features)
        out = self.transformer_encoder(out)
        out = out.mean(dim=1)
        out = self.fc_out(out)
        return out

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


            

def get_model(model_type, n_epigenetic_cols, n_1kb_bins, input_size=None, d_model=None, nhead=None, dim_feedforward=None, num_layers=None):
    if model_type == "ConvNet":
        return ConvNet(n_epigenetic_cols, n_1kb_bins)
    elif model_type == "SimpleDNN":
        if input_size is None:
            raise ValueError("input_size must be provided for SimpleDNN model")
        return SimpleDNN(input_size)
    elif model_type == "HybridConvNetTransformer":
        if d_model is None or nhead is None or dim_feedforward is None or num_layers is None:
            raise ValueError("d_model, nhead, dim_feedforward, and num_layers must be provided for HybridConvNetTransformer model")
        return HybridConvNetTransformer(d_model, nhead, dim_feedforward, num_layers, n_epigenetic_cols, n_1kb_bins)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
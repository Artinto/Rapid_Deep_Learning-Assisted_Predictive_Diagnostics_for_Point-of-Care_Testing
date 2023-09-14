from torchvision.models import resnet18, resnet34, resnet50, vgg11, densenet121
from torchvision.models import vgg19
from torch.nn import Module, LSTM, GRU
from torch import nn
import torch






def weights_init(model):
    if isinstance(model, nn.Linear):
        nn.init.kaiming_uniform_(model.weight)


class encoders(Module):
    def __init__(self, img_embedding_size: int=784, img_encoder: str='d-121', seq_model: str='lstm',
                 pretrained: bool=False, drop_rate: float=0, hidden_size: int=784, channel=3):
        super(encoders, self).__init__()
        if img_encoder == 'r-18':
            self.encoder = resnet18(pretrained, num_classes=img_embedding_size)
            self.encoder.conv1 = nn.Conv2d(channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif img_encoder == 'r-34':
            self.encoder = resnet34(pretrained, num_classes=img_embedding_size)
        elif img_encoder == 'r-50':
            self.encoder = resnet50(pretrained, num_classes=img_embedding_size)
        elif img_encoder == 'd-121':
            self.encoder = densenet121(pretrained, num_classes=img_embedding_size)





        if seq_model == 'lstm':
            self.sequential = LSTM(img_embedding_size, hidden_size, batch_first=True, dropout=drop_rate)
        elif seq_model == 'gru':
            self.sequential = GRU(img_embedding_size, hidden_size, batch_first=True, dropout=drop_rate)

    def forward(self, img):
        # img = # b, n_img, 3, h, w
        img_size = img.size()
        img = img.view(-1, *img_size[2:])
        img = self.encoder(img)
        img = img.view(img_size[0], img_size[1], -1)
        _, x = self.sequential(img)         # lstm: out, (hidden, cell) / gru: out, hidden
        hidden = x if len(x) == 1 else x[0]
        # out = b, seq, hidden_size
        hidden = hidden.squeeze(0)
        return hidden


class main_model(Module):
    def __init__(self, config, detach=False):
        super(main_model, self).__init__()
        self.config = config
        self.detach = detach
        self.feature_size = config.latent_size
        self.encoder = encoders(img_embedding_size=config.img_embedding_size,
                                img_encoder=config.img_encoder,
                                seq_model=config.seq_model,
                                pretrained=config.pretrained,
                                drop_rate=config.drop_rate,
                                hidden_size=self.feature_size,
                                channel=3 if config.img_use_hsv != 'concat' else 6)
        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size // 2),
            nn.ReLU(),
            nn.Dropout(p=config.drop_rate)
        )
        # self.fc2 = nn.Linear(self.feature_size//2, 2)
        # self.fc3 = nn.Linear(2, 1)
        self.fc2 = nn.Linear(self.feature_size//2, 1)

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.fc1(hidden)
        out_dst = self.fc2(out)
        # out_dst = self.fc3(out_cls)
        return out_dst.squeeze(1)

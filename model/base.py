import torch.nn as nn

class base(nn.Module):
    def __init__(self, num_points=4):
        super(base, self).__init__()
        hidden1 = block(in_chs= 3, out_chs= 48, k_size=5, pad=2, c_stride= 2, mp_stride= 2)
        hidden2 = block(in_chs= 48, out_chs= 64, k_size=5, pad=2, c_stride= 1, mp_stride= 1)
        hidden3 = block(in_chs= 64, out_chs= 128, k_size=5, pad=2, c_stride= 1, mp_stride= 2)
        hidden4 = block(in_chs= 128, out_chs= 160, k_size=5, pad=2, c_stride= 1, mp_stride= 1)
        hidden5 = block(in_chs= 160, out_chs= 192, k_size=5, pad=2, c_stride= 1, mp_stride= 2)
        hidden6 = block(in_chs= 192, out_chs= 192, k_size=5, pad=2, c_stride= 1, mp_stride= 1)
        hidden7 = block(in_chs= 192, out_chs= 192, k_size=5, pad=2, c_stride= 1, mp_stride= 2)
        hidden8 = block(in_chs= 192, out_chs= 192, k_size=5, pad=2, c_stride= 1, mp_stride= 1)
        hidden9 = block(in_chs= 192, out_chs= 192, k_size=3, pad=1, c_stride= 1, mp_stride= 2)
        hidden10 = block(in_chs= 192, out_chs= 192, k_size=3, pad=1, c_stride= 1, mp_stride= 1)

        self.features = nn.Sequential(
            hidden1,
            hidden2,
            hidden3,
            hidden4,
            hidden5,
            hidden6,
            hidden7,
            hidden8,
            hidden9,
            hidden10
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(23232, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            # nn.ReLU(inplace=True),
            nn.Linear(100, num_points),
        )

    def forward(self, x):
        x1 = self.features(x)
        x11 = x1.view(x1.size(0), -1)
        x = self.classifier(x11)
        return x

    def block(self, in_chs=192, out_chs=192, k_size=3, pad=1, c_stride= 2, mp_stride= 2):
        return nn.Sequential(
            nn.Conv2d(in_channels= in_chs, out_channels= out_chs, kernel_size= k_size, padding= pad, stride= c_stride),
            nn.BatchNorm2d(num_features= out_chs),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size= 2, stride= mp_stride, padding= 1),
            nn.Dropout(0.2)
        )

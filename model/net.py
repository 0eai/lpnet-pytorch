import torch
import torch.nn as nn
from roi_pooling import roi_pooling_ims
#from base import base

class net(nn.Module):
    def __init__(self, num_points, num_classes, basePath=None):
        super(net, self).__init__()
        self.load_base(basePath)
        self.classifier1 = classifier(numClasses)
        self.classifier2 = classifier(numClasses)
        self.classifier3 = classifier(numClasses)
        self.classifier4 = classifier(numClasses)
        self.classifier5 = classifier(numClasses)
        self.classifier6 = classifier(numClasses)
        self.classifier7 = classifier(numClasses)

    def load_base(self, path):
        self.base = base(numPoints)
        self.base = torch.nn.DataParallel(self.base, device_ids=range(torch.cuda.device_count()))
        if not path is None:
            self.base.load_state_dict(torch.load(path))
            # self.base = self.base.cuda()
        # for param in self.base.parameters():
        #     param.requires_grad = False

    def classifier(self, numClasses):
        return nn.Sequential(
            # nn.Dropout(),
            nn.Linear(53248, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            nn.Linear(128, provNum),
        )

    def forward(self, x):
        x0 = self.base.module.features[0](x)
        _x1 = self.base.module.features[1](x0)
        x2 = self.base.module.features[2](_x1)
        _x3 = self.base.module.features[3](x2)
        x4 = self.base.module.features[4](_x3)
        _x5 = self.base.module.features[5](x4)

        x6 = self.base.module.features[6](_x5)
        x7 = self.base.module.features[7](x6)
        x8 = self.base.module.features[8](x7)
        x9 = self.base.module.features[9](x8)
        x9 = x9.view(x9.size(0), -1)
        boxLoc = self.base.module.classifier(x9)

        h1, w1 = _x1.data.size()[2], _x1.data.size()[3]
        p1 = Variable(torch.FloatTensor([[w1,0,0,0],[0,h1,0,0],[0,0,w1,0],[0,0,0,h1]]).cuda(), requires_grad=False)
        h2, w2 = _x3.data.size()[2], _x3.data.size()[3]
        p2 = Variable(torch.FloatTensor([[w2,0,0,0],[0,h2,0,0],[0,0,w2,0],[0,0,0,h2]]).cuda(), requires_grad=False)
        h3, w3 = _x5.data.size()[2], _x5.data.size()[3]
        p3 = Variable(torch.FloatTensor([[w3,0,0,0],[0,h3,0,0],[0,0,w3,0],[0,0,0,h3]]).cuda(), requires_grad=False)

        # x, y, w, h --> x1, y1, x2, y2
        assert boxLoc.data.size()[1] == 4
        postfix = Variable(torch.FloatTensor([[1,0,1,0],[0,1,0,1],[-0.5,0,0.5,0],[0,-0.5,0,0.5]]).cuda(), requires_grad=False)
        boxNew = boxLoc.mm(postfix).clamp(min=0, max=1)

        # input = Variable(torch.rand(2, 1, 10, 10), requires_grad=True)
        # rois = Variable(torch.LongTensor([[0, 1, 2, 7, 8], [0, 3, 3, 8, 8], [1, 3, 3, 8, 8]]), requires_grad=False)
        roi1 = roi_pooling_ims(_x1, boxNew.mm(p1), size=(16, 8))
        roi2 = roi_pooling_ims(_x3, boxNew.mm(p2), size=(16, 8))
        roi3 = roi_pooling_ims(_x5, boxNew.mm(p3), size=(16, 8))
        rois = torch.cat((roi1, roi2, roi3), 1)

        _rois = rois.view(rois.size(0), -1)

        y0 = self.classifier1(_rois)
        y1 = self.classifier2(_rois)
        y2 = self.classifier3(_rois)
        y3 = self.classifier4(_rois)
        y4 = self.classifier5(_rois)
        y5 = self.classifier6(_rois)
        y6 = self.classifier7(_rois)
        return boxLoc, [y0, y1, y2, y3, y4, y5, y6]

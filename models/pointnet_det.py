#  Ref: https://github.com/hansen7/OcCo
# TODO: Reimplement chamfer distance in Torch and add completion/topo loss back; decoder

import sys, torch, itertools, numpy as np, torch.nn as nn, torch.nn.functional as F
from models.pointnet_util import PointNetEncoder, feature_transform_regularizer


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class get_model(nn.Module):
    def __init__(self, **kwargs):
        super(get_model, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.__dict__.update(kwargs)  # to update args, num_coarse, grid_size, grid_scale

        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)

        ''' === detection layer === '''
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 31 * 3)

        self.dp1 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)

        self.view = View((3, 31))


        # batch normalisation will destroy limit the expression
        self.folding1 = nn.Sequential(
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(1024+2+3, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1))


    def forward(self, input_pc):
        x, trans, trans_feats = self.feat(input_pc)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view([-1, 31, 3])
        return x, trans_feats


class get_loss(nn.Module):
    def __init__(self, matdiff_scale=0.001):
        super(get_loss, self).__init__()
        self.matdiff_scale = matdiff_scale

    def forward(self, pred, target, trans_feat):
        matdiff_loss = feature_transform_regularizer(trans_feat)
        loss = F.mse_loss(pred, target) + self.matdiff_scale * matdiff_loss
        return loss

if __name__ == '__main__':
    model = get_model().cuda()
    print(model)
    input_pc = torch.rand(7, 3, 1024).type(torch.float32).cuda()
    x = model(input_pc)
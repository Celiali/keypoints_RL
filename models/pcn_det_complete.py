#  Ref: https://github.com/hansen7/OcCo
# TODO: Reimplement chamfer distance in Torch and add completion/topo loss back; decoder

from models.pcn_util import PCNEncoder
import torch, torch.nn as nn, torch.nn.functional as F
from util.Torch_Utility import chamfer_distance_with_batch

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

        ''' === encoder for extracting feature === '''
        self.feat = PCNEncoder(global_feat=True, channel=3)

        ''' === detection layer === '''
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 31 * 3)

        self.dp1 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)

        self.view = View((3, 31))

        ''' === completion layer === '''
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
        x = self.feat(input_pc)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x_det_vec = self.fc3(x)
        x_kp = x.view([-1, 31, 3])



        coarse = self.folding1(x_det_vec)
        coarse = coarse.view(-1, self.num_coarse, 3)

        grid = self.build_grid(x.shape[0])
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = point_feat.view([-1, self.num_fine, 3])

        global_feat = self.tile(self.expand_dims(x_det_vec, 1), [1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)

        center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.view([-1, self.num_fine, 3])

        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center

        return x_kp, x, coarse, fine



        return x_kp, x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target)
        return loss


if __name__ == '__main__':
    model = get_model().cuda()
    print(model)
    input_pc = torch.rand(7, 3, 1024).type(torch.float32).cuda()
    x = model(input_pc)
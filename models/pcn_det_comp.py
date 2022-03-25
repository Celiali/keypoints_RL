#  Ref: https://github.com/hansen7/OcCo
# TODO: Reimplement chamfer distance in Torch and add completion/topo loss back; decoder

from models.pcn_util import PCNEncoder
import torch, torch.nn as nn, torch.nn.functional as F
from util.Torch_Utility import chamfer_distance_with_batch
import numpy as np
import itertools

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
        self.num_coarse = kwargs['num_coarse']
        self.numKeypoint = kwargs['num_cp']
        self.grid_scale = kwargs['grid_scale']
        self.grid_size = kwargs['grid_size']
        self.num_fine = self.grid_size ** 2 * self.num_coarse # 1024

        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]

        # self.__dict__.update(kwargs)  # to update args, num_coarse, grid_size, grid_scale

        ''' === encoder for extracting feature === '''
        self.feat = PCNEncoder(global_feat=True, channel=3)

        ''' === detection layer === '''
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.numKeypoint * 3)

        self.dp1 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)

        self.view = View((3, self.numKeypoint))

        ''' === completion layer === '''
        # batch normalisation will destroy limit the expression
        self.folding1 = nn.Sequential(
            nn.Linear(self.numKeypoint * 3, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(self.numKeypoint * 3+2+3, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1))

    def build_grid(self, batch_size):
        # a simpler alternative would be: torch.meshgrid()
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        # another solution is: torch.unsqueeze(tensor, dim=dim)
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def tile(self, tensor, multiples):
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:  # increase the speed effectively
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
        return tensor

    def forward(self, input_pc):
        x = self.feat(input_pc)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x_det_vec = self.fc3(x)
        x_kp = x_det_vec.view([-1, self.numKeypoint, 3])

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

        return x_kp, coarse, fine



        return x_kp, x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, coarse, fine, gt_pc):
        kp_loss = F.mse_loss(pred, target)
        coarse_loss_batch_bidirection = chamfer_distance_with_batch(coarse.permute(0, 2, 1), gt_pc)
        fine_loss_batch_bidirection = chamfer_distance_with_batch(fine.permute(0, 2, 1), gt_pc)
        coarse_loss_batch = (coarse_loss_batch_bidirection[0] + coarse_loss_batch_bidirection[1]).mean()
        fine_loss_batch = (fine_loss_batch_bidirection[0] + fine_loss_batch_bidirection[1]).mean()
        loss = kp_loss + coarse_loss_batch + fine_loss_batch
        return loss


if __name__ == '__main__':
    model = get_model().cuda()
    print(model)
    input_pc = torch.rand(7, 3, 1024).type(torch.float32).cuda()
    x = model(input_pc)
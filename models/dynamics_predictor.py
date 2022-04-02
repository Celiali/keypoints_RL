#  dynamics predictor for the detected keypoint

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
        self.numKeypoint = kwargs['num_cp']
        ''' === dynamics prediction layer === '''
        self.fc1 = nn.Linear(self.numKeypoint * 3 + 3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.numKeypoint * 3)
        self.view = View((3, self.numKeypoint))

    def forward(self, input_pc, input_action):
        x_det_vec = input_pc.view([-1, self.numKeypoint * 3])
        x_action_vec = input_action.view([-1, self.numKeypoint * 3])
        x = torch.hstack((x_det_vec, x_action_vec))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view([-1, self.numKeypoint, 3])
        x = input_pc + x
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target)
        return loss


if __name__ == '__main__':

    pass

    # args = {'num_coarse':64, 'numkp':40, 'grid_scale':0.05, 'grid_size':4}
    # from easydict import EasyDict as edict
    # args = edict(args)
    # model = get_model(args=args, grid_size=args.grid_size,grid_scale=args.grid_scale, num_coarse=args.num_coarse, num_channel=3, num_cp = args.numkp).cuda()
    # print(model)
    # # be careful that the dimension order is different for keypoint and point cloud, need to permute
    # input_keypoint = torch.rand(7, args.numkp, 3).type(torch.float32).cuda()
    #
    # pc_gt = torch.rand(7, 3, 1024).type(torch.float32).cuda()
    #
    # print(f"input keypoint shape: {input_keypoint.shape}")
    # pc_coarse, pc_fine = model(input_keypoint)
    # print(f"pc coarse shape:{pc_coarse.shape}")
    # print(f"pc fine shape: {pc_fine.shape}")
    #
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=0.001,
    #     betas=(0.9, 0.999),
    #     eps=1e-08,
    #     weight_decay=0)
    #
    # # compute loss
    # cri = get_loss()
    # loss, _, _ = cri(pc_coarse, pc_fine, pc_gt)
    #
    # for i in range(10000):
    #     optimizer.zero_grad()
    #     pc_coarse, pc_fine = model(input_keypoint)
    #     loss, _, fineloss = cri(pc_coarse, pc_fine, pc_gt)
    #     loss.backward()
    #     optimizer.step()
    #
    #     print(f"input_keypoint: {input_keypoint[0,0,:10]}")
    #     print(f"loss: {loss}")
    #     print(f"fine loss: {fineloss}")
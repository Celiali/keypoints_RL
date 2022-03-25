#  Ref: https://github.com/hansen7/OcCo
# TODO: Reimplement chamfer distance in Torch and add completion/topo loss back; decoder

from models.pcn_util import PCNEncoder
import torch, torch.nn as nn, torch.nn.functional as F

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

        # ''' === encoder for extracting feature === '''
        # self.feat = PCNEncoder(global_feat=True, channel=3)
        #
        # ''' === detection layer === '''
        # self.fc1 = nn.Linear(1024, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 31 * 3)
        #
        # self.dp1 = nn.Dropout(p=0.3)
        # self.bn1 = nn.BatchNorm1d(512)
        #
        # self.view = View((3, 31))

        self.map1 = nn.Linear(31*3, 31*3)
        self.map2 = nn.Linear(31*3, 31*3)


    def forward(self, input_pc):
        x = input_pc.view([-1, 31*3])
        x = self.map1(x)
        # x = F.relu(x)
        # x = self.map2(x)
        x = x.view([-1, 31, 3])
        x = input_pc + x
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target)
        return loss


if __name__ == '__main__':
    model = get_model().cuda()
    print(model)
    input_pc = torch.rand(32, 31, 3).type(torch.float32).cuda()
    x = model(input_pc)
    print("hello")
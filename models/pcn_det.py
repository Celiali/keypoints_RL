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
        self.numKeypoint = kwargs['num_cp']
        # self.__dict__.update(kwargs)  # to update args, num_coarse, grid_size, grid_scale

        ''' === encoder for extracting feature === '''
        self.feat = PCNEncoder(global_feat=True, channel=3)

        ''' === detection layer === '''
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, 31 * 3)
        self.fc3 = nn.Linear(256, self.numKeypoint * 3)

        self.dp1 = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)

        self.view = View((3, self.numKeypoint))


    def forward(self, input_pc):
        x = self.feat(input_pc)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dp1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view([-1, self.numKeypoint, 3])
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
    input_pc = torch.rand(7, 3, 1024).type(torch.float32).cuda()
    x = model(input_pc)
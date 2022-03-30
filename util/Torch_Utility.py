import torch, os, random, trimesh, numpy as np
# from util.SimulatedData import keypoint_mesh_face


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU Usage
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def copy_parameters(model, pretrained, verbose=True):
    # ref: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3

    model_dict = model.state_dict()
    pretrained_dict = pretrained['model_state_dict']
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       k in model_dict and pretrained_dict[k].size() == model_dict[k].size()}

    if verbose:
        print('=' * 27)
        print('Restored Params and Shapes:')
        for k, v in pretrained_dict.items():
            print(k, ': ', v.size())
        print('=' * 68)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def weights_init(m):
    """
    Xavier normal initialisation for weights and zero bias,
    find especially useful for completion and segmentation Tasks
    """
    classname = m.__class__.__name__
    if (classname.find('Conv1d') != -1) or (classname.find('Conv2d') != -1) or (classname.find('Linear') != -1):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def kp_pred(dataset=None, seqid=0, scale_factor = 10.0, detnet=None, modeltype=None, offset=np.array([-0.02, 0, -0.07])):
    scale_factor_list = [6.0, 7.0, 8.0, 9.0, 10.0]
    pointseq_xyz = dataset[seqid][2][:, ::2, :]
    gt_markerseq_xyz = dataset[seqid][4]
    numFrame = pointseq_xyz.shape[0]
    predseq_xyz = []
    mesh_list = []
    point_list = []
    marker_list = []


    for frameid in range(numFrame):
        point_xyz = pointseq_xyz[frameid]
        marker_xyz = gt_markerseq_xyz[frameid]
        ref_xyz = marker_xyz[0]

        point_list.append(point_xyz)
        marker_list.append(marker_xyz)
        
        if scale_factor == 0:
            pred_multiscale = []
            for scale_factor_i in scale_factor_list:
                point_process = (point_xyz - ref_xyz) * scale_factor_i
                marker_process = (marker_xyz - ref_xyz) * scale_factor_i
                points = torch.tensor(point_process).transpose(1, 0).float().cuda().unsqueeze(0)
                if modeltype == 'pointnet_det':
                    pred_singlescale, trans_feat = detnet(points)
                else:
                    pred_singlescale = detnet(points)

                pred_singlescale = pred_singlescale.cpu().squeeze(0).detach().numpy()
                pred_singlescale = pred_singlescale / scale_factor_i + ref_xyz
                pred_multiscale.append(pred_singlescale)

            pred_output = np.mean(pred_multiscale, axis=0)

        else:
            point_process = (point_xyz - ref_xyz) * scale_factor
            marker_process = (marker_xyz - ref_xyz) * scale_factor

            # points = torch.tensor(point_process).transpose(1, 0).float().cuda().unsqueeze(0)
            points = torch.tensor(point_process).transpose(1, 0).float().cuda().unsqueeze(0)

            if modeltype == 'pointnet_det':
                pred, trans_feat = detnet(points)
            else:
                pred = detnet(points)

            pred_output = pred.cpu().squeeze(0).detach().numpy()
            pred_output = pred_output / scale_factor + ref_xyz


        mesh = trimesh.Trimesh(vertices=pred_output, faces=keypoint_mesh_face)

        mesh_list.append(mesh)
        predseq_xyz.append(pred_output)

    return point_list, marker_list, predseq_xyz, mesh_list


def calc_smooth(predseq):
    numFrame = len(predseq)
    S_tmp = []

    for i in range(2,numFrame):
        S_i = np.linalg.norm(predseq[i] - 2*predseq[i-1] + predseq[i-2], axis=1)
        S_tmp.append(S_i)
    return np.mean(S_tmp), np.mean(S_tmp, axis=0)

def calc_dist_point2mesh(predseq, markerseq):
    return None

def chamfer_distance_with_batch(p1, p2, verbose=False):

    '''
    Calculate Chamfer Distance between two point sets
    the arrangement of axes is different from the original implementation as we take coordiantes as channels
    :param p1: size[B, D, N]
    :param p2: size[B, D, M]
    :param debug: whether need to output debug info
    :return: sum of all batches of Chamfer Distance of two point sets
    '''

    assert p1.size(0) == p2.size(0) and p1.size(1) == p2.size(1)

    if verbose:
        print('num of pointsets: ', p1[0])

    p1 = p1.unsqueeze(1)
    p2 = p2.unsqueeze(1)
    if verbose:
        print('p1 size is {}'.format(p1.size()))
        print('p2 size is {}'.format(p2.size()))

    p1 = p1.repeat(1, p2.size(3), 1, 1)
    if verbose:
        print('p1 size is {}'.format(p1.size()))

    p1 = p1.transpose(1, 3)
    if verbose:
        print('p1 size is {}'.format(p1.size()))

    p2 = p2.repeat(1, p1.size(1), 1, 1)
    if verbose:
        print('p2 size is {}'.format(p2.size()))

    dist = torch.add(p1, torch.neg(p2))
    if verbose:
        print('dist size is {}'.format(dist.size()))
        print(dist[0])

    dist = torch.norm(dist, 2, dim=2)
    if verbose:
        print('dist size is {}'.format(dist.size()))
        print(dist)

    dist_p1_p2 = torch.min(dist, dim=2)[0].mean(dim=1)
    dist_p2_p1 = torch.min(dist, dim=1)[0].mean(dim=1)
    
    return dist_p1_p2, dist_p2_p1
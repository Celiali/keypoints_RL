import os, sys, torch, importlib, argparse, numpy as np, trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append('util')
sys.path.append('models')
sys.path.append('checkpoint')
sys.path.append('/Midgard/home/zehang/project/keypoint_humanoids')

from torch.utils.data import DataLoader
from util import Datasets
from util.Torch_Utility import copy_parameters, kp_pred, calc_smooth, calc_dist_point2mesh
from util.loaddata import General_PartKPDataLoader_HDF5
from util.SimulatedData import keypoint_edges_31
import time
from util.SimulatedData import keypoint_mesh_face
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser('Point Cloud Keypoint Detection')

    ''' === Training Setting === '''
    parser.add_argument('--log_dir', type=str, help='log folder [default: ]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU [default: 0]')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size [default: 32]')
    parser.add_argument('--epoch', type=int, default=50, help='number of epoch [default: 50]')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate [default: 1e-4]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='lr decay rate [default: 0.7]')
    parser.add_argument('--step_size', type=int, default=20, help='lr decay step [default: 20 epoch]')
    parser.add_argument('--savemodel', action='store_true', help='save the model')
    parser.add_argument('--restore', action='store_true', help='loaded from restore [default: False]')
    parser.add_argument('--restore_path', type=str, help='path to saved pre-trained model [default: ]')
    parser.add_argument('--steps_eval', type=int, default=1000, help='# steps to evaluate [default: 1e3]')
    parser.add_argument('--epochs_save', type=int, default=5, help='# epochs to save [default: 5 epochs]')
    parser.add_argument("--tasklist", nargs="+", default=None)
    parser.add_argument('--numkp', type=int, default=3, help='number of keypoints [default: 3]')
    parser.add_argument('--ck_path', type=str, help='path to save trained model [default: ]')
    parser.add_argument('--data_path', type=str, help='path to the dataset')

    ''' === Model Setting === '''
    parser.add_argument('--model', type=str, default='pcn_det', help='model [pcn_occo]')
    parser.add_argument('--padding', type=str, default='replace', help='method for padding')
    parser.add_argument('--augrot', action='store_true', help='y rotation augmentation [default: False]')
    parser.add_argument('--augocc', action='store_true', help='occlusion augmentation [default: False]')
    parser.add_argument('--augsca', action='store_true', help='scaling augmentation [default: False]')
    parser.add_argument('--k', type=int, default=20, help='# nearest neighbors in DGCNN [20]')
    parser.add_argument('--grid_size', type=int, default=4, help='edge length of the 2D grid [4]')
    parser.add_argument('--grid_scale', type=float, default=0.05, help='scale of the 2D grid [0.5]')
    parser.add_argument('--num_coarse', type=int, default=64, help='# points in coarse gt [1024]')
    parser.add_argument('--emb_dims', type=int, default=1024, help='# dimension of DGCNN encoder [1024]')
    parser.add_argument('--input_pts', type=int, default=1024, help='# points of occluded inputs [1024]')
    parser.add_argument('--gt_pts', type=int, default=1024, help='# points of ground truth inputs [1024]')

    ''' === Testing Setting === '''
    parser.add_argument("--seqid", nargs="+", default=[])

    return parser.parse_args()


def main(args, task_index, num_kp=3):
    '''
    keypoint detection main
    '''

    ''' === Set up Task and Load Data === '''
    root = args.data_path
    print("load data from {}".format(root))
    kp_dict_file = "src/kp_ind_list.pickle"
    NUM_KP = 31

    checkpoints_dir = os.path.join(args.ck_path, "{}_{}_{}_{}/{}/{}/".format(args.model, args.augrot, args.augocc, args.augsca, task_index, args.numkp))

    if os.path.exists(checkpoints_dir) is False:
        print("invalid checkpoint path")


    ''' === Load Model and Backup Scripts === '''
    MODEL = importlib.import_module(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MODEL.get_model(args=args, grid_size=args.grid_size,
                                grid_scale=args.grid_scale, num_coarse=args.num_coarse, num_channel=3, num_cp = NUM_KP).to(device)

    detector = torch.nn.DataParallel(detector)
    print('=' * 27)
    print('Using %d GPU,' % torch.cuda.device_count(), 'Indices: %s' % args.gpu)
    print('=' * 27)

    ''' === Restore Model from Checkpoints, If there is any === '''

    bestepoch = np.max([int(ckfile.split('.')[0].split('_')[-1]) for ckfile in os.listdir(checkpoints_dir)])
    args.restore_path_root = os.path.join(checkpoints_dir, "model_epoch_{}.pth".format(bestepoch))
    detector.load_state_dict(torch.load(args.restore_path_root)['model_state_dict'])
    detector.eval()

    '''
       load the dataset
       '''

    root = "../data/"
    tasks_path = os.path.join(root, "h5data/tasks/pointclouds_sim")
    print("Chosen task:", task_index)
    task = Datasets.get_task_by_index(task_index)

    H5DataPath = os.path.join(tasks_path, "{}_{}_clean.h5".format(task_index, 'train'))
    TRAIN_DATASET = General_PartKPDataLoader_HDF5(H5DataPath, augrot=args.augrot, augocc=args.augocc, augsca=args.augsca, ref="left")
    # VAL
    H5DataPath = os.path.join(tasks_path, "{}_{}_clean.h5".format(task_index, 'valid'))
    VAL_DATASET = General_PartKPDataLoader_HDF5(H5DataPath, augrot=args.augrot, augocc=args.augocc, augsca=args.augsca, ref="left")
    # TEST
    H5DataPath = os.path.join(tasks_path, "{}_{}_clean.h5".format(task_index, 'test'))
    TEST_DATASET = General_PartKPDataLoader_HDF5(H5DataPath, augrot=args.augrot, augocc=args.augocc, augsca=args.augsca, ref="left")


    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valDataLoader = DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    # evaluation
    with tqdm(trainDataLoader, unit="batch") as tepoch:
        batchcount = 0
        trn_errorlist = []
        # for  points, target, _ in tepoch:
        for points, target, ref, sid, fid in tepoch:
            batchcount += 1
            points, target = points.transpose(2, 1).float().cuda(), target.float().cuda()

            if args.model == 'pointnet_det':
                pred, trans_feat = detector(points)
            elif args.model == 'pcn_det':
                pred = detector(points)
            elif args.model == 'pcn_det_comp':
                pred, coarse, fine = detector(points)
            else:
                raise NotImplementedError("only support pointnet_det, pcn_det, pcn_det_comp")


            batcherror = torch.linalg.norm(target - pred, axis=-1).mean(axis=-1).data.cpu().numpy()
            trn_errorlist = np.hstack((trn_errorlist, batcherror))



    with tqdm(valDataLoader, unit="batch") as tepoch:
        batchcount = 0
        val_errorlist = []
        # for  points, target, _ in tepoch:
        for points, target, ref, sid, fid in tepoch:
            batchcount += 1
            points, target = points.transpose(2, 1).float().cuda(), target.float().cuda()

            if args.model == 'pointnet_det':
                pred, trans_feat = detector(points)
            elif args.model == 'pcn_det':
                pred = detector(points)
            elif args.model == 'pcn_det_comp':
                pred, coarse, fine = detector(points)
            else:
                raise NotImplementedError("only support pointnet_det, pcn_det, pcn_det_comp")

            batcherror = torch.linalg.norm(target - pred, axis=-1).mean(axis=-1).data.cpu().numpy()
            val_errorlist = np.hstack((val_errorlist, batcherror))

    with tqdm(testDataLoader, unit="batch") as tepoch:
        batchcount = 0
        test_errorlist = []
        # for  points, target, _ in tepoch:
        for points, target, ref, sid, fid in tepoch:
            batchcount += 1
            points, target = points.transpose(2, 1).float().cuda(), target.float().cuda()

            if args.model == 'pointnet_det':
                pred, trans_feat = detector(points)
            elif args.model == 'pcn_det':
                pred = detector(points)
            elif args.model == 'pcn_det_comp':
                pred, coarse, fine = detector(points)
            else:
                raise NotImplementedError("only support pointnet_det, pcn_det, pcn_det_comp")

            batcherror = torch.linalg.norm(target - pred, axis=-1).mean(axis=-1).data.cpu().numpy()
            test_errorlist = np.hstack((test_errorlist, batcherror))

    return trn_errorlist, val_errorlist, test_errorlist

if __name__ == '__main__':
    '''
        python src/Evaluate_acc_sim.py --batch_size 256 --model pointnet_det --augrot --augocc --augsca  '''
    args = parse_args()
    task_index_list = args.tasklist
    if task_index_list == None:
        task_index_list = np.arange(1,21)
        print("test all")

    num_kp_list = [3,5,10,15,20,25,30,35,40,45,50,55,60]

    for num_kp in num_kp_list:
        trn_error_tasks = []
        val_error_tasks = []
        test_error_tasks = []
        for task_index in task_index_list:
            trn_error_list, val_error_list, test_error_list = main(args, task_index=task_index, num_kp=num_kp)
            trn_error_tasks.append(trn_error_list)
            val_error_tasks.append(val_error_list)
            test_error_tasks.append(test_error_list)

        error_total = [trn_error_tasks, val_error_tasks, test_error_tasks]

        import pickle
        errorfile = open("logs/exp1_sim_{}.acc".format(args.model), 'wb')
        pickle.dump(error_total, errorfile)
        errorfile.close()



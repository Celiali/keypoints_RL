'''
    Train keypoint detector for different resolutions
    python src/trainKp.py --batch_size 32 --epoch 50 --model pcn_det --augrot --augocc --savemodel --tasklist 8 18 10 20 --numkp 3
'''

import os, sys, torch, importlib, argparse, numpy as np
from tqdm import tqdm
sys.path.append('util')
sys.path.append('models')
sys.path.append('checkpoint')
sys.path.append('/Midgard/home/zehang/project/keypoint_humanoids')

from torch.utils.data import DataLoader
from util import Datasets
from util.Torch_Utility import copy_parameters
from util.loaddata import General_PartKPDataLoader_HDF5, General_PartKPDataLoader_dyn_HDF5

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
    parser.add_argument("--tasklist", nargs="+", default=[1, 2])
    parser.add_argument('--numkp', type=int, default=3, help='number of keypoints [default: 3]')
    parser.add_argument('--detector_ck_path', type=str, help='path to the trained detector model [default: ]')
    parser.add_argument('--dyn_predictor_ck_path', type=str, help='path to save dyn_predictor model [default: ]')
    parser.add_argument('--data_path', type=str, help='path to the dataset')

    ''' === Model Setting === '''
    parser.add_argument('--model', type=str, default='pcn_det', help='model [pcn_occo]')
    parser.add_argument('--padding', type=str, default='replace', help='method for padding')
    parser.add_argument('--augrot', action='store_true', help='y rotation augmentation [default: False]')
    parser.add_argument('--augocc', action='store_true', help='occlusion augmentation [default: False]')
    parser.add_argument('--augsca', action='store_true', help='scaling augmentation [default: False]')
    parser.add_argument('--k', type=int, default=20, help='# nearest neighbors in DGCNN [20]')
    parser.add_argument('--grid_size', type=int, default=4, help='edge length of the 2D grid [4]') # 4
    parser.add_argument('--grid_scale', type=float, default=0.05, help='scale of the 2D grid [0.5]') # 0.5
    parser.add_argument('--num_coarse', type=int, default=64, help='# points in coarse gt [64]') # 1024 , change to 64
    parser.add_argument('--num_fine', type=int, default=1024, help='# points in fine [1024]]') # 1024 , change to 64
    parser.add_argument('--emb_dims', type=int, default=1024, help='# dimension of DGCNN encoder [1024]')
    parser.add_argument('--input_pts', type=int, default=1024, help='# points of occluded inputs [1024]')
    parser.add_argument('--gt_pts', type=int, default=1024, help='# points of ground truth inputs [1024]') # 16384

    return parser.parse_args()


def main(args, task_index):
    '''
    keypoint detection main
    '''

    #TODO: read the kp ind file

    ''' === Set up Task and Load Data === '''
    root = args.data_path
    print("load data from {}".format(root))
    kp_dict_file = "src/kp_ind_list.pickle"
    # checkpoints_dir = "checkpoint/{}_{}_{}_{}/{}/{}/".format(args.model, args.augrot, args.augocc, args.augsca, task_index, args.numkp)
    detector_checkpoints_dir = os.path.join(args.detector_ck_path, "{}_{}_{}_{}/{}/{}/".format(args.model, args.augrot, args.augocc, args.augsca, task_index, args.numkp))
    dyn_predictor_checkpoints_dir = os.path.join(args.dyn_predictor_ck_path, "{}_{}_{}_{}/{}/{}/".format(args.model, args.augrot, args.augocc, args.augsca, task_index, args.numkp))

    log_dir = os.path.join(args.log_dir, "{}_{}_{}_{}/{}/".format(args.model, args.augrot, args.augocc, args.augsca, args.numkp))
    if os.path.exists(detector_checkpoints_dir) is False:
        raise NameError(f"{detector_checkpoints_dir} is not the correct path to load detector")

    if os.path.exists(dyn_predictor_checkpoints_dir) is False:
        os.makedirs(dyn_predictor_checkpoints_dir)
    if os.path.exists(log_dir) is False:
        os.makedirs(log_dir)

    tstErrfile_name = os.path.join(log_dir, "testerror_{}.txt".format(task_index))
    tstErrfile_Total_name = os.path.join(log_dir, "whole.txt")

    tstErrfile = open(tstErrfile_name, 'w')
    tstErrfile_Total_name = open(tstErrfile_Total_name, 'a')

    tasks_path = os.path.join(root, "h5data/tasks")
    print("Chosen task:", task_index)
    # task = Datasets.get_task_by_index(task_index)

    H5DataPath = os.path.join(tasks_path, "{}_{}_full.h5".format(task_index, 'train'))
    TRAIN_DATASET = General_PartKPDataLoader_HDF5(H5DataPath, augrot=args.augrot, augocc=args.augocc, augsca=args.augsca, ref="left", kp_dict_file=kp_dict_file, numkp=args.numkp)
    # VAL
    H5DataPath = os.path.join(tasks_path, "{}_{}_full.h5".format(task_index, 'valid'))
    VAL_DATASET = General_PartKPDataLoader_HDF5(H5DataPath, augrot=args.augrot, augocc=args.augocc, augsca=args.augsca, ref="left", kp_dict_file=kp_dict_file, numkp=args.numkp)
    # TEST
    H5DataPath = os.path.join(tasks_path, "{}_{}_full.h5".format(task_index, 'test'))
    TEST_DATASET = General_PartKPDataLoader_HDF5(H5DataPath, augrot=args.augrot, augocc=args.augocc, augsca=args.augsca, ref="left", kp_dict_file=kp_dict_file, numkp=args.numkp)


    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    valDataLoader = DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    ''' === Restore Keypoint detection model from keypoint generation=== '''
    MODEL_DET = importlib.import_module(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MODEL_DET.get_model(args=args, grid_size=args.grid_size,
                                grid_scale=args.grid_scale, num_coarse=args.num_coarse, num_channel=3, num_cp = args.numkp).to(device)

    detector = torch.nn.DataParallel(detector)
    print('=' * 27)
    print('Using %d GPU,' % torch.cuda.device_count(), 'Indices: %s' % args.gpu)
    print('=' * 27)
    bestepoch = np.max([int(ckfile.split('.')[0].split('_')[-1]) for ckfile in os.listdir(detector_checkpoints_dir)])
    args.restore_path_root = os.path.join(detector_checkpoints_dir, "model_epoch_{}.pth".format(bestepoch))
    detector.load_state_dict(torch.load(args.restore_path_root)['model_state_dict'])
    print(f"load state dict: {detector_checkpoints_dir}")
    detector.eval()


    MODEL_DYN = importlib.import_module("dynamics_predictor")
    dyn_predictor = MODEL_DYN.get_model(args=args, grid_size=args.grid_size,
                                   grid_scale=args.grid_scale, num_coarse=args.num_coarse, num_fine=args.num_fine, num_channel=3,
                                   num_cp=args.numkp).to(device)
    dyn_predictor = torch.nn.DataParallel(dyn_predictor)
    criterion = MODEL_DYN.get_loss().to(device)

    ''' === Optimizer setup === '''
    optimizer = torch.optim.Adam(
        dyn_predictor.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0)

    loss_val_best = 10000
    loss_test_final = 10000

    for epoch in range(args.epoch):

        ''' === Training === '''
        # for points, target, _ in tqdm(trainDataLoader, total=len(trainDataLoader), smoothing=0.9):
        with tqdm(trainDataLoader, unit="batch") as tepoch:
            loss_trn = 0
            batchcount = 0
            errorlist = np.array([])
            dyn_predictor.train()
            # for  points, target, _ in tepoch:
            for  points_t1, points_t2, target_t1, target_t2, ref_t1, ref_t2, sid, fid in tepoch:
                batchcount += 1
                points_t1 = points_t1.transpose(2, 1).float().cuda()
                points_t2 = points_t2.transpose(2, 1).float().cuda()

                pred_kp_t1 = detector(points_t1)
                pred_kp_t2 = detector(points_t2)

                action = ref_t2 - ref_t1
                est_kp_t2 = dyn_predictor(pred_kp_t1, action)
                loss = criterion(est_kp_t2, pred_kp_t2)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                loss_trn += loss.item()

            loss_trn /= batchcount

        print("epoch {}: training loss={}".format(epoch, loss_trn))

        with tqdm(valDataLoader, unit="batch") as tepoch:
            loss_val = 0
            batchcount = 0
            errorlist = []
            dyn_predictor.eval()
            # for  points, target, _ in tepoch:
            for  points_t1, points_t2, target_t1, target_t2, ref_t1, ref_t2, sid, fid in tepoch:
                batchcount += 1
                points_t1 = points_t1.transpose(2, 1).float().cuda()
                points_t2 = points_t2.transpose(2, 1).float().cuda()

                pred_kp_t1 = detector(points_t1)
                pred_kp_t2 = detector(points_t2)

                action = ref_t2 - ref_t1
                est_kp_t2 = dyn_predictor(pred_kp_t1, action)
                loss = criterion(est_kp_t2, pred_kp_t2)

                tepoch.set_postfix(loss=loss.item())
                loss_val += loss.item()
            loss_val /= batchcount


        print("epoch {}: validation loss={}".format(epoch, loss_val))

        with tqdm(testDataLoader, unit="batch") as tepoch:
            loss_test = 0
            batchcount = 0
            errorlist = []
            dyn_predictor.eval()
            # for  points, target, _ in tepoch:

            # for  points, target, _ in tepoch:
            for  points_t1, points_t2, target_t1, target_t2, ref_t1, ref_t2, sid, fid in tepoch:
                batchcount += 1
                points_t1 = points_t1.transpose(2, 1).float().cuda()
                points_t2 = points_t2.transpose(2, 1).float().cuda()

                pred_kp_t1 = detector(points_t1)
                pred_kp_t2 = detector(points_t2)

                action = ref_t2 - ref_t1
                est_kp_t2 = dyn_predictor(pred_kp_t1, action)
                loss = criterion(est_kp_t2, pred_kp_t2)

                tepoch.set_postfix(loss=loss.item())
                loss_test += loss.item()

            loss_test /= batchcount

        print("epoch {}: testing loss={}".format(epoch, loss_test))


        if loss_val <= loss_val_best:
            # renew the loss
            loss_val_best = loss_val

            if args.savemodel:
                # store the model
                state = {
                    'epoch': epoch,
                    'model_state_dict': dyn_predictor.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(dyn_predictor_checkpoints_dir,
                                               "model_epoch_{}.pth".format(epoch)))
                loss_test_final = loss_test
                # only record the test error when we have a better validation loss
                tstErrfile.write("epoch{}:{}\n".format(epoch, loss_test))

    tstErrfile_Total_name.write("task{}:{}\n".format(task_index, loss_test_final))

    tstErrfile.close()
    tstErrfile_Total_name.close()


if __name__ == '__main__':
    '''
        python src/trainKp.py --batch_size 32 --epoch 50 --model pcn_det --augrot --augocc --augsca --savemodel --tasklist 1 2 3 4 5
        python src/trainKp.py --batch_size 32 --epoch 50 --model pointnet_det --augrot --augocc --augsca --savemodel --tasklist 5 10 15 16  
        python src/trainKp.py --batch_size 32 --epoch 50 --model pointnet_det --augrot --augocc --augsca --savemodel --tasklist 17 18 19 20  
    '''
    args = parse_args()
    task_index_list = args.tasklist
    # # task_index_list = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # # task_index_list = [17,18,19,20]
    # # task_index_list = [7, 9, 17, 19]
    # # task_index_list = [1,2,33]
    print(task_index_list)
    for task_index in task_index_list:
        main(args, task_index=task_index)


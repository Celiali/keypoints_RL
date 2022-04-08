import os, sys, torch, importlib, argparse, numpy as np, trimesh

sys.path.append('util')
sys.path.append('models')
sys.path.append('checkpoint')
sys.path.append('/Midgard/home/zehang/project/keypoint_humanoids')

from tqdm import tqdm
from util.Torch_Utility import chamfer_distance_with_batch
import matplotlib.pyplot as plt
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
    parser.add_argument("--tasklist", nargs="+", default=[1, 2])
    parser.add_argument('--numkp', type=int, default=3, help='number of keypoints [default: 3]')
    parser.add_argument('--detector_ck_path', type=str, help='path to the trained detector model [default: ]')
    parser.add_argument('--decoder_ck_path', type=str, help='path to save decoder model [default: ]')
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

    numkp = 3

    ''' === Set up Task and Load Data === '''
    root = args.data_path
    print("load data from {}".format(root))
    # kp_dict_file = "src/kp_ind_list.pickle"

    # checkpoints_dir = os.path.join(args.ck_path, "{}_{}_{}_{}/{}/{}/".format(args.model, args.augrot, args.augocc, args.augsca, task_index, args.numkp))
    detector_checkpoints_dir = os.path.join(args.detector_ck_path, "{}_{}_{}_{}/{}/{}/".format(args.model, args.augrot, args.augocc, args.augsca, task_index, numkp))
    completor_checkpoints_dir = os.path.join(args.decoder_ck_path, "{}_{}_{}_{}/{}/{}/".format(args.model, args.augrot, args.augocc, args.augsca, task_index, numkp))


    if os.path.exists(detector_checkpoints_dir) is False:
        print("invalid detector checkpoint path")

    if os.path.exists(completor_checkpoints_dir) is False:
        print("invalid completor/decoder checkpoint path")


    ''' === Load Model and Backup Scripts === '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load the detector
    MODEL = importlib.import_module(args.model)
    detector = MODEL.get_model(args=args, grid_size=args.grid_size,
                                grid_scale=args.grid_scale, num_coarse=args.num_coarse, num_channel=3, num_cp = numkp).to(device)

    detector = torch.nn.DataParallel(detector)
    # load the decoder
    MODEL_COMP = importlib.import_module("decoder_comp")
    completor = MODEL_COMP.get_model(args=args, grid_size=args.grid_size,
                                   grid_scale=args.grid_scale, num_coarse=args.num_coarse, num_fine=args.num_fine, num_channel=3,
                                   num_cp=numkp).to(device)
    completor = torch.nn.DataParallel(completor)

    ''' === Restore detector Model from Checkpoints, If there is any === '''

    bestepoch_detector = np.max([int(ckfile.split('.')[0].split('_')[-1]) for ckfile in os.listdir(detector_checkpoints_dir)])
    args.restore_path_root = os.path.join(detector_checkpoints_dir, "model_epoch_{}.pth".format(bestepoch_detector))
    print(f"{args.restore_path_root}")
    detector.load_state_dict(torch.load(args.restore_path_root)['model_state_dict'])
    detector.eval()

    ''' === Restore decoder Model from Checkpoints, If there is any === '''

    bestepoch_decoder = np.max([int(ckfile.split('.')[0].split('_')[-1]) for ckfile in os.listdir(completor_checkpoints_dir)])
    args.restore_path_root = os.path.join(completor_checkpoints_dir, "model_epoch_{}.pth".format(bestepoch_decoder))
    print(f"{args.restore_path_root}")
    detector.load_state_dict(torch.load(args.restore_path_root)['model_state_dict'])
    completor.eval()

    '''
       load the dataset
       '''

    root = "../data/"
    tasks_path = os.path.join(root, "h5data/tasks")
    print("Chosen task:", task_index)

    # TEST
    H5DataPath = os.path.join(tasks_path, "{}_{}_clean.h5".format(task_index, 'test'))
    TEST_DATASET = General_PartKPDataLoader_HDF5(H5DataPath, augrot=args.augrot, augocc=args.augocc, augsca=args.augsca, ref="left")


    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)

    with tqdm(testDataLoader, unit="batch") as tepoch:
        batchcount = 0
        test_errorlist = []
        # for  points, target, _ in tepoch:
        for points, target, ref, sid, fid in tepoch:
            batchcount += 1
            points, target = points.transpose(2, 1).float().cuda(), target.float().cuda()
            if args.model == 'pcn_det':
                pred_kp = detector(points)
            else:
                raise NotImplementedError("only support pcn_det for testing here")

            _, pc_fine = completor(pred_kp)
            fine_loss_batch_bidirection = chamfer_distance_with_batch(pc_fine.permute(0, 2, 1), points)
            fine_chamferdist_batch = (fine_loss_batch_bidirection[0] + fine_loss_batch_bidirection[1])


            batcherror = fine_chamferdist_batch.data.cpu().numpy()
            print(batcherror.shape)
            test_errorlist = np.hstack((test_errorlist, batcherror))

    return test_errorlist

if __name__ == '__main__':
    '''
        python src/Evaluate_acc_sim.py --batch_size 256 --model pcn_det --augrot --augocc --augsca  
        
        --data_path /local_storage/users/zehang/keypoint_humanoids/data --detector_ck_path /nas/zehang/keypoint_humanoids/detect_checkpoint --decoder_ck_path /nas/zehang/keypoint_humanoids/kp2comp_checkpoint --batch_size 32 --model pcn_det --augrot --augocc --augsca --savemodel --tasklist 2 4 6 8 10 12 14 16 18 20--numkp 3
        
        '''
    args = parse_args()
    # task_index_list = args.tasklist
    # if task_index_list == None:
    #     task_index_list = np.arange(1,21)
    #     print("test all")

    test_error_total = []
    for i in range([2,4,6,8,10,12,14,16,18,20]):
        test_error_list = main(args, task_index=i)
        test_error_total.append(test_error_list)
    import pickle
    errorfile = open("./exp1_sim_{}.acc".format(args.model), 'wb')
    pickle.dump(test_error_total, errorfile)
    errorfile.close()

    # num_kp_list = [3,5,10,15,20,25,30,35,40,45,50,55,60]
    #
    # for num_kp in num_kp_list:
    #     trn_error_tasks = []
    #     val_error_tasks = []
    #     test_error_tasks = []
    #     for task_index in task_index_list:
    #         trn_error_list, val_error_list, test_error_list = main(args, task_index=task_index)
    #         trn_error_tasks.append(trn_error_list)
    #         val_error_tasks.append(val_error_list)
    #         test_error_tasks.append(test_error_list)

        # error_total = [trn_error_tasks, val_error_tasks, test_error_tasks]
        #
        # import pickle
        # errorfile = open("logs/exp1_sim_{}.acc".format(args.model), 'wb')
        # pickle.dump(error_total, errorfile)
        # errorfile.close()
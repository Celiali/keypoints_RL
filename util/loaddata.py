import numpy as np
import open3d as o3d
import os
from operator import itemgetter
from colorsys import rgb_to_hsv
import pickle

'''

Function for loading real data

'''

def GetFilelist(srcpath: str):
    '''
    Input: path of the raw dataset folder
    Return:
        pcd_markerlist_all: list of marker file path
        pcd_xyzlist_all: list of xyz file path
        pcd_rgblist_all: list of rgb file path
    '''

    pcdlist = os.listdir(srcpath)
    # create list to hold src files
    pcd_speed_folder = []
    pcd_markerlist_all = []
    pcd_xyzlist_all = []
    pcd_rgblist_all = []
    for i in pcdlist:
        pcd_speed_folder.append(i)

    for pcd_speed_i in pcd_speed_folder:
        pcd_file_list = os.listdir(os.path.join(srcpath, pcd_speed_i))
        pcd_markerlist = []
        pcd_xyzlist = []
        pcd_rgblist = []
        frameidlist = []
        pcd_markerlist1 = []
        pcd_xyzlist1 = []
        pcd_rgblist1 = []
        frameidlist1 = []
        for pcd_file_i in pcd_file_list:
            if "times" in pcd_file_i:
                continue

            if "markers" in pcd_file_i:
                markerfilename = pcd_file_i
                frameid = int(markerfilename[:-4].split('_')[6])
                # xyzfilename = markerfilename.replace("markers", "xyz")
                # rgbfilename = markerfilename.replace("markers", "rgb")
                if pcd_file_i.split('_')[4] == '0':
                    pcd_markerlist.append(os.path.join(srcpath, pcd_speed_i, markerfilename))
                    frameidlist.append(frameid)
                elif pcd_file_i.split('_')[4] == '1':
                    pcd_markerlist1.append(os.path.join(srcpath, pcd_speed_i, markerfilename))
                    frameidlist1.append(frameid)

        frameidlist_sortind = np.argsort(frameidlist)
        frameidlist1_sortind = np.argsort(frameidlist1)

        pcd_markerlist = list(itemgetter(*frameidlist_sortind)(pcd_markerlist))
        pcd_markerlist1 = list(itemgetter(*frameidlist1_sortind)(pcd_markerlist1))
        pcd_markerlist_all.append(pcd_markerlist)
        pcd_markerlist_all.append(pcd_markerlist1)

        for markerfilename in pcd_markerlist:
            xyzfilename = markerfilename.replace("markers", "xyz")
            rgbfilename = markerfilename.replace("markers", "rgb")
            pcd_xyzlist.append(xyzfilename)
            pcd_rgblist.append(rgbfilename)

        for markerfilename in pcd_markerlist1:
            xyzfilename = markerfilename.replace("markers", "xyz")
            rgbfilename = markerfilename.replace("markers", "rgb")
            pcd_xyzlist1.append(xyzfilename)
            pcd_rgblist1.append(rgbfilename)

        pcd_xyzlist_all.append(pcd_xyzlist)
        pcd_xyzlist_all.append(pcd_xyzlist1)
        pcd_rgblist_all.append(pcd_rgblist)
        pcd_rgblist_all.append(pcd_rgblist1)

    return pcd_markerlist_all, pcd_xyzlist_all, pcd_rgblist_all

def read_frame_raw(frameid, pcd_marker_trylist,pcd_xyz_trylist, pcd_rgb_trylist):
    # read the ith frame
    pc_marker = np.load(pcd_marker_trylist[frameid])[:,:3]
    pc_xyz = np.load(pcd_xyz_trylist[frameid])
    pc_rgb = np.load(pcd_rgb_trylist[frameid]) / 255.

    # post-processing for the point clouds
    pc_xyz[:,1] = -pc_xyz[:,1]
    pc_xyz[:,2] = -pc_xyz[:,2]
    # post-processing for the markers
    pc_marker[:, 0] = -pc_marker[:, 0]
    pc_marker[:, 1] = -pc_marker[:, 1]
    pc_marker = pc_marker[:,[1,2,0]]

    point_valid_ind = np.arange(len(pc_xyz))
    pc_xyz = pc_xyz[point_valid_ind]


    # remove outlier
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    _, ind = pcd.remove_radius_outlier(nb_points=10, radius=0.02)
    ind = np.array(ind)
    ind = ind[np.linspace(0, len(ind) - 1, 70000).astype(int)]
    # stack the point xyz with markers
    pc_xyz = pc_xyz[ind]
    pc_xyz = np.vstack((pc_xyz, pc_marker))
    # stack point rgb with markers
    pc_rgb = pc_rgb[point_valid_ind]
    pc_rgb = pc_rgb[ind]
    m_color = np.zeros(pc_marker.shape)
    m_color[:,0] = 1
    pc_rgb = np.vstack((pc_rgb, m_color))
    return pc_xyz, pc_rgb

def read_frame2process_raw(frameid, pcd_marker_trylist,pcd_xyz_trylist, pcd_rgb_trylist):
    # read the ith frame
    pc_marker = np.load(pcd_marker_trylist[frameid])[:,:3]
    pc_xyz = np.load(pcd_xyz_trylist[frameid])
    pc_rgb = np.load(pcd_rgb_trylist[frameid]) / 255.

    # post-processing
    pc_xyz[:,1] = -pc_xyz[:,1]
    pc_xyz[:,2] = -pc_xyz[:,2]

    pc_marker[:, 0] = -pc_marker[:, 0]
    pc_marker[:, 1] = -pc_marker[:, 1]
    pc_marker = pc_marker[:,[1,2,0]]


    point_valid_ind = np.array(PointFilteringSep(pc_xyz, pc_rgb, pc_marker))
    point_valid_ind = point_valid_ind[np.linspace(0, len(point_valid_ind)-1, 5000).astype(int)]
    pc_xyz = pc_xyz[point_valid_ind]


    # remove outlier
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    _, ind = pcd.remove_radius_outlier(nb_points=3, radius=0.03)
    ind = np.array(ind)
    ind = ind[np.linspace(0, len(ind) - 1, 2048).astype(int)]
    pc_xyz = pc_xyz[ind]



    # pc_xyz = np.vstack((pc_xyz, pc_marker))

    pc_rgb = pc_rgb[point_valid_ind]
    # remove outlier
    pc_rgb = pc_rgb[ind]



    #m_color = np.zeros(pc_marker.shape)
    #m_color[:,0] = 1
    #pc_rgb = np.vstack((pc_rgb, m_color))


    return pc_xyz, pc_rgb, pc_marker

def RGB2YUV(R, G, B):
    Y =  0.257 * R + 0.504 * G + 0.098 * B +  16
    U = -0.148 * R - 0.291 * G + 0.439 * B + 128
    V =  0.439 * R - 0.368 * G - 0.071 * B + 128
    return Y, U, V

def checkkpneighbor(kpset, m):
    for kp_id in range(kpset.shape[0]):
        if np.all(kpset[kp_id] == [0,0,0]):
            continue
        else:
            if np.linalg.norm(kpset[kp_id] - m) < 1.5: # 0.1:
                return True
    return False

def checkvalid(m, kpset, point):
    x = point[0]
    y = point[1]
    z = point[2]

    H, S, V = np.array(rgb_to_hsv(m[0], m[1], m[2]))*255
    if -0.2<x<0.2 and -0.26<y<0.26 and -0.8<z<0.6:
        if np.linalg.norm(point-[0,0,0]) <= 0.0001:
            return False

        heightthre = 0.05
        if kpset[2:6,1].mean() != 0:
            heightthre = kpset[2:6,1]
            # heightthre = heightthre[heightthre != 0].mean()
            heightthre = heightthre[heightthre != 0].max() + 0.01

        farhandleflag = 1
        leftvalid = np.linalg.norm(kpset[0] - [0, 0, 0]) != 0
        rightvalid = np.linalg.norm(kpset[1] - [0, 0, 0]) != 0

        if leftvalid and rightvalid:
            if (x < kpset[0, 0] - 0.04) or (x > kpset[1, 0] + 0.04):
                return False

        if y < heightthre:
            return True

        else:
            if 220 > H > 180:
                return True
            else:
                return False
    else: return False

def PointFilteringSep(pcxyz, pcrgb, pckp):
    point_valid_ind = []
    for point_i in range(len(pcxyz)):
        if checkvalid(pcrgb[point_i], pckp, pcxyz[point_i].astype(np.float32)):
            point_valid_ind.append(point_i)

    return point_valid_ind


'''
Functions to load simulation data

'''
from torch.utils.data import Dataset
from sklearn.utils import shuffle
import h5py
from util.SimulatedData import RandomRotateY, RandomScale
import random

class General_PartKPDataLoader_HDF5(Dataset):
    """
        Dataloader for loading partially observed point cloud and keypoints. Note that the occluded point clouds may have different size, we need to do padding in order to form the batches
        The dataloader will return:
            point_xyz: partially-observed normalized simulated point cloud
            point_kp: fully-observed (31) normalized keypoint positions
            point_ref: reference (left-hand position) point for normalization
    """

    def __init__(self,
                 H5dataPath, num_points = 1024, padding = "replace",augrot=False, augocc=False, augsca=False, ref = "left", kp_dict_file=None, numkp= 2):

        # self.data = h5py.File(H5dataPath, 'r')
        data = h5py.File(H5dataPath, 'r')
        self.num_points = num_points
        self.padding = padding
        self.numkp = numkp

        with (open(kp_dict_file, "rb")) as openfile:
            self.kp_ind_dict = pickle.load(openfile)

        self.kp_ind = self.kp_ind_dict[numkp]

        frame_step = 0
        # self.num_scenarios, self.num_frames, _, _ = self.data["refPos"][:].shape
        self.num_scenarios, self.num_frames, _, _ = data["refPos"][:].shape
        self.num_samples = self.num_scenarios * (self.num_frames - frame_step)



        self.indices = [(scenario_index, frame_index)
                        for scenario_index in range(0, self.num_scenarios)
                        for frame_index in range(0, self.num_frames - frame_step)]
        self.indices = shuffle(self.indices)
        #
        # # Number of generated epochs (increased when the complete dataset has been generated/returned)
        # self.epoch_count = 0
        self.augrot = augrot
        self.augocc = augocc
        self.augsca = augsca
        self.ref = ref

        self.pc = data["cleanPC"][:]
        # self.kp = data["kpGt"][:]
        self.kp = data["cleanMesh"][:][:,:,self.kp_ind,:]
        self.lefthandPos = data["refPos"][:]
        # self.righthandPos = data["righthandPos"][:]


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # read frame id
        scenario_index = index // self.num_frames
        frame_index = index % self.num_frames
        # read the fully observed simulation point cloud and the corresponding ground-truth keypoint position
        # point_xyz = self.data["cleanPC"][scenario_index, frame_index]
        # point_kp = self.data["kpGt"][scenario_index, frame_index]
        point_xyz = self.pc[scenario_index, frame_index]
        point_kp = self.kp[scenario_index, frame_index]

        point_xyz_container = np.zeros((self.num_points, 4))

        # read the handle for nomalization
        if self.ref == "left":
            # point_ref = self.data["lefthandPos"][scenario_index, frame_index]
            point_ref = self.lefthandPos[scenario_index, frame_index]
        elif self.ref == "right":
            # point_ref = self.data["righthandPos"][scenario_index, frame_index]
            # point_ref = self.righthandPos[scenario_index, frame_index]
            raise NotImplementedError("To be implemented")

        point_xyz -= point_ref
        point_kp -= point_ref

        num_mt_map = self.num_points
        if self.augrot is True:
            angle = np.random.uniform(0, 2 * np.pi)
            point_xyz = RandomRotateY(point_xyz,angle)
            point_kp = RandomRotateY(point_kp,angle)
        if self.augocc is True:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_xyz)
            diameter = np.linalg.norm(
                np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
            camera = [0, 0, diameter]
            radius = diameter * 100
            _, pt_map = pcd.hidden_point_removal(camera, radius)

            pt_map = pt_map[:self.num_points]
            num_mt_map = len(pt_map)

            pcd = pcd.select_by_index(pt_map)  # can we do double sample?
            # pcd = pcd.select_down_sample(pt_map) # can we do double sample?
            point_xyz = np.array(pcd.points)
        if self.augsca is True:
            sx = random.random() + 0.5
            sy = random.random() + 0.5
            sz = random.random() + 0.5
            point_xyz = RandomScale(point_xyz, sx, sy, sz)
            point_kp = RandomScale(point_kp, sx, sy, sz)

        # padding
        point_xyz_container[:num_mt_map, :3] = point_xyz[:num_mt_map]

        if self.padding == "replace":
            choice = np.random.choice(num_mt_map, self.num_points-num_mt_map, replace=True)
            point_xyz_container[num_mt_map:, :3] = point_xyz[choice]
            return point_xyz_container[:,:3], point_kp, point_ref, scenario_index, frame_index
        elif self.padding == "padzero":
            return point_xyz_container[:,:3], point_kp, point_ref, scenario_index, frame_index
        elif self.padding == "padzeroflag":
            point_xyz_container[:num_mt_map, 3] = 1
            return point_xyz_container, point_kp, point_ref, scenario_index, frame_index

class General_PartKPDataLoader_dyn_HDF5(Dataset):
    """
    TODO: Change description later
        Dataloader for loading partially observed point cloud and keypoints. Note that the occluded point clouds may have different size, we need to do padding in order to form the batches
        The dataloader will return:
            point_xyz_t1: partially-observed normalized simulated point cloud
            point_xyz_t2: partially-observed normalized simulated point cloud
            point_kp: fully-observed (31) normalized keypoint positions
            point_ref: reference (left-hand position) point for normalization
    """

    def __init__(self,
                 H5dataPath, num_points = 1024, padding = "replace",augrot=False, augocc=False, augsca=False, ref = "left", kp_dict_file=None, numkp= 2,frame_step=5):

        # self.data = h5py.File(H5dataPath, 'r')
        data = h5py.File(H5dataPath, 'r')
        self.num_points = num_points
        self.padding = padding
        self.numkp = numkp

        with (open(kp_dict_file, "rb")) as openfile:
            self.kp_ind_dict = pickle.load(openfile)

        self.kp_ind = self.kp_ind_dict[numkp]

        self.frame_step = frame_step
        # self.num_scenarios, self.num_frames, _, _ = self.data["refPos"][:].shape
        self.num_scenarios, self.num_frames, _, _ = data["refPos"][:].shape
        self.num_samples = self.num_scenarios * (self.num_frames - self.frame_step)



        self.indices = [(scenario_index, frame_index)
                        for scenario_index in range(0, self.num_scenarios)
                        for frame_index in range(0, self.num_frames - self.frame_step)]
        self.indices = shuffle(self.indices)
        #
        # # Number of generated epochs (increased when the complete dataset has been generated/returned)
        # self.epoch_count = 0
        self.augrot = augrot
        self.augocc = augocc
        self.augsca = augsca
        self.ref = ref

        self.pc = data["cleanPC"][:]
        # self.kp = data["kpGt"][:]
        self.kp = data["cleanMesh"][:][:,:,self.kp_ind,:]
        self.lefthandPos = data["refPos"][:]
        # self.righthandPos = data["righthandPos"][:]


    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # read frame id
        scenario_index = index // (self.num_frames - self.frame_step)
        frame_index = index % (self.num_frames - self.frame_step)
        # t1
        point_xyz_t1 = self.pc[scenario_index, frame_index]
        point_kp_t1 = self.kp[scenario_index, frame_index]
        # t2
        point_xyz_t2 = self.pc[scenario_index, frame_index+self.frame_step]
        point_kp_t2 = self.kp[scenario_index, frame_index+self.frame_step]

        point_xyz_container_t1 = np.zeros((self.num_points, 4))
        point_xyz_container_t2 = np.zeros((self.num_points, 4))

        print(f"point xyz container t1 shape: {point_xyz_container_t1}")
        print(f"point xyz container t2 shape: {point_xyz_container_t2}")

        # read the handle for nomalization
        if self.ref == "left":
            # point_ref = self.data["lefthandPos"][scenario_index, frame_index]
            point_ref_t1 = self.lefthandPos[scenario_index, frame_index]
            point_ref_t2 = self.lefthandPos[scenario_index, frame_index+self.frame_step]
        elif self.ref == "right":
            # point_ref = self.data["righthandPos"][scenario_index, frame_index]
            # point_ref = self.righthandPos[scenario_index, frame_index]
            raise NotImplementedError("To be implemented")

        point_xyz_t1 -= point_ref_t1
        point_kp_t1 -= point_ref_t1

        point_xyz_t2 -= point_ref_t2
        point_kp_t2 -= point_ref_t2

        num_mt_map_t1 = self.num_points
        num_mt_map_t2 = self.num_points
        if self.augrot is True:
            angle = np.random.uniform(0, 2 * np.pi)
            #t1
            point_xyz_t1 = RandomRotateY(point_xyz_t1,angle)
            point_kp_t1 = RandomRotateY(point_kp_t1,angle)
            #t2
            point_xyz_t2 = RandomRotateY(point_xyz_t2,angle)
            point_kp_t2 = RandomRotateY(point_kp_t2,angle)
        if self.augocc is True:
            pcd_t1 = o3d.geometry.PointCloud()
            pcd_t2 = o3d.geometry.PointCloud()
            pcd_t1.points = o3d.utility.Vector3dVector(point_xyz_t1)
            pcd_t2.points = o3d.utility.Vector3dVector(point_xyz_t2)
            # use same camera for t1 and t2
            diameter = np.linalg.norm(
                np.asarray(pcd_t1.get_max_bound()) - np.asarray(pcd_t1.get_min_bound()))
            camera = [0, 0, diameter]
            radius = diameter * 100

            _, pt_map_t1 = pcd_t1.hidden_point_removal(camera, radius)
            _, pt_map_t2 = pcd_t2.hidden_point_removal(camera, radius)

            pt_map_t1 = pt_map_t1[:self.num_points]
            num_mt_map_t1 = len(pt_map_t1)

            pt_map_t2 = pt_map_t2[:self.num_points]
            num_mt_map_t2 = len(pt_map_t2)

            pcd_t1 = pcd_t1.select_by_index(pt_map_t1)
            pcd_t2 = pcd_t2.select_by_index(pt_map_t1)
            # pcd = pcd.select_down_sample(pt_map) # can we do double sample?
            point_xyz_t1 = np.array(pcd_t1.points)
            point_xyz_t2 = np.array(pcd_t2.points)
        if self.augsca is True:
            sx = random.random() + 0.5
            sy = random.random() + 0.5
            sz = random.random() + 0.5

            point_xyz_t1 = RandomScale(point_xyz_t1, sx, sy, sz)
            point_kp_t1 = RandomScale(point_kp_t1, sx, sy, sz)

            point_xyz_t2 = RandomScale(point_xyz_t2, sx, sy, sz)
            point_kp_t2 = RandomScale(point_kp_t2, sx, sy, sz)

        print(f"point xyz container t1 shape: {point_xyz_container_t1}")
        print(f"point xyz container t2 shape: {point_xyz_container_t2}")
        print(f"num_mt_map_t1: {num_mt_map_t1}")
        print(f"num_mt_map_t2: {num_mt_map_t2}")

        # padding
        point_xyz_container_t1[:num_mt_map_t1, :3] = point_xyz_t1[:num_mt_map_t1]
        point_xyz_container_t2[:num_mt_map_t2, :3] = point_xyz_t2[:num_mt_map_t2]

        if self.padding == "replace":
            choice_t1 = np.random.choice(num_mt_map_t1, self.num_points-num_mt_map_t1, replace=True)
            point_xyz_container_t1[num_mt_map_t1:, :3] = point_xyz_t1[choice_t1]

            choice_t2 = np.random.choice(num_mt_map_t2, self.num_points-num_mt_map_t2, replace=True)
            point_xyz_container_t2[num_mt_map_t2:, :3] = point_xyz_t2[choice_t2]

            return point_xyz_container_t1[:,:3], point_xyz_container_t2[:,:3], point_kp_t1, point_kp_t2, point_ref_t1, point_ref_t2, scenario_index, frame_index
        elif self.padding == "padzero":
            return point_xyz_container_t1[:,:3], point_xyz_container_t2[:,:3], point_kp_t1, point_kp_t2, point_ref_t1, point_ref_t2, scenario_index, frame_index
        elif self.padding == "padzeroflag":
            point_xyz_container_t1[:num_mt_map_t1, 3] = 1
            point_xyz_container_t2[:num_mt_map_t2, 3] = 1
            return  point_xyz_container_t1, point_xyz_container_t2, point_kp_t1, point_kp_t2, point_ref_t1, point_ref_t2, scenario_index, frame_index

def correctHandle(markerarray_miss, l=True, r=True, loffset=np.array([-0.02, 0, -0.07]), roffset=np.array([0, 0, -0.06])):
    # input: markerarray -> ndarray (numframe, nummarker, 3)
    markerarray = np.copy(markerarray_miss)
    leftmarker = markerarray[:,0,:]
    rightmarker = markerarray[:,1,:]



    if l is True:


        leftmiss = np.argwhere(np.linalg.norm(leftmarker, axis=1) < 0.0001).reshape(-1)
        leftexist = np.argwhere(np.linalg.norm(leftmarker, axis=1) >= 0.0001).reshape(-1)
        leftspeed = (leftmarker[leftexist.max()] - leftmarker[leftexist.min()]) / (leftexist.max() - leftexist.min())

        for i_left in leftmiss:


            if i_left < leftexist.min():
                step = leftexist.min() - i_left
                leftmarker[i_left] = leftmarker[leftexist.min()] - leftspeed * step
                continue
            # if i_left > leftexist.max():
            #     step = i_left - leftexist.max()
            #     leftmarker[i_left] = leftmarker[leftexist.max()] + leftspeed * step
            #     continue

            leftmarker[i_left] = leftmarker[i_left-1] + leftspeed

    if r is True:
        rightmiss = np.argwhere(np.linalg.norm(rightmarker, axis=1) < 0.0001).reshape(-1)
        rightexist = np.argwhere(np.linalg.norm(rightmarker, axis=1) >= 0.0001).reshape(-1)
        rightspeed = (rightmarker[rightexist.max()] - rightmarker[rightexist.min()]) / (
                    rightexist.max() - rightexist.min())

        if np.linalg.norm(rightspeed) < 0.001:
            rightmarker[:] = rightmarker[rightexist].mean(axis=0)
        else:
            for i_right in rightmiss:
                if i_right < rightexist.min():
                    step = rightexist.min() - i_right
                    rightmarker[i_right] = rightmarker[rightexist.min()] - rightspeed * step
                    continue
                # if i_right > rightexist.max():
                #     step = i_right - rightexist.max()
                #     rightmarker[i_right] = rightmarker[rightexist.max()] + rightspeed * step
                #     continue

                rightmarker[i_right] = rightmarker[i_right - 1] + rightspeed

    numFrame = markerarray.shape[0]
    for frameid in range(numFrame):
        markerarray[frameid][0, :] = markerarray[frameid][0,:] + loffset
        markerarray[frameid][1, :] = markerarray[frameid][1,:] + roffset

    return markerarray
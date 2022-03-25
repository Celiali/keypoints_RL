
import numpy as np
import h5py
import pickle
import open3d as o3d
import os
from util.Datasets import readTopo, GetDataset, GetDenseFullPCL
from util import SimulatedData


def main():
    # taskIdlist = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # taskIdlist = [12,13,14,15,16,17,18,19,20]
    taskIdlist = [8,18,10,20]

    set_namelist = ["train", "valid", "test"]
    # set_name = set_namelist[0]
    rootpath = "data/h5data/tasks"
    #
    # meshtopofile = os.path.join(rootpath, "topo_{}.pkl".format(set_name))
    # topoArray = readTopo(meshtopofile)

    ptNum = 2048 # size of the generated point cloud
    tolerence = 0.03
    kpNum = 31


    for taskId in taskIdlist:
        print(taskId)
        for set_name in set_namelist:
            meshtopofile = os.path.join(rootpath, "topo_{}.pkl".format(set_name))
            topoArray = readTopo(meshtopofile)

            # get the database of a specific task
            database, task = GetDataset(taskId, set_name, rootpath)
            # newPCLDataset = np.zeros((database.num_scenarios, database.num_frames, ptNum, 3))
            # numTotalFrame = database.num_scenarios * database.num_frames

            # h5 file for the clean pcl
            h5file_cleanpc_name = os.path.join(rootpath, "{}_{}_full.h5".format(taskId, set_name))
            db = h5py.File(h5file_cleanpc_name, 'w')
            # db.create_dataset("posCloth", (database.num_scenarios, database.num_frames,database.num_mesh_points,3), dtype='float')
            db.create_dataset("cleanMesh", (database.num_scenarios, database.num_frames, 1277, 3))
            db.create_dataset("cleanPC", (database.num_scenarios, database.num_frames, ptNum, 3))
            db.create_dataset("kpGt", (database.num_scenarios, database.num_frames, kpNum, 3))
            db.create_dataset("refPos", (database.num_scenarios, database.num_frames, 1, 3))

            for scenario_index in range(database.num_scenarios):
                print(scenario_index)
                for frame_index in range(database.num_frames):
                    frame = database.scenario(scenario_index).frame(frame_index)
                    # if task.effector_motion.name == "Ball" and task.left_hand_motion.name == "Fixed":
                    #     # Action is performed on the sphere ball
                    #     currentRef = frame.get_effector_pose()
                    # elif task.effector_motion.name == "NoBall" and task.left_hand_motion.name != "Fixed":
                    #     # Action is performed on the left hand
                    #     currentRef = frame.get_left_hand_position()
                    # else:
                    #     raise ("Not supported actions")

                    # always use the left hand to convert the point cloud to canonical frame
                    currentRef = frame.get_left_hand_position()

                    point_xyz = frame.get_cloth_all()  # (1277, 3)
                    # point_xyz = point_xyz - hand_left_xyz_current
                    point_kp = point_xyz[SimulatedData.keypoint_indices]  # (31, 3)
                    # clean resampled point cloud
                    point_pc = GetDenseFullPCL(point_xyz, topoArray, ptNum, tolerence)


                    db["cleanMesh"][scenario_index, frame_index, ...] = point_xyz
                    db["kpGt"][scenario_index, frame_index, ...] = point_kp
                    db["cleanPC"][scenario_index, frame_index, ...] = point_pc
                    db["refPos"][scenario_index, frame_index, ...] = currentRef#[:,:3]
            db.close()

if __name__ == '__main__':
    main()
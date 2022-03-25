'''
    Generate keypoint list for different resolution and save in a dictionary
'''

import numpy as np
import h5py
import pickle
import open3d as o3d
import os
from util.Datasets import readTopo, GetDataset, GetDenseFullPCL
from util import SimulatedData

def main():
    # n_samples_list = [3, 10, 30, 60, 120, 180]
    n_samples_list = [i for i in range(3,201)]
    kp_dict = {}

    for n_samples in n_samples_list:
        H5dataPath = "/home/zehang/Downloads/s18_stiff_noballin_open_f_ballnomove_noeffector_test.h5"
        data = h5py.File(H5dataPath, 'r')
        meshcloud = data['posCloth'][0,0,:]


        sample_inds = np.zeros(n_samples, dtype='int')
        # Initialise distances to inf
        points_left = np.arange(meshcloud.shape[0])
        dists = np.ones_like(points_left) * float('inf')

        initkp_ind = [756, 1069]#, 1258]
        sample_inds[:len(initkp_ind)] = initkp_ind
        points_left = np.delete(points_left, initkp_ind)

        # Iteratively select points for a maximum of n_samples
        for i in range(len(initkp_ind), n_samples):
            # Find the distance to the last added point in selected
            # and all the others
            last_added = sample_inds[i - 1]

            dist_to_last_added_point = ((meshcloud[last_added] - meshcloud[points_left]) ** 2).sum(-1)  # [P - i]

            # If closer, updated distances
            dists[points_left] = np.minimum(dist_to_last_added_point,dists[points_left])  # [P - i]

            # We want to pick the one that has the largest nearest neighbour
            # distance to the sampled points
            selected = np.argmax(dists[points_left])
            sample_inds[i] = points_left[selected]

            # Update points_left
            points_left = np.delete(points_left, selected)

        kp_dict[n_samples] = sample_inds

        # sample_inds = [
        #     # Frontsample_inds
        #     4, 127, 351, 380, 395, 557, 535, 550, 756, 783, 818, 1258,
        #     # Back
        #     150, 67, 420, 436, 920, 952, 1069, 1147, 1125, 1099, 929, 464,
        #     # Left
        #     142, 851, 1178,
        #     # Right
        #     49, 509, 1000,
        #     # Bottom
        #     641
        # ]

        # sample_kp = meshcloud[sample_inds,:]
        # sample_kp = RandomRotateY(sample_kp, 30)
        #
        # vis = o3d.visualization.VisualizerWithEditing()
        # vis.create_window()
        # opt = vis.get_render_option()
        # opt.show_coordinate_frame = True
        # opt.point_size=10
        #
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(sample_kp)
        # pcd.colors = o3d.utility.Vector3dVector(np.zeros_like(sample_kp))
        #
        # vis.add_geometry(pcd)
        # vis.run()
        #
        #
        # print("hello")

    try:
        dictfile = open('./kp_ind_list.pickle', 'wb')
        pickle.dump(kp_dict, dictfile)
        dictfile.close()

    except:
        print("Something went wrong")

if __name__ == '__main__':
    main()
"""
Code for loading the different datasets.
Examples:
    s1_soft_ballin_f_f_ballmove_*
    s3_soft_ballin_open_f_ballnomove_noeffector_*
"""

from util import SimulatedData

from enum import Enum
import os
import pickle as pkl
import point_cloud_utils as pcu

class BagContent(Enum):
    Empty = 0
    BallInside = 1

    def filename(self):
        switcher = {
            BagContent.Empty: "noballin",
            BagContent.BallInside: "ballin",
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("BagContent is not handled in filename() function:", self)
        else:
            return result


class HandMotion(Enum):
    Fixed = 0
    Released = 1
    Open = 2
    Circle = 3
    Lift = 4

    def filename(self):
        switcher = {
            HandMotion.Fixed: "f",
            HandMotion.Released: "r",
            HandMotion.Open: "open",
            HandMotion.Circle: "circle",
            HandMotion.Lift: "lift"
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("HandMotion is not handled in filename() function:", self)
        else:
            return result


class EffectorMotion(Enum):
    NoBall = 0
    Ball = 1

    def filename(self):
        switcher = {
            EffectorMotion.NoBall: "ballnomove_noeffector",
            EffectorMotion.Ball: "ballmove",
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("EffectorMotion is not handled in filename() function:", self)
        else:
            return result


class Subset(Enum):
    Training = 0
    Validation = 1
    Test = 2

    def filename(self):
        switcher = {
            Subset.Training: "train",
            Subset.Validation: "valid",
            Subset.Test: "test"
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("Subset is not handled in filename() function:", self)
        else:
            return result

    @staticmethod
    def from_name(set_name):
        for subset in Subset:
            if subset.filename() == set_name:
                return subset
        return None


class BagStiffness(Enum):
    Soft = 0
    Stiff = 1

    def filename(self):
        switcher = {
            BagStiffness.Soft: "soft",
            BagStiffness.Stiff: "stiff",
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("BagStiffness is not handled in filename() function:", self)
        else:
            return result


class Action(Enum):
    PushObject = 0
    MoveHandleCircular = 1
    OpenBag = 2
    LiftBag = 3

    def plot_name(self):
        switcher = {
            Action.PushObject: "Pushing an Object into the Bag",
            Action.MoveHandleCircular: "Handle Motion Along Circular Trajectory",
            Action.OpenBag: "Opening the Bag",
            Action.LiftBag: "Lifting the Bag",
        }
        result = switcher.get(self, None)
        if result is None:
            raise ValueError("Action is not handled in plot_name() function:", self)
        else:
            return result


class TaskDataset:
    def __init__(self,
                 index: int = 1,
                 bag_stiffness: BagStiffness = BagStiffness.Soft,
                 bag_content: BagContent = BagContent.Empty,
                 left_hand_motion: HandMotion = HandMotion.Fixed,
                 right_hand_motion: HandMotion = HandMotion.Fixed,
                 effector_motion: EffectorMotion = EffectorMotion.NoBall
                 ):
        self.index = index
        self.bag_stiffness = bag_stiffness
        self.bag_content = bag_content
        self.left_hand_motion = left_hand_motion
        self.right_hand_motion = right_hand_motion
        self.effector_motion = effector_motion
        self.isdemo = False

    def action(self):
        if self.effector_motion == EffectorMotion.Ball:
            return Action.PushObject
        if self.left_hand_motion == HandMotion.Circle:
            return Action.MoveHandleCircular
        if self.left_hand_motion == HandMotion.Open:
            return Action.OpenBag
        if self.left_hand_motion == HandMotion.Lift:
            return Action.LiftBag
        raise NotImplementedError("Could not determine action for task", self)

    def filename(self, subset: Subset) -> str:
        if self.isdemo == True:
            return f"s{self.index}_{subset.filename()}.h5"
        else:
            return f"s{self.index}_{self.bag_stiffness.filename()}_{self.bag_content.filename()}_" + \
                   f"{self.left_hand_motion.filename()}_{self.right_hand_motion.filename()}_" + \
                   f"{self.effector_motion.filename()}_{subset.filename()}.h5"

    def path_to_dataset(self, root_path: str, subset: Subset) -> str:
        return os.path.join(root_path, self.filename(subset))

    def path_to_topodict(self, root_path: str, subset: Subset) -> str:
        topo_filename = f"topo_{subset.filename()}.pkl"
        return os.path.join(root_path, topo_filename)

s1 = TaskDataset(index=1,
                 bag_content=BagContent.BallInside,
                 left_hand_motion=HandMotion.Fixed,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.Ball)

s2 = TaskDataset(index=2,
                 bag_content=BagContent.Empty,
                 left_hand_motion=HandMotion.Fixed,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.Ball)

s3 = TaskDataset(index=3,
                 bag_content=BagContent.BallInside,
                 left_hand_motion=HandMotion.Circle,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.NoBall)

s4 = TaskDataset(index=4,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Circle,
                  right_hand_motion=HandMotion.Fixed,
                  effector_motion=EffectorMotion.NoBall)

s5 = TaskDataset(index=5,
                 bag_content=BagContent.BallInside,
                 left_hand_motion=HandMotion.Circle,
                 right_hand_motion=HandMotion.Released,
                 effector_motion=EffectorMotion.NoBall)

s6 = TaskDataset(index=6,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Circle,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

s7 = TaskDataset(index=7,
                 bag_content=BagContent.BallInside,
                 left_hand_motion=HandMotion.Open,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.NoBall)

s8 = TaskDataset(index=8,
                 bag_content=BagContent.Empty,
                 left_hand_motion=HandMotion.Open,
                 right_hand_motion=HandMotion.Fixed,
                 effector_motion=EffectorMotion.NoBall)

s9 = TaskDataset(index=9,
                  bag_content=BagContent.BallInside,
                  left_hand_motion=HandMotion.Lift,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

s10 = TaskDataset(index=10,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Lift,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

s11 = TaskDataset(index=11,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.BallInside,
                  left_hand_motion=HandMotion.Fixed,
                  right_hand_motion=HandMotion.Fixed,
                  effector_motion=EffectorMotion.Ball)

s12 = TaskDataset(index=12,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Fixed,
                  right_hand_motion=HandMotion.Fixed,
                  effector_motion=EffectorMotion.Ball)

s13 = TaskDataset(index=13,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.BallInside,
                  left_hand_motion=HandMotion.Circle,
                  right_hand_motion=HandMotion.Fixed,
                  effector_motion=EffectorMotion.NoBall)

s14 = TaskDataset(index=14,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Circle,
                  right_hand_motion=HandMotion.Fixed,
                  effector_motion=EffectorMotion.NoBall)

s15 = TaskDataset(index=15,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.BallInside,
                  left_hand_motion=HandMotion.Circle,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

s16 = TaskDataset(index=16,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Circle,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

s17 = TaskDataset(index=17,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.BallInside,
                  left_hand_motion=HandMotion.Open,
                  right_hand_motion=HandMotion.Fixed,
                  effector_motion=EffectorMotion.NoBall)

s18 = TaskDataset(index=18,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Open,
                  right_hand_motion=HandMotion.Fixed,
                  effector_motion=EffectorMotion.NoBall)

s19 = TaskDataset(index=19,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.BallInside,
                  left_hand_motion=HandMotion.Lift,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

s20 = TaskDataset(index=20,
                  bag_stiffness=BagStiffness.Stiff,
                  bag_content=BagContent.Empty,
                  left_hand_motion=HandMotion.Lift,
                  right_hand_motion=HandMotion.Released,
                  effector_motion=EffectorMotion.NoBall)

tasks = [
    s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14,
    s15, s16, s17, s18, s19, s20
]

def get_task_by_index(task_index: int):
    for task in tasks:
        if task.index == task_index:
            return task
    return None

def GetDataset(taskId, setName, rootPath):
    '''
    Retrieve the H5 dataset by using the task id
    :param taskId: task id: from 1 to 20
    :return: dataset with task id
    '''
    task_index = taskId
    set_name = setName
    rootPath = rootPath
    task = get_task_by_index(task_index)
    subset = Subset.from_name(set_name)
    path_to_dataset = task.path_to_dataset(rootPath, subset)
    path_to_topodict = task.path_to_topodict(rootPath, subset)
    SIM_MESH_DATASET = SimulatedData.SimulatedData.load(path_to_topodict, path_to_dataset)
    return SIM_MESH_DATASET, task

# function to read the topo
def readTopo(meshtopofile):
    topodict = None
    with open(meshtopofile, 'rb') as pickle_file:
        topodict = pkl.load(pickle_file)
    return topodict


def GetDenseFullPCL(meshArray, topoArray, ptNum, tolerence=0.03, sampleMethod="Random"):
    '''
    Get full clean point cloud by using mesh array and topo file
    :param meshArray: mesh of a certain frame
    :param topoArray: topology file of a certain frame
    :param ptNum: number of points in the generated point cloud
    :param sampleMethod: Method of point cloud sampling
    :return: A clean full point cloud with ptNum points, ndarray: (ptNum, 3)
    '''

    if sampleMethod == "PossionDisk":
        f_i, bary_c = pcu.sample_mesh_poisson_disk(meshArray, topoArray[20], int(np.ceil(ptNum*(1+tolerence))), sample_num_tolerance=tolerence-0.01)
        v_poisson = pcu.interpolate_barycentric_coords(topoArray[20], f_i, bary_c, meshArray)
        v_poisson = v_poisson[:ptNum]
    else:
        f_i, bary_c = pcu.sample_mesh_random(meshArray, topoArray[20], ptNum)
        v_poisson = pcu.interpolate_barycentric_coords(topoArray[20], f_i, bary_c, meshArray)
    # import matplotlib.pyplot as plt
    # plt.scatter(bary_c[:,0], bary_c[:,1])
    # # plt.scatter(v_poisson[:,0], v_poisson[:,1])
    # plt.show()

    # v_dense, n_dense = pcu.sample_mesh_poisson_disk(meshArray, topoArray[20], 2048, sample_num_tolerance=0.0001)
    return v_poisson


if __name__ == '__main__':
    # Verify that the files for the task datasets exist
    tasks_path = "./h5data/tasks/"

    for s in Subset:
        # Check for topology files
        topo_path = os.path.join(tasks_path, f"topo_{s.filename()}.pkl")
        if not os.path.exists(topo_path):
            print("Could not find topo file:", topo_path)

        # Check for data files
        for task in tasks:
            filename = task.filename(s)
            full_path = os.path.join(tasks_path, filename)
            if not os.path.exists(full_path):
                print("Could not find data file:", full_path)

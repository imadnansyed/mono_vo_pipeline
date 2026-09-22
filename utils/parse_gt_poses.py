# will parse ground truth poses from KITTI dataset

import numpy as np

def parse_gt_poses():
    path = r"E:\Workspace\PC\AI Engineer\cv\projects\VO and SLAM\vo_pipeline\data\poses\00.txt"
    
    poses = np.zeros((100, 4, 4), dtype=np.float32)
    
    with open(path, 'r') as f:
        lines = f.readlines()[:100]
        
        for i, line in enumerate(lines):
            pose = line.split(sep=" ")
            pose = [float(x) for x in pose]
            pose = np.array(pose).reshape(3,4)
            # now convert pose to 4x4 matrix
            pose = np.vstack((pose, [0, 0, 0, 1]))
            poses[i] = pose
            
    print(poses[0])
    return poses


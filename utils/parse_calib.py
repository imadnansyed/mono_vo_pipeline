import numpy as np
# will keep only P2 for monocular VO
def parse_calib_file():
    path = r"E:\Workspace\PC\AI Engineer\cv\projects\VO and SLAM\vo_pipeline\data\00\calib.txt"
    
    with open(path, 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            if line.startswith('P2:'):
                P2 = line.split(sep=":")[1].strip()
                P2 = [float(x) for x in P2.split()]
                P2 = np.array(P2).reshape(3,4)
                K = np.asarray(P2[:, :3], dtype=np.float32)
                return P2, K

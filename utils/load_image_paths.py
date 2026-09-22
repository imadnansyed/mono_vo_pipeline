# will only load images paths

import os


def load_image_paths():
    path = r"E:\Workspace\PC\AI Engineer\cv\projects\VO and SLAM\vo_pipeline\data\00\image_2"
    
    image_paths = []
    for file in os.listdir(path):
        if file.endswith(".png"):
            image_paths.append(os.path.join(path, file))
    
    image_paths.sort()
    return image_paths
        
    

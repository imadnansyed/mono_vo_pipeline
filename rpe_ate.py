

# now lets estimate the APE and ATE

import subprocess


# subprocess.run([
#     "evo_rpe",
#     "kitti",
#     "ground_truth.txt",
#     "estimated_poses.txt",
#     "-va",
#     "--plot"
# ])

subprocess.run([
    "evo_ape",
    "kitti",
    "ground_truth.txt",
    "estimated_poses.txt",
    "-va",
    "--align",
    "--correct_scale",
    "--plot"
])
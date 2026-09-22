import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

IMAGE_DIR = Path(r"E:\Workspace\PC\AI Engineer\cv\projects\VO and SLAM\vo_pipeline\data\00\image_2")
POSE_FILE = Path(r"E:\Workspace\PC\AI Engineer\cv\projects\VO and SLAM\vo_pipeline\estimated_poses.txt")
OUTPUT_VIDEO = "vo_visualization.mp4"

FPS = 10


# ============================================================
# LOAD POSES
# ============================================================

def load_poses(file_path):
    poses = []

    with open(file_path, "r") as f:
        for line in f:
            values = np.fromstring(line, sep=" ")

            if len(values) != 12:
                continue

            T = np.eye(4, dtype=np.float64)
            T[:3, :4] = values.reshape(3, 4)

            poses.append(T)

    return poses


poses = load_poses(POSE_FILE)

print(f"Loaded {len(poses)} poses")


# ============================================================
# CAMERA CENTERS
# ============================================================

camera_centers = []

for T in poses:
    # T is T_wc
    C = T[:3, 3]
    camera_centers.append(C)

camera_centers = np.asarray(camera_centers)


# ============================================================
# IMAGE PATHS
# ============================================================

image_paths = sorted(
    IMAGE_DIR.glob("*.png")
)

if len(image_paths) == 0:
    image_paths = sorted(
        IMAGE_DIR.glob("*.jpg")
    )

print(f"Found {len(image_paths)} images")


num_frames = min(
    len(image_paths),
    len(camera_centers)
)

if num_frames == 0:
    raise RuntimeError("No frames or poses found.")


# ============================================================
# VIDEO WRITER
# ============================================================

first_image = cv2.imread(
    str(image_paths[0])
)

if first_image is None:
    raise RuntimeError(
        f"Could not read {image_paths[0]}"
    )

first_image = cv2.cvtColor(
    first_image,
    cv2.COLOR_BGR2RGB
)

h, w = first_image.shape[:2]


# Matplotlib figure
fig = plt.figure(
    figsize=(16, 7),
    dpi=100
)

ax_image = fig.add_subplot(1, 2, 1)
ax_traj = fig.add_subplot(1, 2, 2)


# ------------------------------------------------------------
# FIX TRAJECTORY AXIS
# ------------------------------------------------------------

x = camera_centers[:, 0]
z = camera_centers[:, 2]

margin = 5

xmin = x.min() - margin
xmax = x.max() + margin

zmin = z.min() - margin
zmax = z.max() + margin


# ============================================================
# CREATE VIDEO
# ============================================================

video_writer = None


for i in range(num_frames):

    # --------------------------------------------------------
    # LEFT: ORIGINAL IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_paths[i])
    )

    if image is None:
        print(f"Skipping frame {i}")
        continue

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    ax_image.clear()

    ax_image.imshow(image)

    ax_image.set_title(
        f"Original Camera Frame {i:03d}"
    )

    ax_image.axis("off")


    # --------------------------------------------------------
    # RIGHT: ESTIMATED TRAJECTORY
    # --------------------------------------------------------

    ax_traj.clear()

    current_x = x[:i + 1]
    current_z = z[:i + 1]

    # trajectory
    ax_traj.plot(
        current_x,
        current_z,
        linewidth=2
    )

    # current camera position
    ax_traj.scatter(
        current_x[-1],
        current_z[-1],
        s=60
    )

    # starting position
    ax_traj.scatter(
        x[0],
        z[0],
        s=50,
        marker="s"
    )

    ax_traj.set_xlim(xmin, xmax)
    ax_traj.set_ylim(zmin, zmax)

    ax_traj.set_xlabel("X position (m)")
    ax_traj.set_ylabel("Z position (m)")

    ax_traj.set_title(
        f"Estimated VO Trajectory — Frame {i:03d}"
    )

    ax_traj.grid(True)

    ax_traj.set_aspect(
        "equal",
        adjustable="box"
    )


    # --------------------------------------------------------
    # FRAME NUMBER
    # --------------------------------------------------------

    fig.suptitle(
        f"Monocular Visual Odometry | Frame {i:03d} / {num_frames - 1:03d}",
        fontsize=16
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # MATPLOTLIB → IMAGE
    # --------------------------------------------------------

    fig.canvas.draw()

    frame = np.asarray(
        fig.canvas.buffer_rgba()
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_RGBA2BGR
    )


    # --------------------------------------------------------
    # CREATE VIDEO WRITER
    # --------------------------------------------------------

    if video_writer is None:

        height, width = frame.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        video_writer = cv2.VideoWriter(
            OUTPUT_VIDEO,
            fourcc,
            FPS,
            (width, height)
        )


    video_writer.write(frame)

    print(
        f"\rCreating frame {i + 1}/{num_frames}",
        end=""
    )


# ============================================================
# CLEANUP
# ============================================================

if video_writer is not None:
    video_writer.release()

plt.close(fig)

print()
print(f"\nVideo saved to:")
print(OUTPUT_VIDEO)
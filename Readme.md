# Monocular Visual Odometry

A monocular Visual Odometry pipeline built from scratch using **Python, OpenCV, NumPy, and KITTI Sequence 00**.

The project estimates the camera trajectory from consecutive monocular images using feature matching, Essential Matrix initialization, triangulation, PnP, and incremental map management.

The estimated trajectory is evaluated against KITTI ground truth using **ATE** and **RPE** with `evo`.

---

## Pipeline

```text
KITTI SEQUENCE

├── 100 × images
│       └── 00/image2/
│
├── Calibration
│       ├── P2
│       └── K = first 3×3 of P2
│
└── Ground Truth
        └── 100 × 3×4 poses
                │
                ▼
        ┌─────────────────────┐
        │ 1. LOAD DATA        │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 2. FIRST FRAME      │
        │ Detect features     │
        │ Store 2D features   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 3. SECOND FRAME     │
        │ Detect features     │
        │ Match 2D ↔ 2D       │
        │ Lowe ratio          │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 4. INITIALIZATION   │
        │ E + RANSAC          │
        │        ↓            │
        │ recoverPose         │
        │        ↓            │
        │ R, t                │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 5. TRIANGULATION    │
        │ 2D ↔ 2D + R,t       │
        │        ↓            │
        │ Initial 3D map      │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 6. NEXT FRAME       │
        │ 3D ↔ 2D             │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 7. PnP + RANSAC     │
        │        ↓            │
        │ Current R,t         │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 8. POSE ESTIMATION  │
        │ Current camera pose │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ 9. MAP MANAGEMENT   │
        │                     │
        │ Mapped features     │
        │        ↓            │
        │ PnP                 │
        │                     │
        │ New features        │
        │        ↓            │
        │ Triangulate         │
        │        ↓            │
        │ Add to map          │
        └──────────┬──────────┘
                   │
                   ▼
              NEXT FRAME
                   │
                   ▼
        ┌─────────────────────┐
        │ 10. EVALUATION      │
        │ Estimated poses     │
        │        ↓            │
        │ Alignment           │
        │        ↓            │
        │ ATE + RPE            │
        └─────────────────────┘
```

---

# Features

* ORB feature detection
* Binary feature descriptors
* BF Hamming feature matching
* Lowe's ratio test
* Essential Matrix estimation with RANSAC
* Relative pose recovery
* Two-view triangulation
* Incremental 3D landmark map
* 3D–2D feature association
* PnP with RANSAC
* Reprojection-error validation
* Incremental camera pose estimation
* KITTI trajectory export
* ATE evaluation
* RPE evaluation
* Frame-by-frame VO visualization

---

# Requirements

* Python 3.9+
* OpenCV
* NumPy
* Matplotlib
* evo

Install the dependencies:

```bash
pip install numpy opencv-python matplotlib evo
```

---

# Dataset

This project uses **KITTI Odometry Sequence 00**.

The dataset is intentionally **not included in this repository**.

Download the KITTI Odometry dataset separately and place the required files inside the project.

The repository expects the following structure:

```text
dataset/
└── 00/
    ├── image_2/
    │   ├── 000000.png
    │   ├── 000001.png
    │   ├── 000002.png
    │   ├── ...
    │   └── 000099.png
    │
    ├── calib.txt
    └── poses/
        └── 00.txt
```

Only the first **100 images** of Sequence 00 are used by the current experiment.

---

# Calibration

The KITTI calibration file contains the projection matrix:

```text
P2
```

The intrinsic camera matrix is extracted from the first `3 × 3` portion of `P2`:

```text
P2 = [ K | t ]
```

Therefore:

```text
K = P2[:, :3]
```

The same camera calibration is used throughout the sequence.

---

# Ground Truth

KITTI provides ground-truth camera poses as `3 × 4` matrices:

```text
[R | t]
```

These are converted into homogeneous `4 × 4` matrices:

```text
T = ┌         ┐
    │ R   t   │
    │ 0   1   │
    └         ┘
```

The ground-truth poses are used only for **evaluation**, not for estimating the VO trajectory.

---

# Project Structure

A typical repository structure is:

```text
vo_pipeline/
│
├── main.py
├── vo_pipeline.py
├── requirements.txt
├── README.md
│
├── utils/
│   ├── parse_calib.py
│   ├── parse_gt_poses.py
│   └── load_image_paths.py
│
├── dataset/
│   └── 00/
│       ├── image_2/
│       ├── calib.txt
│       └── poses/
│
├── estimated_poses.txt
├── ground_truth.txt
│
└── vo_visualization.mp4
```

> The `dataset/` directory should **not be committed to GitHub**. Add it to `.gitignore`.

Example:

```gitignore
dataset/
*.mp4
```

If you want the demonstration video available directly in the repository, you can remove `*.mp4` from `.gitignore` or use GitHub Releases/LFS depending on the file size.

---

# How to Run

## 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd vo_pipeline
```

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install numpy opencv-python matplotlib evo
```

## 4. Download KITTI

Download **KITTI Odometry Sequence 00** and place the required files according to the dataset structure shown above.

Do not upload the dataset to this repository.

## 5. Configure dataset paths

Update the dataset paths in the project configuration/code if necessary.

For example:

```python
IMAGE_DIR = "dataset/00/image_2"
CALIB_FILE = "dataset/00/calib.txt"
GT_FILE = "dataset/00/poses/00.txt"
```

## 6. Run Visual Odometry

```bash
python main.py
```

The pipeline will:

```text
Load images
    ↓
Detect ORB features
    ↓
Match features
    ↓
Initialize using Essential Matrix
    ↓
Triangulate initial landmarks
    ↓
Track 3D landmarks
    ↓
Estimate camera pose using PnP
    ↓
Triangulate new landmarks
    ↓
Update map
    ↓
Save estimated trajectory
```

---

# Output

After execution, the estimated trajectory is saved as:

```text
estimated_poses.txt
```

Ground truth can be exported/loaded as:

```text
ground_truth.txt
```

Both files use KITTI pose format:

```text
r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
```

---

# Evaluation

The trajectory is evaluated using [`evo`](https://github.com/MichaelGrupp/evo).

## ATE

Absolute Trajectory Error is evaluated using **Sim(3) Umeyama alignment** to account for the unknown global scale of monocular VO.

```bash
evo_ape kitti ground_truth.txt estimated_poses.txt \
    -va --align --correct_scale --plot
```

### Current ATE Result

```text
Alignment: Sim(3) Umeyama

Maximum:          21.0947 m
Mean:              9.4860 m
Median:           10.0139 m
Minimum:           0.1993 m
RMSE:             10.6612 m
Standard deviation: 4.8660 m
```

Estimated scale correction:

```text
2.59049
```

---

# RPE

Relative Pose Error evaluates the local frame-to-frame motion accuracy.

```bash
evo_rpe kitti ground_truth.txt estimated_poses.txt \
    -va --plot
```

### Current RPE Result

```text
Alignment: SE(3) Umeyama

Maximum:          0.8468 m
Mean:             0.5660 m
Median:           0.5701 m
Minimum:          0.1204 m
RMSE:             0.6075 m
Standard deviation: 0.2208 m
```

---

# Results Summary

| Metric | Alignment |          RMSE |
| ------ | --------- | ------------: |
| ATE    | Sim(3)    | **10.6612 m** |
| RPE    | SE(3)     |  **0.6075 m** |

The local motion error is considerably smaller than the global trajectory error, indicating accumulated drift over the sequence.

---

# Visualization

A frame-by-frame visualization was generated to demonstrate the VO process.

The video contains:

```text
┌──────────────────────┬──────────────────────┐
│   Original Frame     │    Estimated VO      │
│                      │                      │
│     Frame 000        │     Frame 000        │
│     Frame 001        │     Frame 001        │
│        ...           │        ...           │
│     Frame 099        │     Frame 099        │
└──────────────────────┴──────────────────────┘
```

The visualization allows the estimated trajectory to be inspected as the sequence progresses.

**Demo video:** `utils\vo_visualization.mp4`

---

# Coordinate Convention

The implementation uses:

```text
T_wc = Camera → World
```

where the translation component represents the camera position in the world coordinate system.

OpenCV `solvePnP` returns:

```text
T_cw = World → Camera
```

Therefore the estimated PnP transform is inverted before storing the global camera pose:

```text
T_wc = inverse(T_cw)
```

Frame `0` is used as the initial world coordinate system.

---

# Current Limitations

This is an incremental monocular VO implementation and therefore has several limitations:

* Monocular scale is initially unknown.
* Tracking can fail when too few landmarks remain.
* Pose errors accumulate over time.
* No loop closure is implemented.
* No global pose graph optimization is implemented.
* No full bundle adjustment is currently used.
* Landmark quality depends on feature matching and triangulation.
* KITTI ground truth is used for evaluation only.

These limitations provide opportunities for future improvements.

---

# Future Improvements

Possible extensions include:

* Keyframe selection
* Better landmark culling
* Local bundle adjustment
* Ceres-based optimization
* Motion-model prediction
* More robust feature tracking
* Loop closure detection
* Pose graph optimization
* Global bundle adjustment
* Stereo/depth-assisted scale recovery

---

# Project Status

```text
[✓] Dataset loading
[✓] Camera calibration
[✓] ORB feature detection
[✓] Feature matching
[✓] Essential Matrix initialization
[✓] Pose recovery
[✓] Initial triangulation
[✓] 3D landmark map
[✓] PnP + RANSAC
[✓] Incremental pose estimation
[✓] Map management
[✓] Trajectory export
[✓] ATE evaluation
[✓] RPE evaluation
[✓] Frame-by-frame visualization
[ ] Loop closure
[ ] Pose graph optimization
[ ] Global bundle adjustment
```

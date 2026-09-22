import cv2
import numpy as np
from utils.parse_calib import parse_calib_file
from utils.parse_gt_poses import parse_gt_poses
from utils.load_image_paths import load_image_paths
import subprocess


class MonocularVO:
    def __init__(self):
        self.P2, self.K = parse_calib_file()
        self.poses = parse_gt_poses()
        self.image_paths = load_image_paths()
        self.map_points = {}
        self.estimated_poses = []
        self.orb = cv2.ORB_create(nfeatures=2000)

    def detect_and_compute(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, des1, des2):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        matches = bf.knnMatch(
            des1,
            des2,
            k=2
        )

        good_matches = []

        ratio = 0.75

        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)

        return good_matches

    def compute_pose(self, pts1, pts2, good_matches):

        

        E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        self.K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

        if E is None:
            raise RuntimeError("Essential matrix estimation failed.")

        _, R, t, pose_mask = cv2.recoverPose(
            E, pts1, pts2, self.K, mask = mask
        )

        return R, t, pose_mask
    
    def make_transform(self, R, t):
        T = np.eye(4, dtype=np.float64)

        T[:3, :3] = R
        T[:3, 3] = t.ravel()

        return T

    def invert_transform(self, T):
        R = T[:3, :3]
        t = T[:3, 3]

        T_inv = np.eye(4, dtype=np.float64)

        T_inv[:3, :3] = R.T
        T_inv[:3, 3] = -R.T @ t

        return T_inv

    def triangulate_points(self, R, t, matches, pts1, pts2):

        # create P1 
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])

        # create P2
        P2 = self.K @ np.hstack([R, t])

        # Triangulate to homogeneous coordinates.
        points_4d = cv2.triangulatePoints(
            P1, P2, pts1, pts2
        )

        # Convert to 3D.
        points_3d = (points_4d[:3]/ points_4d[3]).T

        # Keep points in front of the camera.
        depth1 = points_3d[:, 2]

        # Transform points to camera 2.
        points_3d_cam2 = (R @ points_3d.T + t)

        depth2 = points_3d_cam2[2]

        valid = (
            np.isfinite(points_3d).all(axis=1)
            & (depth1 > 0)
            & (depth2 > 0)
        )

        return points_3d, valid

    def update_map_associations(self, matches, inliers, current_kps):

        new_map_points = {}

        for idx in inliers.ravel():

            match = matches[idx]

            prev_idx = match.queryIdx
            current_idx = match.trainIdx

            if prev_idx in self.map_points:
                new_map_points[current_idx] = (self.map_points[prev_idx])

        return new_map_points

    def projection_matrix(self, T_wc):
        T_cw = self.invert_transform(T_wc)

        R = T_cw[:3, :3]
        t = T_cw[:3, 3].reshape(3, 1)

        P = self.K @ np.hstack([
            R,
            t
        ])

        return P

    def triangulate_new_landmarks(
        self,
        prev_kp,
        curr_kp,
        matches,
        T_prev_wc,
        T_curr_wc
            ):
            if len(matches) == 0:
                return {}

            points1 = np.float32([
                prev_kp[m.queryIdx].pt
                for m in matches
            ]).T

            points2 = np.float32([
                curr_kp[m.trainIdx].pt
                for m in matches
            ]).T

            P1 = self.projection_matrix(
                T_prev_wc
            )

            P2 = self.projection_matrix(
                T_curr_wc
            )

            points_4d = cv2.triangulatePoints(
                P1,
                P2,
                points1,
                points2
            )

            points_3d = (
                points_4d[:3] /
                points_4d[3]
            ).T

            new_map = {}

            for match, point in zip(
                matches,
                points_3d
            ):
                if np.isfinite(point).all():

                    curr_idx = match.trainIdx

                    new_map[curr_idx] = point

            return new_map

    def get_new_matches(
        self,
        matches,
        current_map
    ):
        new_matches = []

        for match in matches:

            curr_idx = match.trainIdx

            if curr_idx not in current_map:
                new_matches.append(match)

        return new_matches

    def estimate_pose_pnp(self, points_3d, points_2d):

        points_3d = np.asarray(points_3d, dtype = np.float32)
        points_2d = np.asarray(points_2d, dtype = np.float32)

        # lets check the shape
        if points_3d.ndim != 2 or points_3d.shape[1] != 3:
            raise ValueError(
                f"points_3d must have shape (N, 3), got {points_3d.shape}"
            )

        if points_2d.ndim != 2 or points_2d.shape[1] != 2:
            raise ValueError(
                f"points_2d must have shape (N, 2), got {points_2d.shape}"
            )

        if len(points_3d) != len(points_2d):
            raise ValueError(
                "points_3d and points_2d must contain the same number of points."
            )

        #Min points required = 6 : Check

        if len(points_3d) < 6:
            raise ValueError(
                f"PnP requires at least 6 correspondences, "
                f"but only {len(points_3d)} were provided."
            )

       #Remove invalid points
        valid = (
            np.isfinite(points_3d).all(axis=1)
            & np.isfinite(points_2d).all(axis=1)
        )

        points_3d = points_3d[valid]
        points_2d = points_2d[valid]

        if len(points_3d) < 6:
            raise ValueError(
                "Not enough valid correspondences after removing NaN/Inf values."
            )


        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            self.K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=4.0,
            confidence=0.99,
            iterationsCount=100
        )

        if not success or inliers is None:
            raise RuntimeError("PnP failed to estimate a valid pose.")

        # Check no of inliers
        if len(inliers) < 6:
            raise RuntimeError(
                f"PnP produced only {len(inliers)} inliers."
            )
        
        # Rodrigues: rotation vector => rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        projected_points, _ = cv2.projectPoints(
                                                points_3d,
                                                rvec,
                                                tvec,
                                                self.K,
                                                None
                                            )
        projected_points = projected_points.reshape(-1, 2)

        # Reprojection Error
        errors = np.linalg.norm(
            projected_points - points_2d,
            axis=1
        )

        # Only evaluate PnP inliers
        inlier_indices = inliers.ravel()

        inlier_errors = errors[inlier_indices]

        mean_error = np.mean(inlier_errors)

        # Validate the error
        max_mean_error = 3.0

        if mean_error > max_mean_error:
            raise RuntimeError(
                f"PnP rejected: mean reprojection error "
                f"{mean_error:.2f}px > {max_mean_error}px."
            )

        return R, tvec, inliers

    def save_poses_kitti(self, poses, file_path):
        """
        Save 4x4 poses in KITTI trajectory format.

        Each line:
        r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz
        """

        with open(file_path, "w") as f:

            for T in poses:

                pose = T[:3, :4]

                values = pose.reshape(-1)

                f.write(
                    " ".join(f"{v:.12f}" for v in values)
                    + "\n"
                )

    def incremental_VO(self, prev_kps, prev_desc):

        for i in range(2, len(self.image_paths)):

            print(f"processing {i} frame")

            
            # 1. Previous pose
            
            T_prev_wc = self.estimated_poses[-1]

            
            # 2. Load current image
            
            image = cv2.imread(self.image_paths[i])

            current_kps, current_desc = self.detect_and_compute(image)

            
            # 3. Match previous frame -> current frame
            
            good_matches = self.match_features(
                prev_desc,
                current_desc
            )

            
            # 4. Separate mapped and new matches
            
            mapped_matches = []
            new_matches = []

            for match in good_matches:

                prev_idx = match.queryIdx

                if prev_idx in self.map_points:
                    mapped_matches.append(match)
                else:
                    new_matches.append(match)

            print(
                f"matches={len(good_matches)}, "
                f"mapped={len(mapped_matches)}, "
                f"new={len(new_matches)}, "
                f"map={len(self.map_points)}"
            )

            
            # 5. Build 3D -> 2D correspondences
            
            points3d = []
            points2d = []

            for match in mapped_matches:

                prev_idx = match.queryIdx
                current_idx = match.trainIdx

                points3d.append(
                    self.map_points[prev_idx]
                )

                points2d.append(
                    current_kps[current_idx].pt
                )

            points3d = np.asarray(
                points3d,
                dtype=np.float32
            )

            points2d = np.asarray(
                points2d,
                dtype=np.float32
            )

            
            # 6. Check minimum PnP points
            
            if len(points3d) < 6:

                print(
                    f"Not enough PnP points: "
                    f"{len(points3d)}"
                )

                # Move to current frame so that
                # matching does not remain stuck
                prev_kps = current_kps
                prev_desc = current_desc

                continue

            
            # 7. PnP
            
            try:

                R_curr, t_curr, pnp_inliers = (
                    self.estimate_pose_pnp(
                        points3d,
                        points2d
                    )
                )

            except (ValueError, RuntimeError) as e:

                print(f"PnP failed: {e}")

                prev_kps = current_kps
                prev_desc = current_desc

                continue

            
            # 8. Current pose
            
            T_curr_cw = self.make_transform(
                R_curr,
                t_curr
            )

            T_curr_wc = self.invert_transform(
                T_curr_cw
            )

            
            # 9. Store current pose
            
            self.estimated_poses.append(
                T_curr_wc
            )

            
            # 10. Update existing map associations
            # PnP inliers index into mapped_matches
            
            tracked_map = {}

            for idx in pnp_inliers.ravel():

                match = mapped_matches[idx]

                prev_idx = match.queryIdx
                current_idx = match.trainIdx

                tracked_map[current_idx] = (
                    self.map_points[prev_idx]
                )

            
            # 11. Triangulate new landmarks
            
            new_landmarks = (
                self.triangulate_new_landmarks(
                    prev_kps,
                    current_kps,
                    new_matches,
                    T_prev_wc,
                    T_curr_wc
                )
            )

            
            # 12. Add new landmarks to map
            
            tracked_map.update(
                new_landmarks
            )

            self.map_points = tracked_map

            
            # 13. Current frame becomes previous frame
            
            prev_kps = current_kps
            prev_desc = current_desc

    def init(self):
        print("Monocular Visual Odometry initialized.")

        # detect and compute features for the first image
        kps0, des0 = self.detect_and_compute(cv2.imread(self.image_paths[0])) #test first frame only
        kps1, des1 = self.detect_and_compute(cv2.imread(self.image_paths[1])) #test second frame only

        # match features between the two images
        good_matches = self.match_features(des0, des1)

        # get Points
        pts0 = np.float32([
                    kps0[m.queryIdx].pt for m in good_matches
                ])
        
        pts1 = np.float32([
            kps1[m.trainIdx].pt for m in good_matches
        ])

        # compute pose
        R, t, pose_mask = self.compute_pose(pts0, pts1, good_matches)

        # now we have to keep only valid matches in good matches
        inlier_matches = [m for m, valid in zip(good_matches, pose_mask.ravel()) if valid]


        # keep valid Points
        pts0 = np.float32([
                    kps0[m.queryIdx].pt for m in inlier_matches
                ])
        
        pts1 = np.float32([
            kps1[m.trainIdx].pt for m in inlier_matches
        ])

        # Triangulate
        points_3d, valid = self.triangulate_points(R, t, inlier_matches, pts0.T, pts1.T)

        # built initial map

        self.map_points = {}

        for match, point, is_valid in zip(
            inlier_matches,
            points_3d,
            valid
        ):
            if is_valid:
                self.map_points[match.trainIdx] = point

        # store poses

        self.estimated_poses = []

        # here we have frame 0 = world
        # now we will transform every frame into world
        # it mean that we have to invert each time the PnP resulting transformation
        T0_wc = np.eye(4)

        T1_cw = self.make_transform(R,t)
        T1_wc = self.invert_transform(T1_cw)

        self.estimated_poses.append(T0_wc)
        self.estimated_poses.append(T1_wc)


        # incremental VO loop
        prev_kps = kps1
        prev_desc = des1

        self.incremental_VO(prev_kps, prev_desc)

        self.save_poses_kitti(
            self.estimated_poses,
            "estimated_poses.txt"
        )

        self.save_poses_kitti(
            self.poses,
            "ground_truth.txt"
        )

        

    
if __name__ == "__main__":
    vo = MonocularVO()
    vo.init()
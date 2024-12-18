import numpy as np
import cv2

def perform_pnp(pred_tracks, pred_visibility, K, depths,refine_LM = False
                ):
    """
    Perform PnP for each pair of consecutive frames.
    
    Args:
        pred_tracks (np.ndarray): Shape (frames_num, pts_num_N, 2), 2D keypoints.
        pred_visibility (np.ndarray): Shape (frames_num, pts_num_N), visibility (bool).
        K (np.ndarray): Shape (3, 3), intrinsic camera matrix.
        depths (np.ndarray): Shape (frames_num, H, W), depth maps.
    
    Returns:
        poses (list): List of (R, t) for each pair of consecutive frames.
    """
    frames_num, pts_num_N, _ = pred_tracks.shape
    poses = []  # To store R, t for each frame pair
    poses_Rt = []  # To store R, t for each frame pair
    
    # Iterate over consecutive frames
    for frame_idx in range(frames_num - 1):
        # Current frame and next frame
        tracks_1 = pred_tracks[frame_idx]       # (pts_num_N, 2)
        vis_1 = pred_visibility[frame_idx]      # (pts_num_N,)
        depth_1 = depths[frame_idx]             # (H, W)

        tracks_2 = pred_tracks[frame_idx + 1]
        vis_2 = pred_visibility[frame_idx + 1]
        depth_2 = depths[frame_idx + 1]

        # Find keypoints visible in BOTH frames
        common_visibility = vis_1 & vis_2  # Element-wise AND
        if not np.any(common_visibility):  # Skip if no common visible keypoints
            assert 0
            poses_Rt.append((None, None))
            poses.append(np.eye(4))
            continue

        # Extract the visible 2D keypoints
        visible_kpts_1 = tracks_1[common_visibility]  # (n_pts, 2)
        visible_kpts_2 = tracks_2[common_visibility]  # (n_pts, 2)

        # Back-project to 3D points for the first frame
        points_3d_1 = []
        for i, (x, y) in enumerate(visible_kpts_1):
            x, y = int(x), int(y)
            z = depth_1[y, x]  # Get depth from depth map
            if z > 0:  # Valid depth check
                # Back-project using intrinsic matrix K
                X = (x - K[0, 2]) * z / K[0, 0]
                Y = (y - K[1, 2]) * z / K[1, 1]
                points_3d_1.append([X, Y, z])
        
        points_3d_1 = np.array(points_3d_1, dtype=np.float32)
        if len(points_3d_1) < 4:  # Minimum 4 points for PnP
            assert 0, 'less than 4 pts'
            poses_Rt.append((None, None))
            poses.append(np.eye(4))
            continue

        # Solve PnP using visible keypoints in the next frame
        visible_kpts_2 = visible_kpts_2[:len(points_3d_1)]  # Match the 3D points
        # retval, rvec, tvec = cv2.solvePnP(points_3d_1, visible_kpts_2, K, None)
        # use ransac
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d_1, visible_kpts_2, K, None,
            reprojectionError=8.0,  # Adjust error threshold as needed
            confidence=0.99,        # Confidence level for RANSAC
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if refine_LM:
            try:
                pts3d_i_filtered = points_3d_1[inliers]
                visible_kpts_2_filtered = visible_kpts_2[inliers]    
            except:
                assert 0, 'sure take all pts for LM??'
            rvec, tvec = cv2.solvePnPRefineLM(
                pts3d_i_filtered,
                visible_kpts_2_filtered,
                cameraMatrix=K,
                distCoeffs=None,
                rvec=rvec,
                tvec=tvec,
            )




        if retval:
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = tvec.flatten()
            poses_Rt.append((R, tvec))
            poses.append(pose)
        else:
            poses_Rt.append((None, None))
            poses.append(np.eye(4))
    
    return poses,poses_Rt

# Example usage
if __name__ == "__main__":
    # Mock example inputs
    frames_num = 5
    pts_num_N = 10
    H, W = 480, 640

    # Random 2D keypoints (x, y)
    pred_tracks = np.random.rand(frames_num, pts_num_N, 2) * [W, H]
    # Random visibility (boolean)
    pred_visibility = np.random.choice([True, False], size=(frames_num, pts_num_N))
    # Depth maps with values in range [0.1, 10]
    depths = np.random.uniform(0.1, 10.0, size=(frames_num, H, W))
    # Intrinsic matrix K
    K = np.array([[1000, 0, W / 2],
                  [0, 1000, H / 2],
                  [0, 0, 1]])
    # Perform PnP
    refine_LM = False
    poses = perform_pnp(pred_tracks, pred_visibility, K, depths, refine_LM=refine_LM)
    # Print results
    for i, (R, t) in enumerate(poses):
        print(f"Frame Pair {i}-{i+1}:")
        if R is not None and t is not None:
            print("Rotation Matrix:\n", R)
            print("Translation Vector:\n", t.ravel())
        else:
            print("No valid PnP solution.")

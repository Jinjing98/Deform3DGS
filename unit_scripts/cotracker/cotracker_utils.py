import torch
from cotracker_vis_utils import Visualizer
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

def load_data_from_video(video_path):
    #///////////////
    import cv2
    import numpy as np
    # video_path = "path/to/your/local/video/apple.mp4"
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            # Process the frame here
            print("Frame loaded")
    cap.release()
    frames = np.stack(frames,axis=0) # 11 512 640 3

    #//////////////////
    cotracker_video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float()  # B T C H W
    return cotracker_video

def dummy_queries_from_hard(mask, max_num_of_query_pts):
    #max 
    #t x y
    queries = torch.tensor([
        [0., 140., 150.],  # point tracked from the first frame
        [0., 240., 150.],  # point tracked from the first frame
        [0., 340., 150.],  # point tracked from the first frame
        [0., 440., 150.],  # point tracked from the first frame
        [0., 630., 150.],  # point tracked from the first frame
        # [5., 260., 500.], # frame number 5
        [9., 50., 400.], # ...
        [9., 250., 400.], # ...
        [9., 550., 400.], # ...
    ]).cuda()
    #t x y
    queries = torch.tensor([
        [0., 14., 15.],  # point tracked from the first frame
        [0., 24., 15.],  # point tracked from the first frame
        [0., 34., 15.],  # point tracked from the first frame
        [0., 44., 15.],  # point tracked from the first frame
        [0., 63., 15.],  # point tracked from the first frame
        # [5., 260., 500.], # frame number 5
        [9., 5., 40.], # ...
        [9., 25., 40.], # ...
        [9., 55., 40.], # ...
    ]).cuda()
    assert queries.dim() == 2,queries.shape
    return queries

def queries_from_mask(mask_paths, N, which_mask_img_idx = 0, inverse_mask = False):
    """
    Returns:
    - points (np.ndarray): Array of shape (K, 3) containing (which_mask_img_idx, x, y) for sampled points,
                           where K <= N. Returns an empty array if no valid points exist.
    - mask_binary (np.ndarray): Binary mask array.
    - sampled_x (list): List of x coordinates of sampled points.
    - sampled_y (list): List of y coordinates of sampled points.
    """

    # todo: support get quries from muti frames; potentially help if pts are tmp out of frames
    assert N>0
    mask_path = mask_paths[which_mask_img_idx]
    # Load the mask image and convert to binary numpy array
    mask = np.array(Image.open(mask_path).convert("L"))  # Convert to grayscale
    mask_binary = mask > 0  if not inverse_mask else mask == 0 # Threshold to create binary mask
    # Extract indices (y, x) where mask is True
    y_indices, x_indices = np.where(mask_binary)
    total_points = len(y_indices)

    if total_points == 0:
        print("The mask does not contain any valid points.")
        return np.empty((0, 3))#, mask_binary, [], []
    elif N > total_points:
        print(f"Requested {N} points, but only {total_points} valid points are available. Returning all.")
        sampled_x = x_indices
        sampled_y = y_indices
        points = np.array([[which_mask_img_idx, x, y] for x, y in zip(sampled_x, sampled_y)])
    else:
        # Randomly sample N indices
        sampled_indices = random.sample(range(total_points), N)
        sampled_y = y_indices[sampled_indices]
        sampled_x = x_indices[sampled_indices]
        points = np.array([[which_mask_img_idx, x, y] for x, y in zip(sampled_x, sampled_y)])
    
    points = torch.tensor(points).to(torch.float32)
    return points#, mask_binary, sampled_x, sampled_y


def sanity_check_queries(cotracker_video,queries):
    #////////////////
    # sanity make sure within frame and time_length
    max_t,max_x,max_y = queries.max(dim=0).values.squeeze(0).squeeze(0)
    min_t,min_x,min_y = queries.min(dim=0).values.squeeze(0).squeeze(0)
    print('sanity check.....')
    print('max in queries',queries.max(dim=0).values)
    print('min in queries',queries.min(dim=0).values)
    _, total_num_frames, num_c, frame_h, frame_w= cotracker_video.shape
    assert max_t <= total_num_frames-1, max_t
    assert min_t >= 0,min_t
    assert max_x<= frame_w and min_x >= 0
    assert max_y<= frame_h and min_y >= 0
def get_vis_obj(use_which,save_dir):
    assert use_which in ['grid','query','query_bi']
    if use_which == 'grid':
        vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3,)
    elif use_which in ['query','query_bi']:
        vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3,
                    mode='cool',
                    tracks_leave_trace=-1,
                        )
    else:
        assert 0, 'no support'
    return vis



if __name__== "__main__":


    # from utils.cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj
    # from cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj,dummy_queries_from_hard
    import os
    import torch
    #test mask

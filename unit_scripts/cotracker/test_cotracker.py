# from utils.cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj
from cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj,dummy_queries_from_hard
import os
import torch
import glob
import numpy as np
from vis_6Dpose_axis import draw_xyz_axis



if __name__== "__main__":
    device = 'cuda'
    # https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks
    # /demo.ipynb#scrollTo=fe23d210-ed90-49f1-8311-b7e354c7a9f6
    whichs = ['grid','query','query_bi']
    
    use_which = 'query' # looks query can already gave accepatable results
    # use_which = 'query_bi'
    data_piece = 'P2_7_455_465'
    grid_size = 30 #bigger denser

    # query_gen_from_mask
    query_N = 15
    inverse_mask = True
    query_which_mask_img_idx = 0
    query_which_mask_img_idx = 5
    BASE_DATASET_PATH="/mnt/cluster/datasets/StereoMIS_processed"
    mask_root=f"{BASE_DATASET_PATH}/{data_piece}/masks"
    mask_paths = glob.glob(mask_root+'/*')
    mask_paths = sorted(mask_paths)
    assert query_which_mask_img_idx <= (len(mask_paths)-1)



    backward_tracking=True if 'bi' in use_which else False
    #//////////////////
    video_path_root = '/mnt/ceph/tco/TCO-Staff/Homes/jinjing/'
    video_path = video_path_root+f'/exps/train/gs/SM/{data_piece}/deform3dgs_jj/12-17_09-53-45_use_skipMAPF_0_extent10_SHORT/video/ours_3000/gt_video.mp4'
    # video_path = os.path.dirname(video_path)+'/P2_4_idx528_553_id1057_1107_f26.mp4'
    co_vid_save_dir=f"{os.path.dirname(video_path)}/cotracker_videos"
    os.makedirs(co_vid_save_dir,exist_ok=True)
    raw_vid_name = os.path.basename(video_path).split('.')[0]
    co_vid_filename=f'{raw_vid_name}_grid_{grid_size}' if use_which == 'grid' else f'{raw_vid_name}_query_bi{backward_tracking}'
    if 'query' in use_which:
        co_vid_filename += f'_maskIDX{query_which_mask_img_idx}N{query_N}'

    #//////////////////
    # load data
    cotracker_video,frames_cv = load_data_from_video(video_path)
    cotracker_video = cotracker_video.to(device)
    #//////////////////
    # Run Offline CoTracker:
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    if use_which == 'grid':
        pred_tracks, pred_visibility = cotracker(cotracker_video, 
                                                grid_size=grid_size,
                                                ) # B T N 2,  B T N 1
    elif use_which in ['query','query_bi']:
        # queries = dummy_queries_from_hard(mask = None, max_num_of_query_pts = None)
        queries, xy_obj_center_2D = queries_from_mask(mask_paths, N = query_N, which_mask_img_idx = query_which_mask_img_idx,
                                    inverse_mask = inverse_mask,
                                    )
        queries = queries.to(cotracker_video.device)
        sanity_check_queries(cotracker_video,queries)
        pred_tracks, pred_visibility = cotracker(cotracker_video, 
                                                backward_tracking = backward_tracking,
                                                queries=queries[None])  # B T N 2,  B T N 1
    else:
        assert 0, NotImplementedError
    # print(pred_tracks)
    # print(pred_visibility)
    # print(pred_tracks.shape,pred_visibility.shape)
    # //////////////////////////////////////////////////
    #vis
    vis = get_vis_obj(use_which,co_vid_save_dir)
    vis.visualize(cotracker_video, 
                  tracks=pred_tracks, 
                  visibility=pred_visibility,
                  filename=co_vid_filename)

    # pred_tracks: B frames_num N 2(x,y)  float
    # pred_visibility: B frames_num N     bool
    # perfrom PnP based on 


    # [361.2000,  80.1333]
    #plot axis
    cx,cy = 283,241
    fx,fy = 484,484
    K = np.array([[fx,0, cx],[0, fy, cy],[0, 0, 1]])
    plot_axis_scale = 0.01 # the unit of axis length was 1m in 3D space--this scalre represent 1cm

  

    images_paths =[ path.replace('masks','images').replace('mask','color')  for path in mask_paths]
    color_imgs = frames_cv # 11 512 640 3

    pose_base = np.eye(4)
    # xy_center_2D + depth for the px + k. inverse?
    xyz_obj_center_3D = np.array([-0.03,0.01,-0.1])
    poses = []
    for i,path in enumerate(mask_paths):
        pose_i = pose_base.copy()
        pose_i[:3,-1] = np.array([-0.01*i,0.01,-0.1]) #xyz_obj_center_3D
        # poses = [pose_i for i,path in enumerate(mask_paths)]
        poses.append(pose_i)


    assert len(color_imgs)==len(mask_paths)
    assert len(poses)==len(mask_paths)
    assert len(images_paths)==len(mask_paths)
    
    # for path in images_paths:
    for rgb,rgb_img_path,obj_pose in \
        zip(color_imgs,images_paths,poses):

        assert os.path.exists(rgb_img_path),rgb_img_path
        # rgb = cv2.imread(rgb_img_path)
        vis = draw_xyz_axis(rgb[...,::-1], ob_in_cam=obj_pose, 
                            scale=plot_axis_scale, 
                            K=K, 
                            transparency=0, thickness=5)
        vis = vis[...,::-1]


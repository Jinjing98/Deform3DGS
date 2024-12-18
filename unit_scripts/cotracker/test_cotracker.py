# from utils.cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj
from cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj,dummy_queries_from_hard
import os
import torch
import glob
import numpy as np
from vis_6Dpose_axis import draw_xyz_axis
from PIL import Image
from vis_6Dpose_axis import vis_6dpose_axis
 




if __name__== "__main__":
    device = 'cuda'
    # https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks
    # /demo.ipynb#scrollTo=fe23d210-ed90-49f1-8311-b7e354c7a9f6
    whichs = ['grid','query','query_bi']
    
    use_which = 'query' # looks query can already gave accepatable results
    # use_which = 'query_bi'
    data_piece = 'P2_7_455_465'
    exp_name = '12-17_09-53-45_use_skipMAPF_0_extent10_SHORT'
    # data_piece = 'P2_7_1653_1678'
    # exp_name = '12-17_14-13-28_use_skipMAPF_0_extent10'
    # data_piece = 'P2_7_1279_1289'
    # exp_name = '12-17_14-15-51_use_skipMAPF_0_extent10'


    grid_size = 30 #bigger denser

    # query_gen_from_mask
    query_N = 1000
    inverse_mask = True
    query_which_mask_img_idx = 0
    query_which_mask_img_idx = 5
    BASE_DATASET_PATH="/mnt/cluster/datasets/StereoMIS_processed"
    mask_root=f"{BASE_DATASET_PATH}/{data_piece}/masks"
    mask_paths = glob.glob(mask_root+'/*')
    mask_paths = sorted(mask_paths)
    assert query_which_mask_img_idx <= (len(mask_paths)-1)

    #pnp param
    refine_LM = False

    backward_tracking=True if 'bi' in use_which else False
    #//////////////////
    video_path_root = '/mnt/ceph/tco/TCO-Staff/Homes/jinjing/'
    video_path = video_path_root+f'/exps/train/gs/SM/{data_piece}/deform3dgs_jj/{exp_name}/video/ours_3000/gt_video.mp4'
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
    # //////////////////////////////////////////////////
    #vis--vis cotrakcer opts results
    vis = get_vis_obj(use_which,co_vid_save_dir)
    vis.visualize(cotracker_video, 
                  tracks=pred_tracks, 
                  visibility=pred_visibility,
                  filename=co_vid_filename)

    # used for both pnp and axis_plot
    K = np.array([[560.0158,   0.  ,    320.     ],
                    [  0.   ,   560.01587, 256.     ],
                    [  0.   ,    0.  ,      1.     ]] )
    #/////////////////////////////////////
    # pred_tracks: B*frames_num*pts_num_N*2    2 refer to (x,y) for 2D keypoints  float
    # pred_visibility: B*frames_num*pts_num_N     bool
    # perfrom PnP for each pair of continous frames based on pred_tracks,pred_visibility, K
    from pnp_based_on_cotracker_opts import perform_pnp
    _, frames_num, pts_num_N,_ = pred_tracks.shape
    H, W = 512, 640 #480, 640
    depths = np.random.uniform(0.1, 10.0, size=(frames_num, H, W))
    depth_paths = [ mask_path.replace('masks','depth').replace('mask','depth') for mask_path in mask_paths]
    depths = [np.array(Image.open(depth_path)) for depth_path in depth_paths] #unit mm?
    depths = np.stack(depths,axis=0)
    # assert 0,f'{depths}{depths.shape} {pred_tracks} {pred_tracks.shape} {pred_visibility} {pred_visibility.dtype} {pred_visibility.shape}'

    # Perform PnP
    pred_tracks = pred_tracks.squeeze(0).detach().cpu().numpy()
    pred_visibility = pred_visibility.squeeze(0).detach().cpu().numpy()
    poses_mat,poses_Rt = perform_pnp(pred_tracks, pred_visibility, K, depths,refine_LM = refine_LM)
    # Print results
    for i, (R, t) in enumerate(poses_Rt):
        print(f"Frame Pair {i}-{i+1}:")
        if R is not None and t is not None:
            print("Rotation Matrix:\n", R)
            print("Translation Vector:\n", t.ravel())
        else:
            print("No valid PnP solution.")
    # convert poses tp mat
    #/////////////////////////////////////////////////////////////////
    #2D CENTER [361.2000,  80.1333]
    # plot axis--debug poses usage
    # pose_base_anchor
    init_axis_anchor_which_mask_img_idx = 0

    # video_path = video_path_root+f'/exps/train/gs/SM/{data_piece}/deform3dgs_jj/12-17_09-53-45_use_skipMAPF_0_extent10_SHORT/video/ours_3000/gt_video.mp4'
    axis_save_dir = video_path.split('video')[0]
    axis_save_dir = os.path.join(axis_save_dir,'axis_vis')
    os.makedirs(axis_save_dir,exist_ok=True)
    vis_6dpose_axis(mask_paths = mask_paths,
                            init_axis_anchor_which_mask_img_idx =init_axis_anchor_which_mask_img_idx,
                            color_imgs = frames_cv,
                            K = K,
                            inverse_mask = inverse_mask,
                            # CamprvtoCamcurrent_all = None,
                            CamprvtoCamcurrent_all = poses_mat,
                            # axis_save_dir = None,
                            axis_save_dir = axis_save_dir,
                            )
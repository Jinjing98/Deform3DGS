# from utils.cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj
from cotracker_utils import load_data_from_video,load_data_from_images,queries_from_mask,sanity_check_queries,get_vis_obj,dummy_queries_from_hard
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
    
    read_input_imgs_from_which = 'img'#'video'#'img'
    assert read_input_imgs_from_which in ['video','img']
    use_which = 'query' # looks query can already gave accepatable results
    # use_which = 'query_bi'
    backward_tracking=True if 'bi' in use_which else False

    data_piece = 'P2_7_455_465'
    # data_piece = 'P2_7_455_480'
    # data_piece = 'P2_7_1264_1289'
    # data_piece = 'P2_7_1279_1289'
    # data_piece = 'P2_7_1653_1678'
    data_piece = 'P2_6_11070_11095'
    data_piece = 'P2_7_170_195'
    if read_input_imgs_from_which == 'video':
        exp_name = '12-17_09-53-45_use_skipMAPF_0_extent10_SHORT'
        exp_name = '12-17_09-53-45_use_skipMAPF_0_extent10_0init_REF_SHORT'
        # exp_name = '12-17_14-03-39_use_skipMAPF_0_extent10'
        # exp_name = '12-17_14-06-08_use_skipMAPF_0_extent10'
        # # exp_name = '12-17_14-15-51_use_skipMAPF_0_extent10'
        # #<4 pts
        # exp_name = '12-17_14-13-28_use_skipMAPF_0_extent10'
        video_path_root = f'/mnt/ceph/tco/TCO-Staff/Homes/jinjing/exps/train/gs/SM/{data_piece}'
        video_path_details = f'/deform3dgs_jj/{exp_name}/video/ours_3000/gt_video.mp4'
        video_path = video_path_root+video_path_details
        assert os.path.exists(video_path),video_path
    else:
        assert read_input_imgs_from_which == 'img'

    grid_size = 30 #bigger denser
    # query_gen_from_mask
    query_N = 1000
    inverse_mask = True
    query_which_mask_img_idx = 5 #ideally should be 1st frame is not using queire_bis
    query_which_mask_img_idx = 0
    BASE_DATASET_PATH="/mnt/cluster/datasets/StereoMIS_processed"
    mask_root=f"{BASE_DATASET_PATH}/{data_piece}"
    mask_paths = glob.glob(mask_root+'/masks/*')
    mask_paths = sorted(mask_paths)
    assert query_which_mask_img_idx <= (len(mask_paths)-1)
    depth_paths = [ mask_path.replace('masks','depth').replace('mask','depth') for mask_path in mask_paths]
    images_paths =[ path.replace('masks','images').replace('mask','color')  for path in mask_paths]
    #pnp param
    #save the pose in root raw data
    refine_LM = False
    save_pnp_poses = True
    mask_name_for_queries_gen = mask_paths[query_which_mask_img_idx]
    mask_name_for_queries_gen = os.path.basename(mask_name_for_queries_gen).split('.')[0]
    pnp_poses_partname_co = f'CoTracer_{use_which}_queryGenMask{mask_name_for_queries_gen}_ptsN{query_N}' if 'query' in use_which \
        else f'CoTracer_{use_which}_{grid_size}'
    pnp_poses_partname_pnp = f'PnP_LMrf{int(refine_LM)}' 
    save_pnp_poses_path = f'{mask_root}/ObjPoses_rel_{pnp_poses_partname_co}_{pnp_poses_partname_pnp}.pt'
    #//////////////////
    co_vid_save_dir = f'{mask_root}/cotracker_videos'
    # used for both pnp and axis_plot
    K = np.array([[560.0158,   0.  ,    320.     ],
                    [  0.   ,   560.01587, 256.     ],
                    [  0.   ,    0.  ,      1.     ]] )

    # plot axis--debug poses usage
    # pose_base_anchor
    init_axis_anchor_which_mask_img_idx = 0 #used for compute the vis axis center points
    axis_save_dir = os.path.join(mask_root,'axis_vis')




    #//////**********************************************////////////
    # load data
    if read_input_imgs_from_which == 'video':
        cotracker_video,frames_cv = load_data_from_video(video_path)
    elif read_input_imgs_from_which == 'img':
        cotracker_video,frames_cv = load_data_from_images(image_paths=images_paths)
    else:
        assert 0, read_input_imgs_from_which
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
    os.makedirs(co_vid_save_dir,exist_ok=True)
    vis = get_vis_obj(use_which,co_vid_save_dir)
    vis.visualize(cotracker_video, 
                  tracks=pred_tracks, 
                  visibility=pred_visibility,
                  filename=pnp_poses_partname_co)
    #/////////////////////////////////////
    # pred_tracks: B*frames_num*pts_num_N*2    2 refer to (x,y) for 2D keypoints  float
    # pred_visibility: B*frames_num*pts_num_N     bool
    # perfrom PnP for each pair of continous frames based on pred_tracks,pred_visibility, K
    from pnp_based_on_cotracker_opts import perform_pnp
    _, frames_num, pts_num_N,_ = pred_tracks.shape
    depths = [np.array(Image.open(depth_path)) for depth_path in depth_paths] #unit mm?
    depths = np.stack(depths,axis=0)
    # assert 0,f'{depths}{depths.shape} {pred_tracks} {pred_tracks.shape} {pred_visibility} {pred_visibility.dtype} {pred_visibility.shape}'

    # Perform PnP
    pred_tracks = pred_tracks.squeeze(0).detach().cpu().numpy()
    pred_visibility = pred_visibility.squeeze(0).detach().cpu().numpy()
    poses_mat,poses_Rt = perform_pnp(pred_tracks, pred_visibility, K, depths,refine_LM = refine_LM)
    
    if save_pnp_poses:
        # assert 0, f' {poses_Rt} {poses_Rt[0]}'
        poses_Rt_torch_tensors = [torch.tensor([t[i] for t in poses_Rt]) for i in range(len(poses_Rt[0]))]
        
        def rel2traj(rel_poses_prv2crt, start_pose = np.eye(4)):
            rel_num = len(rel_poses_prv2crt)
            trajectory_cams2w = [start_pose]
            for rel_prv2crt in rel_poses_prv2crt:
                abs_crt2world = trajectory_cams2w[-1] @ np.linalg.inv(rel_prv2crt)
                trajectory_cams2w.append(abs_crt2world)
            return  trajectory_cams2w
        trajectory_cams2w = rel2traj(rel_poses_prv2crt = poses_mat, start_pose = np.eye(4))
        torch.save({
                    'trajectory_cams2w': torch.Tensor(np.stack(trajectory_cams2w)), #previous2current 
                    'poses_mat_p2c': torch.Tensor(np.stack(poses_mat)), #previous2current 
                    'poses_Rt_p2c': poses_Rt_torch_tensors,
                    }, save_pnp_poses_path)
    
    
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
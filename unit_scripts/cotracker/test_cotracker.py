# from utils.cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj
from cotracker_utils import load_data_from_video,queries_from_mask,sanity_check_queries,get_vis_obj,dummy_queries_from_hard
import os
import torch



if __name__== "__main__":
    device = 'cuda'
    # https://colab.research.google.com/github/facebookresearch/co-tracker/blob/master/notebooks
    # /demo.ipynb#scrollTo=fe23d210-ed90-49f1-8311-b7e354c7a9f6
    whichs = ['grid','query','query_bi']
    use_which = 'query' # looks query can already gave accepatable results
    data_piece = 'P2_7_455_465'
    # use_which = 'query_bi'
    grid_size = 30 #bigger denser

    query_N = 10
    query_which_mask_img_idx = 0
    query_which_mask_img_idx = 5
    inverse_mask = True
    BASE_DATASET_PATH="/mnt/cluster/datasets/StereoMIS_processed"
    mask_root=f"{BASE_DATASET_PATH}/{data_piece}/masks"
    import glob
    mask_paths = glob.glob(mask_root+'/*')
    mask_paths = sorted(mask_paths)




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
    cotracker_video = load_data_from_video(video_path)
    cotracker_video = cotracker_video.to(device)
    #//////////////////
    # Run Offline CoTracker:
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    if use_which == 'grid':
        pred_tracks, pred_visibility = cotracker(cotracker_video, 
                                                grid_size=grid_size,
                                                ) # B T N 2,  B T N 1
    elif use_which in ['query','query_bi']:
        queries = dummy_queries_from_hard(mask = None, max_num_of_query_pts = None)

        # assert 0, queries.dtype


        # assert 0, mask_paths

        queries = queries_from_mask(mask_paths, N = query_N, which_mask_img_idx = query_which_mask_img_idx,
                                    inverse_mask = inverse_mask,
                                    )
        # assert 0, queries
        queries = queries.to(cotracker_video.device)
        sanity_check_queries(cotracker_video,queries)
        pred_tracks, pred_visibility = cotracker(cotracker_video, 
                                                backward_tracking = backward_tracking,
                                                queries=queries[None])  # B T N 2,  B T N 1
    else:
        assert 0, NotImplementedError
    # //////////////////////////////////////////////////
    #vis
    vis = get_vis_obj(use_which,co_vid_save_dir)
    vis.visualize(cotracker_video, 
                  tracks=pred_tracks, 
                  visibility=pred_visibility,
                  filename=co_vid_filename)

 
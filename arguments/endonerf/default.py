ModelParams = dict(
    extra_mark = 'endonerf',
    # at the core, an extent (capturing-scene-scale) is needed 
    # for 1)set the spatical_lr_scalre(scene-scale-dependent) 2)(misgsonly) used for prune_big_pts
    # such value can be approximted by radius of cam_motions(assume cam round the scene to capture)
    # but at its core, it represent scene scale (gaussians)
    camera_extent = 10,
    # camera_extent_tool can be differ in extent, used for prune_big_pts_for_its_own?
    # camera_extent_tool = 2,
    #to-do:
    #    supprot bash edit
    #    support debug dst opt dur
    #    support run for muti splits
    #    support hard changes iters
    init_mode = 'skipMAPF',#'MAPF', #'skipMAPF' #rand
    tool_mask = 'use',#'use', #'use'(default) 'inverse' 'nouse'
    load_cotrackerPnpPose = True,#True #False



    #jj extend: misgs only
    renderOnce = True,
    compo_all_gs_ordered_renderonce = ['tissue','obj_tool1'],
    remain_redundant_default_param = True,
    trn_render_again_n_save = False, #True #False # use to monitor training process

    #jj extend: shared
    # eval_n_log_test_cam = False,
    eval_n_log_test_cam = True,
    use_ema_train = False,
    use_ema_test = False,

    dbg_print = False,    
    dbg_vis_adc = False,#True
    
    # tool_mask_loss_src = [],
    # tissue_mask_loss_src = ['depth','color'],

    tissue_mask_loss_src = ['depth','color'],
    tool_mask_loss_src = ['depth','color'],


    # tissue_mask_loss_src = ['color'],
    # tissue_mask_loss_src = ['depth'],
    # tool_mask_loss_src = ['color'],
)

#to do 
# steffi termin
# liwen results
# reply mails

OptimizationParams = dict(
    coarse_iterations = 0,
    deformation_lr_init = 0.00016,
    deformation_lr_final = 0.0000016,
    deformation_lr_delay_mult = 0.01,
    iterations = 1000,
    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,
    # prune_interval = 3000, #? wrong-not-used?

    #jj---ADD MORE HARD FOT TOOLL
    densification_interval = 100, #100
    densify_from_iter = 500,#500
    # densify_until_iter = -1,#700,#15_000, #-1 means no densify
    densify_until_iter = 700,#15_000,
    densify_grad_threshold_coarse = 0.0002,
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
    pruning_from_iter = 500,
    pruning_interval = 100,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005, 



    #jj used by misgs
    #jj posemodel needed
    # track_position_lr_delay_mult = 0.01
    track_position_lr_init =  0.05, #0.005
    track_position_max_steps = 3000, #30000
    track_rotation_lr_init = 0.001,
    track_rotation_max_steps = 3000, #30000
    track_warmup_steps = 0,
    # track_warmup_steps = 250,

    # used in misgs
    obj_pose_init = 'cotrackerpnp', #'cotrackerpnp',#'0', 
    obj_pose_rot_optim_space = 'rpy', #'lie'
    percent_dense = 0.01,
    # tool_prune_big_points = False,
    tool_prune_big_points = True,
    percent_big_ws = 0.1,
    # tool_prune_big_points = True, #used in new_Densify_and_prune_tool: there would be no points
    densify_grad_threshold_obj = 0.0002,#0.0004

)

ModelHiddenParams = dict(
    # set to 0 to disable FDM? render also need to set this to 0
    curve_num = 17,#17, # number of learnable basis functions. This number was set to 17 for all the experiments in paper (https://arxiv.org/abs/2405.17835)

    ch_num = 10, # channel number of deformable attributes: 10 = 3 (scale) + 3 (mean) + 4 (rotation)
    init_param = 0.01, )


PipelineParams = dict(
    # save_iterations = [0,3000], 
    # control render for train and offlne render
    # dbg_vis_render = True,
    dbg_vis_render = False,
    
    )


# class PipelineParams(ParamGroup):
#     def __init__(self, parser):
#         convert_SHs_python = False
#         compute_cov3D_python = False
#         debug = False
#         super().__init__(parser, "Pipeline Parameters")

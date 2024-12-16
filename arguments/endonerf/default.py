ModelParams = dict(
    extra_mark = 'endonerf',
    camera_extent = 10,
    #to-do:
    #    supprot bash edit
    #    support debug dst opt dur
    #    support run for muti splits
    #    support hard changes iters
    init_mode = 'skipMAPF',#'MAPF', #'skipMAPF' #rand
    tool_mask = 'use', #'use'(default) 'inverse' 'nouse'
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
    percent_dense = 0.01,
    opacity_reset_interval = 3000,
    position_lr_max_steps = 4000,
    # prune_interval = 3000, #? wrong-not-used?

    #jj
    densification_interval = 100,#00,
    densify_from_iter = 500,#00,
    densify_until_iter = 15_000,
    densify_grad_threshold_coarse = 0.0002,
    densify_grad_threshold_fine_init = 0.0002,
    densify_grad_threshold_after = 0.0002,
    pruning_from_iter = 500,#00,
    pruning_interval = 100,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005, 



    #jj used by misgs
    #jj posemodel needed
    # track_position_lr_delay_mult = 0.01
    track_position_lr_init =  0.05, #0.005
    # track_position_lr_final = 5.0e-5
    track_position_max_steps = 5000, #30000

    # track_rotation_lr_delay_mult = 0.01
    # track_rotation_lr_init = 0.001
    # track_rotation_lr_final = 1.0e-5
    track_rotation_max_steps = 5000, #30000
    track_warmup_steps = 0,
    # track_warmup_steps = 250,


    # percent_big_ws = 1000,
    tool_prune_big_points = False,
    # tool_prune_big_points = True, #used in new_Densify_and_prune_tool: there would be no points
    densify_grad_threshold_obj = 0.0002#0.0004

)

ModelHiddenParams = dict(
    # set to 0 to disable FDM? render also need to set this to 0
    curve_num = 17,#17, # number of learnable basis functions. This number was set to 17 for all the experiments in paper (https://arxiv.org/abs/2405.17835)

    ch_num = 10, # channel number of deformable attributes: 10 = 3 (scale) + 3 (mean) + 4 (rotation)
    init_param = 0.01, )


# PipelineParams = dict(
#     save_iterations = [0,3000], 
    
#     )


# class PipelineParams(ParamGroup):
#     def __init__(self, parser):
#         convert_SHs_python = False
#         compute_cov3D_python = False
#         debug = False
#         super().__init__(parser, "Pipeline Parameters")

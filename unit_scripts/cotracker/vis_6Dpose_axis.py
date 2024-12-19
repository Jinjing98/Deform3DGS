import cv2
import numpy as np
import os
from PIL import Image



def draw_xyz_axis(color, WorldtoCamcurrent, scale=0.1, K=np.eye(3), thickness=3, transparency=0.3,is_input_rgb=False):
  '''
#   WorldtoCamcurrent: pose
  @color: BGR
  '''
  def project_3d_to_2d(pt,K,pose ):
    pt = pt.reshape(4,1)
    pt_3d = (pose@pt)[:3,:]
    projected = K @ pt_3d
    projected = projected.reshape(-1)
    assert projected[2]!=0
    projected = projected/(projected[2])
    return projected.reshape(-1)[:2].round().astype(int)
  #todo we take the 1st frame as ref--for simplicity, add a delta_trans(3D space) for the WorldtoCamcurrent
  if is_input_rgb:
    color = cv2.cvtColor(color,cv2.COLOR_RGB2BGR)
  xx = np.array([1,0,0,1]).astype(float)
  yy = np.array([0,1,0,1]).astype(float)
  zz = np.array([0,0,1,1]).astype(float)
  xx[:3] = xx[:3]*scale
  yy[:3] = yy[:3]*scale
  zz[:3] = zz[:3]*scale

  origin = tuple(project_3d_to_2d(np.array([0,0,0,1]), K, WorldtoCamcurrent))
  
  xx = tuple(project_3d_to_2d(xx, K, WorldtoCamcurrent))
  yy = tuple(project_3d_to_2d(yy, K, WorldtoCamcurrent))
  zz = tuple(project_3d_to_2d(zz, K, WorldtoCamcurrent))
  print( f'the axis are draw with {xx} {yy} {zz} {origin}')
  line_type = cv2.FILLED
  arrow_len = 0
  tmp = color.copy()
  tmp1 = tmp.copy()
  #red
  tmp1 = cv2.arrowedLine(tmp1, origin, xx, color=(0,0,255), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  #grenn
  tmp1 = cv2.arrowedLine(tmp1, origin, yy, color=(0,255,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp1 = tmp.copy()
  #blue
  tmp1 = cv2.arrowedLine(tmp1, origin, zz, color=(255,0,0), thickness=thickness,line_type=line_type, tipLength=arrow_len)
  mask = np.linalg.norm(tmp1-tmp, axis=-1)>0
  tmp[mask] = tmp[mask]*transparency + tmp1[mask]*(1-transparency)
  tmp = tmp.astype(np.uint8)
  if is_input_rgb:
    tmp = cv2.cvtColor(tmp,cv2.COLOR_BGR2RGB)

  return tmp



def vis_6dpose_axis(mask_paths,
                    init_axis_anchor_which_mask_img_idx,
                    color_imgs,
                    K,
                    inverse_mask,
                    CamprvtoCamcurrent_all,
                    axis_save_dir = None
                    ):
    def get_pts_cam(depth, K, mask, disable_mask=False):
        focal = K[0,0]
        # W, H = img_wh
        # assert 0,depth.shape
        H,W = depth.shape
        assert depth.shape == mask.shape
        i, j = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H))
        cx = K[0,-1]
        cy = K[1,-1]
        X_Z = (i-cx) / focal#[0]
        Y_Z = (j-cy) / focal#[1]
        Z = depth
        X, Y = X_Z * Z, Y_Z * Z
        pts_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        if not disable_mask:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam[mask, :]
        else:
            mask = mask.reshape(-1).astype(bool)
            pts_valid = pts_cam
        return pts_valid,  mask    

    assert len(color_imgs)==len(mask_paths)
    assert len(CamprvtoCamcurrent_all)==len(mask_paths)-1,f'{len(CamprvtoCamcurrent_all)} {len(mask_paths)}'
    mask_path = mask_paths[init_axis_anchor_which_mask_img_idx]
    mask = Image.open(mask_path)
    mask = np.array(mask) / 255.0
    mask = mask if not inverse_mask else (1-mask)
    depth_path = mask_path.replace('masks','depth').replace('mask','depth')
    depth = np.array(Image.open(depth_path)) #unit mm
    plot_axis_scale = 10 # the unit of axis length was 1m in 3D space--this scalre represent 1cm

    debug_try_use_m_unit = True
    debug_try_use_m_unit = False
    if debug_try_use_m_unit:
      '''
      exactly as the mm unit
      '''
      # convert everything-related to m unit
      # xyz_obj_center_3D_anchor unit will be auto adjusted later when computed based on depth
      depth = depth.astype(np.float32)/1000.
      plot_axis_scale = 0.01 # 0.01m
      # do not forget the input-pose(its trans unit is mm)
      CamprvtoCamcurrent_all = np.stack(CamprvtoCamcurrent_all)
      CamprvtoCamcurrent_all[:,:3,3:4] = CamprvtoCamcurrent_all[:,:3,3:4]/1000


    pts_valid_anchor,  mask_anchor = get_pts_cam(depth, K, mask, disable_mask=False)
    xyz_obj_center_3D_anchor = pts_valid_anchor.mean(axis=0)# trans_unit mm 

    # startingframe_world transformation--we regard its rot to same as world
    pose_base_w2c0 = np.eye(4)
    pose_base_w2c0[:3,-1] = xyz_obj_center_3D_anchor #np.array(xyz_obj_center_3D_anchor)

    Cam0toCamprv = np.eye(4)
    for idx in range(len(mask_paths)):
        if idx ==0:
          CamprvtoCamcurrent = np.eye(4)#CamprvtoCamcurrent_all[idx-1]
        else: 
          CamprvtoCamcurrent = CamprvtoCamcurrent_all[idx-1]
        Cam0toCamcurrent = CamprvtoCamcurrent @ Cam0toCamprv
        WorldtoCamcurrent = Cam0toCamcurrent @ pose_base_w2c0
        Cam0toCamprv = Cam0toCamcurrent
        # poses.append(WorldtoCamcurrent)

        # rgb_path = images_paths[idx]
        # assert os.path.exists(rgb_path),rgb_img_path
        # rgb = cv2.imread(rgb_path)
        rgb = color_imgs[idx]
        vis = draw_xyz_axis(rgb[...,::-1], 
                            WorldtoCamcurrent=WorldtoCamcurrent, 
                            scale=plot_axis_scale, 
                            K=K, 
                            transparency=0, thickness=5)
        if axis_save_dir:
          vis_img_save_path = os.path.join(axis_save_dir,f'{idx}.png')
          success = cv2.imwrite(vis_img_save_path, vis)
          print(f'saved in {vis_img_save_path}')
        vis = vis[...,::-1]





if __name__=='__main__':
    # xyz: red /green /blue
    # positive dir: left /up /hit_eye

    # scale=100
    # be default, regard the length of the axis in 3D as 1 meter, then considering the scale
    scale=0.01
    WorldtoCamcurrent_dbg = np.eye(4)
    delta_trans_cm = np.zeros_like(WorldtoCamcurrent_dbg)
    #0,0,0 if the very center?
    # 0 0 0 mean the obj starts with the lense center location
    # in cm
    delta_trans_cm[:3,-1] = np.array([ 
                                # the abs value: bigger==far away from center in x; posistive is left
                                -3.101, 
                                # the abs value: bigger==far away from center in y; posistive is up
                                1.0,
                                # abs bigger(can't be too small,at least 10), the axis looks smaller (deeper)
                                # can only be negative(positive): its positive and negative will affect location
                                -10, 
                                  ])
    #unit should be meter
    # cm -> m
    delta_trans = delta_trans_cm*0.01
    WorldtoCamcurrent_dbg += delta_trans



    WorldtoCamcurrent = WorldtoCamcurrent_dbg
    
    # 484.6379395           0 283.3963318
    #           0  484.553772 241.3950958
    #           0           0           1

    cx = 283
    cy = 241
    fx = 484
    fy = 484
    K = np.array([
       [fx,0, cx],
       [0, fy, cy],
       [0, 0, 1],
    ])

    # rgb = cv2.resize(rgb, dsize=(self.W,self.H), interpolation=cv2.INTER_LINEAR)
    rgb_img_root = '/mnt/ceph/tco/TCO-All/Projects/MisGS/liwen_rp_output/BundleSDF_output/stereomis/color'
    rgb_img_name = '000246l.png'
    rgb_img_path = os.path.join(rgb_img_root,rgb_img_name)
    
    

    
    assert os.path.exists(rgb_img_path)
    rgb = cv2.imread(rgb_img_path)
    vis = draw_xyz_axis(rgb[...,::-1], WorldtoCamcurrent=WorldtoCamcurrent, scale=scale, K=K, transparency=0, thickness=5)
    vis = vis[...,::-1]



    # to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    # for color_file in color_files:
    #     color = imageio.imread(color_file)
    #     pose = np.loadtxt(color_file.replace('.png','.txt').replace('color','WorldtoCamcurrent'))
    #     pose = pose@np.linalg.inv(to_origin)


    # if self.use_gui:
    #     WorldtoCamcurrent = np.linalg.inv(frame._pose_in_model)
 

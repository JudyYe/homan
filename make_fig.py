from tqdm import tqdm
from torchvision.transforms import ToTensor
from hydra import main
import argparse
from PIL import Image
from glob import glob
from collections import defaultdict
import pickle
import torch
import torch.nn.functional as F
import os
import os.path as osp
import numpy as np
from jutils import mesh_utils, image_utils, hand_utils, geom_utils, slurm_utils, plot_utils
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Meshes
from homan.homan import HOMan
from homan_wrapper import HomanWrapper

device = 'cuda:0'
H = W = 512
skip_step = 4

def interpolate_model(param_list, T, idty_keys=[]):
    """

    Args:
        param_list {dict } of para list of length T
        T (_type_): _description_

    Returns:
        _type_: _description_
    """

    upsample_param_list = defaultdict(list)

    for t in range(T):
        for key in param_list:
            
            if t % skip_step == 0:
                upsample_param_list[key].append(param_list[key][t // skip_step])
                continue
            if key in idty_keys:
                upsample_param_list[key].append(param_list[key][t//skip_step])
                continue

            t0 = t // skip_step
            # TODO extrapolate
            t1 = t0 + 1
            alpha = (t % skip_step) / skip_step
            if t1 >= len(param_list[key]):
                t0 = len(param_list[key]) - 2
                t1 = len(param_list[key]) - 1
                # alpha = (t-t0*skip_step) / (t1-t0) * skip_step
                alpha = 1
            param_0 = param_list[key][t0]
            param_1 = param_list[key][t1]
            param = param_0 * (1-alpha) + param_1 * alpha
            
            upsample_param_list[key].append(param)
    return upsample_param_list


@main(config_path="configs", config_name="eval")
@slurm_utils.slurm_engine()
def run_all(cfg):
    model_list = glob(cfg.exp_dir + '*/model_seq.pkl')
    for model_dir in model_list:
        if cfg.video:
            run_video(osp.dirname(model_dir))
        if cfg.fig:
            run_fig(osp.dirname(model_dir), cfg.T_num)


@torch.no_grad()
def run_video(model_dir):
    save_dir = osp.join(model_dir, 'vis_video')
    os.makedirs(save_dir, exist_ok=True)

    jObj_list, jHand_list, K_ndc_list, jTc_list, inp_image = get_model_jTc_intrinsics(model_dir)

    name_list = ['input', 'render_0', 'render_1', 'jHoi', 'jObj', 'vHoi', 'vObj', 'vObj_t', 'vHoi_fix']
    image_list = [[] for _ in name_list]
    T = len(jObj_list)
    for i in tqdm(range(T)):
        gt = inp_image[i:i+1]
        image_list[0].append(gt)

        jHand, jTc, K_ndc = jHand_list[i], jTc_list[i], K_ndc_list[i]
        jObj = jObj_list[i]

        jObj.texture = mesh_utils.pad_texture(jObj)
        jHand.textures = mesh_utils.pad_texture(jHand, 'blue')

        hoi, _ = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTc, H=H, K_ndc=K_ndc, bin_size=32)
        image_list[1].append(hoi)

        # rotate by 90 degree in world frame 
        # 1. 
        jTcp = mesh_utils.get_wTcp_in_camera_view(np.pi/2, wTc=jTc)
        hoi, _ = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTcp, H=H, K_ndc=K_ndc, bin_size=32)
        image_list[2].append(hoi)

        if i == T//2:
            # coord = plot_utils.create_coord(device, size=1)
            jHoi = mesh_utils.join_scene([jHand, jObj])
            image_list[3] = mesh_utils.render_geom_rot(jHoi, scale_geom=True, out_size=H) 
            image_list[4] = mesh_utils.render_geom_rot(jObj, scale_geom=True, out_size=H) 
            
            # rotation around z axis
            vTj = torch.FloatTensor(
                [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
                [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]).to(device)[None].repeat(1, 1, 1)
            vHoi = mesh_utils.apply_transform(jHoi, vTj)
            vObj = mesh_utils.apply_transform(jObj, vTj)
            image_list[5] = mesh_utils.render_geom_rot(vHoi, scale_geom=True, out_size=H) 
            image_list[6] = mesh_utils.render_geom_rot(vObj, scale_geom=True, out_size=H) 
        
        jHoi = mesh_utils.join_scene([jHand, jObj])                
        vTj = torch.FloatTensor(
            [[np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
            [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]]).to(device)[None].repeat(1, 1, 1)
        vObj = mesh_utils.apply_transform(jObj, vTj)
        iObj_list = mesh_utils.render_geom_rot(vObj, scale_geom=True, out_size=H, bin_size=32) 
        image_list[7].append(iObj_list[i%len(iObj_list)])
        
        # HOI from fixed view point 
        scale = mesh_utils.Scale(0.5, ).cuda()
        trans = mesh_utils.Translate(0, 0.4, 0, ).cuda()
        fTj = scale.compose(trans)
        fHand = mesh_utils.apply_transform(jHand, fTj)
        fObj = mesh_utils.apply_transform(jObj, fTj)
        iHoi, iObj = mesh_utils.render_hoi_obj(fHand, fObj, 0, scale_geom=False, scale=1, bin_size=32)
        image_list[8].append(iHoi)
    # save 
    for n, im_list in zip(name_list, image_list):
        for t, im in enumerate(im_list):
            image_utils.save_images(im, osp.join(save_dir, n, f'{t:03d}'))
    return


def get_model_jTc_intrinsics(model_dir, ):
    index = '_'.join(osp.basename(model_dir).split('_')[:2])
    data_dir = f'/private/home/yufeiy2/scratch/result/HOI4D/{index}'
    image_list = sorted(glob(osp.join(data_dir, 'image/*.png')))
    image_list = [Image.open(image_path) for image_path in image_list]
    image_list = [np.array(image) for image in image_list]
    image_list = np.stack(image_list, axis=0)
    with open(os.path.join(model_dir, "model_seq.pkl"), "rb") as f:
        obj = pickle.load(f)
    model_seq = obj['model']
    images_np = image_list[:-1]
    hand_wrapper = hand_utils.ManopthWrapper().cuda()
    # images_np = obj['images']  # T, H, W, 3
    print(images_np.shape)
    hA, beta, hTo, oObj = HomanWrapper.extract_my_hoi(model_seq)
    jTh = hand_utils.get_nTh(hA=hA, hand_wrapper=hand_wrapper)

    save_dir = osp.join(model_dir, 'vis_clip')
    
    hand_wrapper = hand_utils.ManopthWrapper().cuda()
    oHand, _ = hand_wrapper(geom_utils.inverse_rt(mat=hTo, return_mat=True), hA)

    param_list = {
        'obj_faces': oObj.faces_padded(),
        'hand_faces': oHand.faces_padded(),
        'intrinsics': model_seq.camintr,
        'obj_verts': oObj.verts_padded(),
        'hTo': hTo,
        'jTh': jTh,
        'hand_verts': oHand.verts_padded(),
    }
    
    up_list = interpolate_model(param_list, len(images_np), idty_keys=['obj_faces', 'hand_faces', 'intrinsics'])
    oObj = Meshes(up_list['obj_verts'], up_list['obj_faces']).to(device)
    oHand = Meshes(up_list['hand_verts'], up_list['hand_faces']).to(device)
    intrinsics = torch.stack(up_list['intrinsics'])
    
    hTo = geom_utils.project_back_to_rot(torch.stack(up_list['hTo']))
    jTh = geom_utils.project_back_to_rot(torch.stack(up_list['jTh']))
    # o == c
    jTc_list = jTh @ hTo

    oHand.textures = mesh_utils.pad_texture(oHand, 'blue')
    oObj.textures = mesh_utils.pad_texture(oObj,)    
    oHoi = mesh_utils.join_scene([oHand, oObj])

    image_list = mesh_utils.render_geom_rot_v2(oHoi, time_len=1)
    image_utils.save_images(image_list[0], os.path.join(save_dir, 'hoi.png'))

    # intrinsics = model_seq.camintr
    # intrinsics = torch.FloatTensor(intrinsics).cuda()
    intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = 0
    intrinsics[..., 0, 0] *= 2
    intrinsics[..., 1, 1] *= 2
    images_np = torch.FloatTensor(images_np).to(device)
    inp_image = images_np.permute(0, 3, 1, 2).cuda() / 255
    inp_image = F.adaptive_avg_pool2d(inp_image, (H, H))

    # image_utils.save_images(image['image'], os.path.join(save_dir, 'overlay'), bg=inp_image, mask=image['mask'])
    # gif = [e for e in image['image'].split(1, dim=0)]
    # image_utils.save_gif(gif, os.path.join(save_dir, 'hoi'))
    
    # coordinate o == c
    jHand_list = mesh_utils.apply_transform(oHand, jTc_list)
    jObj_list = mesh_utils.apply_transform(oObj, jTc_list)

    jHand_list = [Meshes(jHand_list.verts_padded()[t:t+1], jHand_list.faces_padded()[t:t+1]).to(device) for t in range(len(images_np))]
    jObj_list = [Meshes(jObj_list.verts_padded()[t:t+1], jObj_list.faces_padded()[t:t+1]).to(device) for t in range(len(images_np))]
    K_ndc = [intrinsics[t:t+1] for t in range(len(images_np))]
    jTc_list = [jTc_list[t:t+1] for t in range(len(images_np))]
    # for t, (inp_image) in enumerate(images_np):
    #     # gt = torch.FloatTensor(inp_image).permute(2, 0, 1).unsqueeze(0).cuda()
    #     gt = inp_image[t:t+1]
    #     jHand = Meshes(jHand_list.verts_padded()[t:t+1], jHand_list.faces_padded()[t:t+1]).to(device)
    #     jObj = Meshes(jObj_list.verts_padded()[t:t+1], jObj_list.faces_padded()[t:t+1]).to(device)
    #     K_ndc = intrinsics[t:t+1]
    #     jTc = jTc_list[t:t+1]

    return jObj_list, jHand_list, K_ndc, jTc_list, inp_image


def run_fig(model_dir, T_num=10):
    index = '_'.join(osp.basename(model_dir).split('_')[:2])
    data_dir = f'/private/home/yufeiy2/scratch/result/HOI4D/{index}'
    image_list = sorted(glob(osp.join(data_dir, 'image/*.png')))
    image_list = [Image.open(image_path) for image_path in image_list]
    image_list = [np.array(image) for image in image_list]
    image_list = np.stack(image_list, axis=0)
    with open(os.path.join(model_dir, "model_seq.pkl"), "rb") as f:
        obj = pickle.load(f)
    model_seq = obj['model']
    images_np = image_list[:-1]
    hand_wrapper = hand_utils.ManopthWrapper().cuda()
    # images_np = obj['images']  # T, H, W, 3
    print(images_np.shape)
    hA, beta, hTo, oObj = HomanWrapper.extract_my_hoi(model_seq)
    jTh = hand_utils.get_nTh(hA=hA, hand_wrapper=hand_wrapper)

    save_dir = osp.join(model_dir, 'vis_clip')
    
    hand_wrapper = hand_utils.ManopthWrapper().cuda()
    oHand, _ = hand_wrapper(geom_utils.inverse_rt(mat=hTo, return_mat=True), hA)

    param_list = {
        'obj_faces': oObj.faces_padded(),
        'hand_faces': oHand.faces_padded(),
        'intrinsics': model_seq.camintr,
        'obj_verts': oObj.verts_padded(),
        'hTo': hTo,
        'jTh': jTh,
        'hand_verts': oHand.verts_padded(),
    }
    
    up_list = interpolate_model(param_list, len(images_np), idty_keys=['obj_faces', 'hand_faces', 'intrinsics'])
    oObj = Meshes(up_list['obj_verts'], up_list['obj_faces']).to(device)
    oHand = Meshes(up_list['hand_verts'], up_list['hand_faces']).to(device)
    intrinsics = torch.stack(up_list['intrinsics'])
    
    hTo = geom_utils.project_back_to_rot(torch.stack(up_list['hTo']))
    jTh = geom_utils.project_back_to_rot(torch.stack(up_list['jTh']))
    # o == c
    jTc_list = jTh @ hTo

    oHand.textures = mesh_utils.pad_texture(oHand, 'blue')
    oObj.textures = mesh_utils.pad_texture(oObj,)    
    oHoi = mesh_utils.join_scene([oHand, oObj])

    image_list = mesh_utils.render_geom_rot_v2(oHoi, time_len=1)
    image_utils.save_images(image_list[0], os.path.join(save_dir, 'hoi.png'))

    # intrinsics = model_seq.camintr
    # intrinsics = torch.FloatTensor(intrinsics).cuda()
    intrinsics[..., 0, 2] = intrinsics[..., 1, 2] = 0
    intrinsics[..., 0, 0] *= 2
    intrinsics[..., 1, 1] *= 2
    images_np = torch.FloatTensor(images_np).to(device)
    inp_image = images_np.permute(0, 3, 1, 2).cuda() / 255
    inp_image = F.adaptive_avg_pool2d(inp_image, (H, H))

    # image_utils.save_images(image['image'], os.path.join(save_dir, 'overlay'), bg=inp_image, mask=image['mask'])
    # gif = [e for e in image['image'].split(1, dim=0)]
    # image_utils.save_gif(gif, os.path.join(save_dir, 'hoi'))
    
    jHand_list = mesh_utils.apply_transform(oHand, jTc_list)
    jObj_list = mesh_utils.apply_transform(oObj, jTc_list)

    # reconstruct HOI and render in origin, 45, 60, 90 degree
    degree_list = [0, 45, 60, 90, 180, 360-60, 360-90]
    name_list = ['gt', 'overlay_hoi', 'overlay_obj']
    for d in degree_list:
        name_list += ['%d_hoi' % d, '%d_obj' % d]  


    image_list = [[] for _ in name_list]
    T = len(images_np)

    if T_num is None:
        T_list = [0, T//2, T-1]
    else:
        T_list = np.linspace(0, T-1, T_num).astype(np.int).tolist() 
    print('len', T, T_list)
    for t, (inp_image) in enumerate(images_np):
        if t not in T_list:
            continue

        # gt = torch.FloatTensor(inp_image).permute(2, 0, 1).unsqueeze(0).cuda()
        gt = inp_image[t:t+1]
        jHand = Meshes(jHand_list.verts_padded()[t:t+1], jHand_list.faces_padded()[t:t+1]).to(device)
        jObj = Meshes(jObj_list.verts_padded()[t:t+1], jObj_list.faces_padded()[t:t+1]).to(device)
        K_ndc = intrinsics[t:t+1]
        jTc = jTc_list[t:t+1]

        # jHand, jTc, jTh, intrinsics = trainer.get_jHand_camera(
        #     indices.to(device), model_input, ground_truth, H, W)
        hoi, obj = mesh_utils.render_hoi_obj_overlay(jHand, jObj, jTc, H=H, K_ndc=K_ndc)
        # image1, mask1 = render(renderer, jHand, jObj, jTc, intrinsics, H, W)
        image_list[0].append(gt)
        image_list[1].append(hoi)  # view 0
        image_list[2].append(obj)

        for i, az in enumerate(degree_list):
            img1, img2 = mesh_utils.render_hoi_obj(jHand, jObj, az, jTc=jTc, H=H, W=W)
            image_list[3 + 2*i].append(img1)  
            image_list[3 + 2*i+1].append(img2) 
        
        # save 
        name = 'homan'
        for n, im_list in zip(name_list, image_list):
            im = im_list[-1]
            image_utils.save_images(im, osp.join(save_dir, f'{t:03d}_{name}_{n}'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default='/private/home/yufeiy2/scratch/result/homan/default_1_1/*')
    # parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--T_num', type=int, default=10)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    run_all()
    # args = parse_args()
    # for model_path in glob.glob(args.model_dir):
    #     run_fig(model_path, T_num=args.T_num)


    # model_path = '/private/home/yufeiy2/scratch/result/homan/default_1_1/Kettle_1'
    # run_video(model_path)
    # run_fig(model_path)
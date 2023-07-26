import numpy as np
from homan.utils.camera import (
    compute_transformation_persp,
)
from homan.utils.geometry import rot6d_to_matrix
import torch
from pytorch3d.structures import Meshes
from jutils import mesh_utils, image_utils, image_utils, hand_utils,geom_utils, plot_utils
import os
import os.path as osp
import pickle
from filelock import FileLock
import wandb
from hydra import main
from homan_wrapper import HomanWrapper
from homan.homan import minimal_cat

def build_logger(cfg):
    os.makedirs(cfg.exp_dir + '/wandb', exist_ok=True)
    wandb.login(key='8e99ff14eba9677d715999d7a282c9ff79cfb9bf')
    # add lock of runid
    lockfile = FileLock(f"{cfg.exp_dir}/runid.lock")
    with lockfile:
        runid = None
        if os.path.exists(f"{cfg.exp_dir}/runid.txt"):
            runid = open(f"{cfg.exp_dir}/runid.txt").read().strip()
        log = wandb.init(
            entity=cfg.environment.user,
            project=cfg.project_name + osp.dirname(cfg.expname),
            name=osp.basename(cfg.expname),
            dir=cfg.exp_dir,
            id=runid,
            save_code=True,
            settings=wandb.Settings(start_method='fork'),
        )
        with  open(f"{cfg.exp_dir}/runid.txt", 'w') as fp:
            fp.write(log.id)
    return log


@main(config_path="configs", config_name="fit")
def eval_model(cfg):
    # save save mano hA, and hand?
    homan_wrapper = HomanWrapper(
        cfg.exp_dir, 
        cfg.image_dir,
        cfg.obj_file,
        start_idx=0,
        batch_size=10,
    )
    model_list, image_list = [], []
    print(len(homan_wrapper.dataset.image_names), len(homan_wrapper.dataset))
    for sample in homan_wrapper.dataset:
        with open(osp.join(sample['sample_folder'], 'model.pkl'), 'rb') as fp:
            model = pickle.load(fp)
        model_list.append(model['model_fine'] )
        image_list.append(np.array(model['images']))

        # if len(model_list) >= 2:
            # break
    model = minimal_cat(model_list)
    image = np.concatenate(image_list, axis=0)
    hand_wrapper = hand_utils.ManopthWrapper().cuda()

    homan_wrapper.sample_folder = homan_wrapper.parent_folder
    images_np = np.concatenate(image_list, 0)
    model_seq = minimal_cat(model_list)
    homan_wrapper.visualize(model_seq, images_np, 'seq')
    with open(os.path.join(homan_wrapper.parent_folder, "model_seq.pkl"), "wb") as f:
        pickle.dump({
            'model': model_seq,
            'images': images_np,
            }, f)
    hA, beta, hTo, oObj = homan_wrapper.extract_my_hoi(model_seq)
    with open(osp.join(homan_wrapper.parent_folder, 'param.pkl'), 'wb') as fp:
        pickle.dump({
            'hA': hA, 'beta': beta, 'hTo': hTo, 'oObj': oObj,
        }, fp)

    # mano_pca_pose = torch.ones([1, 45]).cuda()
    # mano_rot = torch.ones([1, 3]).cuda()
    # mano_trans = torch.ones([1, 3]).cuda()
    # rotations_hand = torch.rand([1,6]).cuda()
    # rotations_hand = rot6d_to_matrix(rotations_hand)
    # s = 1
    # rotations_hand = geom_utils.rot_cvt.axis_angle_to_matrix(torch.ones([1, 3]) * s).cuda()
    # rotations_hand = torch.eye(3).cuda()[None]
    # translations_hand = torch.zeros([1,3]).cuda()
    # side='right'
    # mano_res = model.mano_model.forward_pca(
    #     mano_pca_pose,
    #     rot=mano_rot,
    #     side=side)
    # verts = mano_res['verts'] + mano_trans.unsqueeze(1)
    # verts, _ = compute_transformation_persp(
    #                 meshes=verts,
    #                 translations=translations_hand,
    #                 rotations=rotations_hand,
    #         )
        
    # hand = Meshes(verts, hand_wrapper.hand_faces)
    # hand.textures = mesh_utils.pad_texture(hand, 'red')

    # hA = hand_wrapper.pca_to_pose(mano_pca_pose, True)
    # outside = geom_utils.rt_to_homo(rotations_hand.transpose(-1, -2), translations_hand)
    # hhand, _ = hand_wrapper(None, hA, mano_rot, mano_trans)
    # hhand = mesh_utils.apply_transform(hhand, outside)
    # hhand.textures = mesh_utils.pad_texture(hhand, 'blue')
    # scene = mesh_utils.join_scene([hhand, hand])
    # image_list = mesh_utils.render_geom_rot(scene, scale_geom=True)
    # image_utils.save_gif(image_list, osp.join(cfg.exp_dir, 'hand'))


    # verts_hand, _ = model.get_verts_hand()
    # verts_object, _ = model.get_verts_object()
    # faces_hand = model.faces_hand
    # faces_object = model.faces_object
    # print(verts_hand.shape, verts_object.shape, faces_hand.shape, faces_object.shape)
    # hand = Meshes(verts_hand, faces_hand.repeat(len(verts_hand), 1, 1))
    # hand.textures = mesh_utils.pad_texture(hand, 'blue')
    # obj = Meshes(verts_object, faces_object)
    # obj.textures = mesh_utils.pad_texture(obj, 'white')
    # hoi = mesh_utils.join_scene([hand, obj])
    # image_list = mesh_utils.render_geom_rot(hoi, scale_geom=True)
    # image_utils.save_gif(image_list, osp.join(cfg.exp_dir, 'hoi'))

    # hA, beta, hTo, oObj = homan_wrapper.extract_my_hoi(model)
    # print(hA.shape, model.int_scales_hand)
    # hHand, _ = hand_wrapper(None, hA, th_betas=beta)
    # hObj = mesh_utils.apply_transform(oObj, hTo,)
    # hHoi = mesh_utils.join_scene([hHand, hObj])
    # image_list = mesh_utils.render_geom_rot(hHoi, scale_geom=True)
    # image_utils.save_gif(image_list, osp.join(cfg.exp_dir, 'hoi_param'))
    # image_utils.save_images(image_list[0], osp.join(cfg.exp_dir, 'hoi_param'))

    # # oHoi = mesh_utils.join_scene([oHand, oObj])
    # # image_list = mesh_utils.render_geom_rot(oHoi, scale_geom=True)
    # # image_utils.save_gif(image_list, osp.join(cfg.exp_dir, 'hoi_oHoi'))
    # # image_utils.save_images(image_list[0], osp.join(cfg.exp_dir, 'hoi_oHoi'))

    # oTh = geom_utils.inverse_rt(mat=hTo, return_mat=True)
    # oHoi = mesh_utils.apply_transform(hHoi, oTh)
    # image_list = mesh_utils.render_geom_rot(oHoi, scale_geom=True)
    # image_utils.save_gif(image_list, osp.join(cfg.exp_dir, 'hoi_param_inv'))
    # image_utils.save_images(image_list[0], osp.join(cfg.exp_dir, 'hoi_param_inv'))
    return 


@main(config_path="configs", config_name="fit")
def opt_pose(cfg):
    homan_wrapper = HomanWrapper(
        cfg.exp_dir, 
        cfg.image_dir,
        cfg.obj_file,
        start_idx=0,
        batch_size=10,
    )
    homan_wrapper.run_video(hijack_gt=True, cfg=cfg)
    
    if cfg.logging == 'wandb':
        log = build_logger(cfg)
        for i, sample in enumerate(homan_wrapper.dataset):
            exp_dir = sample['sample_folder']
            wandb.log(
                {
                    'opt/step2': wandb.Video(osp.join(exp_dir, 'jointoptim_step2.mp4')),
                    'opt/step3': wandb.Video(osp.join(exp_dir, 'jointoptim_step3.mp4')),
                    'evidence/mask': wandb.Image(osp.join(exp_dir, 'super2d.png')),
                    'clip/fine': wandb.Video(osp.join(exp_dir, 'final_points_fine.mp4')),
                    'clip/coarse': wandb.Video(osp.join(exp_dir, 'final_points_coarse.mp4')),
                }
            )
    
        wandb.log(
            {'seq', wandb.Video(osp.join(cfg.exp_dir, 'final_points_seq.mp4'))}
        )


if __name__ == '__main__':
    opt_pose()
    # eval_model()
from glob import glob
import numpy as np
from homan.utils.camera import (
    compute_transformation_persp,
)
from homan.utils.geometry import rot6d_to_matrix
import torch
from pytorch3d.structures import Meshes
from jutils import mesh_utils, image_utils, image_utils, hand_utils,geom_utils, plot_utils, web_utils
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
    # homan_wrapper.run_video(hijack_gt=True, cfg=cfg)
    
    if cfg.logging == 'wandb':
        log = build_logger(cfg)    
        wandb.log(
            {'seq/seq': wandb.Video(osp.join(cfg.exp_dir, 'final_points_seq.mp4'))}
        )

        for i, sample in enumerate(homan_wrapper.dataset):
            exp_dir = sample['sample_folder']
            print(exp_dir)
            wandb.log(
                {
                    'opt/step2': wandb.Video(osp.join(exp_dir, 'jointoptim_step2.mp4')),
                    'opt/step3': wandb.Video(osp.join(exp_dir, 'jointoptim_step3.mp4')),
                    'evidence/mask': wandb.Image(osp.join(exp_dir, 'super2d.png')),
                    'clip/fine': wandb.Video(osp.join(exp_dir, 'final_points_fine.mp4')),
                    'clip/coarse': wandb.Video(osp.join(exp_dir, 'final_points_coarse.mp4')),
                }
            )
            wandb.finish()

@main(config_path="configs", config_name="fit")
def make_web(cfg):
    gif_list = sorted(glob(osp.join(cfg.exp_dir + '*_*/final_points_seq.mp4')))
    gif_list = [[e] for e in gif_list]
    web_utils.run(osp.join(cfg.exp_dir, 'vis'), gif_list, width=600)

if __name__ == '__main__':
    # opt_pose()
    make_web()
    # eval_model()
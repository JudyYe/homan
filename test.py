import os
import os.path as osp
from filelock import FileLock
import wandb
from hydra import main
from homan_wrapper import HomanWrapper

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
def opt_pose(cfg):
    # slurm_utils.update_pythonpath_relative_hydra()

    homan_wrapper = HomanWrapper(
        cfg.exp_dir, 
        cfg.image_dir,
        cfg.obj_file,
        start_idx=0,
        batch_size=10,
        # skip_step=1,
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
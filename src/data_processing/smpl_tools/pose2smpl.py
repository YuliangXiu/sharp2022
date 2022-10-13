from cmath import e
from tqdm import tqdm
from meters import Meters
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from train import train
from transform import transform
from save import save_params, save_obj, get_joint_mesh
from load import load
import torch
import numpy as np
from easydict import EasyDict as edict
from armatures import *
from models import *
import sys
from glob import glob
import os
import json
import trimesh
import multiprocessing as mp
from multiprocessing import Pool

sys.path.append(os.getcwd())

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def get_config(name):
    config_path = 'config/{}.json'.format(name)
    with open(config_path, 'r') as f:
        data = json.load(f)
    cfg = edict(data.copy())
    return cfg


def set_device(USE_GPU):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device

def pose2smplobj(path_full):
    name = "SHARP"
    cfg = get_config(name)
    device = set_device(USE_GPU=cfg.USE_GPU)

    smpl_layer = SMPL_Layer(center_idx=0,
                            gender=cfg.MODEL.GENDER,
                            model_root='src/data_processing/smpl_tools/smplpytorch/native/models')

    meters = Meters()
    
    # optimize the SMPL params
    target = torch.from_numpy(transform(name, load(name, path_full))).float()
    res = train(smpl_layer, target, device, cfg, meters, path_full)
    meters.update_avg(meters.min_loss, k=target.shape[0])
    meters.reset_early_stop()
    smpl_params = save_params(res, path_full)

    # save the exported SMPL model
    poses = np.load(os.path.normpath(path_full), allow_pickle=True)
    offset = poses[1] - (smpl_params["Jtr"][16]+smpl_params["Jtr"][17])*0.5
    outpath = path_full.replace("_pose.npy", "_smpl_model.obj")
    save_obj(smpl_params["verts"],
                smpl_layer.th_faces.long().detach().cpu().numpy(), outpath, offset)
    
    # mesh = trimesh.load(outpath)
    # lmks = get_joint_mesh(poses, 0.01)
    # (mesh+lmks).show()

    torch.cuda.empty_cache()
    


if __name__ == "__main__":
    
    torch.multiprocessing.set_start_method('spawn')
    
    name = "SHARP"
    cfg = get_config(name)

    paths = glob(cfg.DATASET.PATH + "/*_smpl/*/*.npy")

    p = Pool(mp.cpu_count())
    for _ in tqdm(p.imap_unordered(pose2smplobj, paths), total=len(paths)):
        pass
    p.close()
    p.join()

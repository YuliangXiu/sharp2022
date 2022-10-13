import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch.nn as nn
import sys
import os
from save import get_joint_mesh

from tqdm import tqdm

sys.path.append(os.getcwd())


def init(smpl_layer, target, device, cfg):
    params = {}
    params["pose_params"] = torch.zeros(target.shape[0], 72)
    params["shape_params"] = torch.zeros(target.shape[0], 10)
    params["trans_params"] = torch.zeros(target.shape[0], 3)
    params["scale"] = torch.ones([1])

    smpl_layer = smpl_layer.to(device)
    target = target.to(device)

    params["pose_params"] = params["pose_params"].to(device)
    params["trans_params"] = params["trans_params"].to(device)
    params["shape_params"] = params["shape_params"].to(device)
    params["scale"] = params["scale"].to(device)

    params["pose_params"].requires_grad = True
    params["trans_params"].requires_grad = True
    params["shape_params"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SHAPE)
    params["scale"].requires_grad = bool(cfg.TRAIN.OPTIMIZE_SCALE)

    optimizer = optim.Adam(
        [params["pose_params"], params["shape_params"], params["scale"], params["trans_params"]],
        lr=cfg.TRAIN.LEARNING_RATE)

    index = {}
    smpl_index = []
    dataset_index = []
    weight_index = []
    for tp in cfg.DATASET.DATA_MAP:
        if not torch.any(torch.isnan(target[:, tp[1], :])):
            smpl_index.append(tp[0][:2])
            dataset_index.append(tp[0][-1])
            weight_index.append(tp[1])

    index["smpl_index"] = torch.tensor(smpl_index).to(device)
    index["dataset_index"] = torch.tensor(dataset_index).to(device)
    index["weight_index"] = torch.tensor(weight_index).to(device)

    return smpl_layer, params, target, optimizer, index


def train(smpl_layer, target, device, cfg, meters, full_path):
    res = []
    smpl_layer, params, target, optimizer, index = \
        init(smpl_layer, target, device, cfg)
    pose_params = params["pose_params"]
    shape_params = params["shape_params"]
    scale = params["scale"]
    trans_params = params["trans_params"]

    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params, th_trans=trans_params)
        loss = F.mse_loss((Jtr.index_select(1, index["smpl_index"][:, 0]) *
                           index["weight_index"][:, 0].unsqueeze(0).unsqueeze(2) +
                           Jtr.index_select(1, index["smpl_index"][:, 1]) *
                           index["weight_index"][:, 1].unsqueeze(0).unsqueeze(2)) * scale,
                          target.index_select(1, index["dataset_index"]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if len(res) == 0:
            res = [pose_params, shape_params, verts * scale, Jtr * scale, scale]
        # pbar.set_description(f"Processing {os.path.basename(full_path)}, Loss {float(loss):.4f}")
        meters.update_early_stop(float(loss))
        if meters.update_res:
            res = [pose_params, shape_params, verts * scale, Jtr * scale, scale]
        if meters.early_stop:
            res = [pose_params, shape_params, verts * scale, Jtr * scale, scale]
            # J = get_joint_mesh(Jtr.detach().cpu().numpy()[0,:], 0.01)
            # J.visual.vertex_colors = np.array([255, 0, 0])
            # T = get_joint_mesh(target.detach().cpu().numpy()[0], 0.01)
            # T.visual.vertex_colors = np.array([0, 255, 0])
            # (J + T).show()
            # import ipdb
            # ipdb.set_trace()
            # print("Early stop at epoch {} !".format(epoch))
            break

    # print('Train ended, min_loss = {:.4f}'.format(float(meters.min_loss)))
    return res

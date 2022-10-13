# from display_utils import display_model
from label import get_label
import sys
import os
import re
from tqdm import tqdm
import numpy as np
import pickle
import trimesh

sys.path.append(os.getcwd())


def get_joint_mesh(joints, radius=2.0):

    ball = trimesh.creation.icosphere(radius=radius)
    combined = None
    for idx, joint in enumerate(joints):
        ball_new = trimesh.Trimesh(vertices=ball.vertices + joint, faces=ball.faces, process=False)
        if combined is None:
            combined = ball_new
        else:
            combined = trimesh.util.concatenate([ball_new, combined])
    return combined


def create_dir_not_exist(path):
    if not os.path.exists(path):
        os.mkdir(path)
        
def save_obj(verts, faces, path, offset):
    """
    Save the SMPL model into .obj file.
    Parameter:
    ---------
    path: Path to save.
    """
    with open(path, 'w') as fp:
      for v in verts:
        fp.write('v %f %f %f\n' % (v[0] + offset[0], v[1] + offset[1], v[2] + offset[2]))
      for f in faces + 1:
        fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))


def save_pic(res, smpl_layer, file, logger, dataset_name, target):
    _, _, verts, Jtr = res
    file_name = re.split('[/.]', file)[-2]
    fit_path = "fit/output/{}/picture/{}".format(dataset_name, file_name)
    create_dir_not_exist(fit_path)
    logger.info('Saving pictures at {}'.format(fit_path))
    for i in tqdm(range(Jtr.shape[0])):
        display_model(
            {'verts': verts.cpu().detach(),
             'joints': Jtr.cpu().detach()},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=os.path.join(fit_path+"/frame_{}".format(i)),
            batch_idx=i,
            show=False,
            only_joint=True)
    logger.info('Pictures saved')


def save_params(res, file):
    path = os.path.normpath(file)
    pose_params, shape_params, verts, Jtr, scale = res
    pose_params = (pose_params.cpu().detach()).numpy().tolist()
    shape_params = (shape_params.cpu().detach()).numpy().tolist()
    Jtr = (Jtr.cpu().detach()[0]).numpy()
    verts = (verts.cpu().detach()[0]).numpy().tolist()
    params = {}
    params["pose_params"] = pose_params
    params["shape_params"] = shape_params
    params["Jtr"] = Jtr
    params["verts"] = verts
    params["scale"] = scale
    export_path = path.replace("_pose.npy", "_smpl_params.pkl")
    with open(export_path, 'wb') as f:
        pickle.dump(params, f)
    return params

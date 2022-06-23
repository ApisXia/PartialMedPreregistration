import numpy as np
import torch
import torch.nn as nn
from visualization.show_data import *
import glob
import scipy
from eval_utils import farthest_point_sampling


def show_pred_gt(path, sampling_ops=False):
    # read in data
    data = np.load(path)
    pred = data['pred']
    gt = data['gt']
    gt = np.squeeze(gt)
    origin = data['origin']
    ra_m = data['ra_m']
    ct_m = data['ct_m']
    ct_align = data['ct_align']
    ra_trans = data['ra_trans']
    c = 1
    # define learned variable
    # model = TransMatrix()
    # optimizer = torch.optim.SGD((model.rot_m, model.tra_m), lr=0.0001)

    # + new way
    if sampling_ops:
        sp_index = farthest_point_sampling(origin, 5)
        pred_temp = pred[sp_index, :].copy()
        origin_temp = origin[sp_index, :].copy()
    else:
        pred_temp = pred.copy()
        origin_temp = origin.copy()
    mean_pred = np.mean(pred_temp, axis=0)
    mean_origin = np.mean(origin_temp, axis=0)
    p_pred = pred_temp - mean_pred
    q_origin = origin_temp - mean_origin
    A_pq = np.matmul(p_pred.transpose((1, 0)), q_origin)
    S_pq = scipy.linalg.sqrtm(np.matmul(A_pq.transpose((1, 0)), A_pq))
    R_pq = np.matmul(A_pq, np.linalg.inv(S_pq))
    the_matrix_rigid_primitive = np.diag(np.ones(4))
    the_matrix_rigid_primitive[0:3, 0:3] = R_pq
    the_matrix_rigid_primitive[0:3, 3] = mean_pred
    # back to our mode
    offset_matrix = np.diag(np.ones(4))
    offset_matrix[0:3, 3] = -mean_origin
    the_matrix_rigid = np.matmul(offset_matrix, the_matrix_rigid_primitive)

    # + old way
    add_one = np.ones((origin.shape[0], 1))
    origin = np.concatenate((origin, add_one), axis=1)
    pred = np.concatenate((pred, add_one), axis=1)
    origin_p = origin.transpose((1, 0))
    pred_p = pred.transpose((1, 0))
    gt = np.concatenate((gt, add_one), axis=1)
    gt_p = gt.transpose((1, 0))

    sample_index = np.random.choice(len(pred), 100, replace=False)

    # the_matrix = np.matmul(pred_p[:, sample_index], np.matmul(origin[sample_index, :], np.linalg.inv(np.matmul(origin_p[:, sample_index], origin[sample_index, :]))))

    the_matrix_ori = np.matmul(gt_p, np.matmul(origin, np.linalg.inv(np.matmul(origin_p, origin))))

    # + describe difference
    diff_offset = np.diag(np.ones(4))
    diff_offset[0:3, 3] = (np.asarray(np.shape(ra_m)) - 1) / 2
    # diff_the_matrix = np.matmul(the_matrix_rigid, diff_offset)
    # diff_the_matrix_ori = np.matmul(the_matrix_ori, diff_offset)
    diff_the_matrix = the_matrix_rigid
    diff_the_matrix_ori = the_matrix_ori
    translation_diff = np.sqrt(np.sum((diff_the_matrix[0:3, 3] - diff_the_matrix_ori[0:3, 3]) ** 2))
    r_ab = np.matmul(np.transpose(diff_the_matrix[0:3, 0:3]), diff_the_matrix_ori[0:3, 0:3])
    rotation_diff = np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2))
    print('translation_diff:', translation_diff)
    print('rotation_diff:   ', rotation_diff)
    # return translation_diff, rotation_diff

    # show surface
    ct_verts, ct_faces, ct_normals, ct_values = get_iso_surface(ct_m, 1000, 2000)
    ra_re_verts, ra_re_faces, ra_re_normals, ra_re_values = get_iso_surface(ra_m, 1000, 2000)

    ct_mesh = to_show_surface(ct_verts, ct_faces, ct_normals)
    ra_re_mesh = to_show_surface(ra_re_verts, ra_re_faces, ra_re_normals)

    ct_mesh_t = copy.deepcopy(ct_mesh).transform(ct_align)
    # ra_re_image_matrix[2, 3] += 60
    ra_re_mesh_t = copy.deepcopy(ra_re_mesh).transform(the_matrix_rigid)
    ra_re_mesh_t_ori = copy.deepcopy(ra_re_mesh).transform(the_matrix_ori)

    ct_mesh_t.paint_uniform_color([1, 0.706, 0])
    ra_re_mesh_t.paint_uniform_color([0, 0.706, 0])
    ra_re_mesh_t_ori.paint_uniform_color([0.5, 0, 0.706])

    # o3d.visualization.draw_geometries([ra_re_mesh_t])
    # o3d.visualization.draw_geometries([ct_mesh_t, ra_re_mesh_t_ori, ra_re_mesh_t])
    # o3d.visualization.draw_geometries([ct_mesh_t, ra_re_mesh_t_ori])
    o3d.visualization.draw_geometries([ra_re_mesh_t_ori, ra_re_mesh_t])

    # origin = torch.from_numpy(origin).float()
    # pred = torch.from_numpy(pred).float()

    # for i in range(100):
    #     optimizer.zero_grad()
    #     new_pred = model(origin)
    #     loss = torch.norm((pred - new_pred[:, 0:3]), p=1)
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())

    # c = 1


class TransMatrix(nn.Module):
    def __init__(self):
        super(TransMatrix, self).__init__()
        self.register_buffer('rot_m', torch.nn.Parameter(torch.randn((3, 3), requires_grad=True)))
        self.register_buffer('tra_m', torch.nn.Parameter(torch.randn((3), requires_grad=True)))
        self.full_m = torch.diag(torch.ones(4))
        self.full_m[0:3, 0:3] = self.rot_m
        self.full_m[0:3, 3] = self.tra_m

    def forward(self, x):
        return torch.matmul(self.full_m, x.transpose(0, 1)).transpose(0, 1)


if __name__ == "__main__":
    data_path = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/preprocess/experiments_group/simplest_model/eval_result/p16_fold1_new'
    data_list = list(glob.glob(data_path + '/*' + '[0-9]' + '.npz'))
    # good
    # show_pred_gt(data_list[0])
    # show_pred_gt(data_list[5])
    # scale
    # show_pred_gt(data_list[6])
    # show_pred_gt(data_list[13])
    # bad
    # show_pred_gt(data_list[1])
    # show_pred_gt(data_list[3])
    trans_diffs_holder = []
    rot_diffs_holder = []
    for i, p in enumerate(data_list):
        print(i)
        show_pred_gt(p, sampling_ops=False)
        # diffs = show_pred_gt(p, sampling_ops=True)
        # trans_diffs_holder.append(diffs[0])
        # rot_diffs_holder.append(diffs[1])
    # diffs_dict = {'translation_diff': np.asarray(trans_diffs_holder),
    #               'rotation_diff': np.asarray(rot_diffs_holder)}
    # np.savez(data_path + '/diffs_statistics_5s.npz', **diffs_dict)

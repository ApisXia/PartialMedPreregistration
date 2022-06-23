import numpy as np
from scipy.linalg import sqrtm, logm


def gen_rigid_matrix(origin, pred):
    # get rigid transformation matrix
    mean_pred = np.mean(pred, axis=0)
    mean_origin = np.mean(origin, axis=0)
    p_pred = pred - mean_pred
    q_origin = origin - mean_origin
    A_pq = np.matmul(p_pred.transpose((1, 0)), q_origin)
    S_pq = sqrtm(np.matmul(A_pq.transpose((1, 0)), A_pq))
    R_pq = np.matmul(A_pq, np.linalg.inv(S_pq))
    the_matrix_rigid_primitive = np.diag(np.ones(4))
    the_matrix_rigid_primitive[0:3, 0:3] = R_pq
    the_matrix_rigid_primitive[0:3, 3] = mean_pred
    # back to our mode
    offset_matrix = np.diag(np.ones(4))
    offset_matrix[0:3, 3] = -mean_origin
    the_matrix_rigid = np.matmul(offset_matrix, the_matrix_rigid_primitive)

    return the_matrix_rigid


def matrix_diff_analysis(gt, ori, gt_direct, ori_direct):
    translation_diff = np.sqrt(np.sum((ori[0:3, 3] - gt[0:3, 3]) ** 2))
    r_ab = np.matmul(np.transpose(gt_direct), ori_direct)
    rotation_diff = np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2))
    return translation_diff, rotation_diff


def matrix_diff_analysis_2(gt, ori):
    '''
    https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
    :param gt:
    :param ori:
    :return:
    '''
    translation_diff = np.sqrt(np.sum((ori[0:3, 3] - gt[0:3, 3]) ** 2))
    r_a = ori[0:3, 0:3]
    r_b = gt[0:3, 0:3]
    F_norm_a = np.linalg.norm(r_a, ord='fro')
    F_norm_b = np.linalg.norm(r_b, ord='fro')
    cos_ab = np.sum(r_a * r_b) / F_norm_a / F_norm_b
    rotation_diff = np.rad2deg(np.arccos(cos_ab))
    return translation_diff, rotation_diff


def matrix_diff_analysis_3(gt, ori):
    '''
    :param gt:
    :param ori:
    :return:
    '''
    translation_diff = np.sqrt(np.sum((ori[0:3, 3] - gt[0:3, 3]) ** 2))
    r_a = ori[0:3, 0:3]
    r_b = gt[0:3, 0:3]
    rotation_diff = np.linalg.norm(logm(np.matmul(np.transpose(r_a), r_b)), ord='fro') / np.sqrt(2)
    return translation_diff, rotation_diff


def lankmark_diff_analysis(gt, ori, shape, num=5):
    x, y, z = shape
    x_l = np.linspace(0, x-1, num)
    y_l = np.linspace(0, y-1, num)
    z_l = np.linspace(0, z-1, num)
    x_s, y_s, z_s = np.meshgrid(x_l, y_l, z_l)
    x_s = x_s.reshape([1, -1])
    y_s = y_s.reshape([1, -1])
    z_s = z_s.reshape([1, -1])
    addon_ones = np.ones_like(x_s)
    input_pts = np.concatenate((x_s, y_s, z_s, addon_ones), axis=0)
    gt_pts = np.matmul(gt, input_pts)[0:3, :]
    ori_pts = np.matmul(ori, input_pts)[0:3, :]
    error = np.mean(np.sqrt(np.sum((gt_pts - ori_pts) ** 2, axis=0)))
    return error

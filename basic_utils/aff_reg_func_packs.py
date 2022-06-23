import sys
import os

import matplotlib.pyplot as plt
import torch as th
import numpy as np
import glob
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import basic_utils.import_util_libs.airlab as al
from basic_utils.show_data import *


def get_paired_reg_matrix(f_image, m_image, iter_num=300, init_3dra_matrix=None, init_ct_matrix=None,
                          device=th.device("cuda:0"), ori_ops=False, vis_ops=False, metric='mse'):
    f_image = th.from_numpy(f_image)
    m_image = th.from_numpy(m_image)

    f_image = f_image.to(device=device).float()
    m_image = m_image.to(device=device).float()

    if metric in ['mse', 'lcc', 'ngf', 'ncc', 'ssim']:
        f_image[f_image < -500.] = -500.  # + set some offset to clean the image
        f_image[f_image > 3000.] = 3000.
        f_image += 500.
    elif metric == 'nmi':
        f_image[f_image < -500.] = -500.  # + set some offset to clean the image
        f_image[f_image > 3000.] = 3000.
    else:
        raise ValueError
    f_image = al.Image(f_image, list(f_image.shape), [1, 1, 1], [0, 0, 0])  # + create airlab data, with image, size, spacing, origin

    if metric in ['mse', 'lcc', 'ngf', 'ncc', 'ssim']:
        m_image[m_image < -500.] = -500.
        m_image[m_image > 3000.] = 3000.
        m_image += 500.
    elif metric == 'nmi':
        m_image[m_image < -500.] = -500.
        m_image[m_image > 3000.] = 3000.
    else:
        raise ValueError
    m_image = al.Image(m_image, list(m_image.shape), [1, 1, 1], [0, 0, 0])  # + same as 'fixed_image'

    # + create pairwise registration object
    registration = al.PairwiseRegistration()

    # + choose the affine transformation model
    transformation = al.transformation.pairwise.RigidTransformation(m_image, opt_cm=False, init_3dra_matrix=init_3dra_matrix, init_ct_matrix=init_ct_matrix)
    # ^ opt_cm means whether to update center of mass
    transformation.init_translation(f_image)
    # ^ use center of mass of fixed_image to init the translation of moving_image

    registration.set_transformation(transformation)  # ^ set transformation in registration mission

    # + choose the Mean Squared Error as image loss
    il_1 = al.loss.pairwise.MSE(f_image, m_image)
    il_2 = al.loss.pairwise.MI(f_image, m_image)
    il_3 = al.loss.pairwise.LCC(f_image, m_image)
    il_4 = al.loss.pairwise.NGF(f_image, m_image)
    il_5 = al.loss.pairwise.NCC(f_image, m_image)
    il_6 = al.loss.pairwise.SSIM(f_image, m_image, dim=3)

    if metric == 'mse':
        registration.set_image_loss([il_1])  # ^ set loss
    elif metric == 'nmi':
        registration.set_image_loss([il_2])
    elif metric == 'lcc':
        registration.set_image_loss([il_3])
    elif metric == 'ngf':
        registration.set_image_loss([il_4])
    elif metric == 'ncc':
        registration.set_image_loss([il_5])
    elif metric == 'ssim':
        registration.set_image_loss([il_6])
    else:
        raise ValueError

    # + choose the Adam optimizer to minimize the objective
    optimizer = th.optim.Adam(transformation.parameters(), lr=0.01)

    registration.set_optimizer(optimizer)  # + set optimizer
    registration.set_number_of_iterations(iter_num)  # + set num of iteration

    # + start the registration
    registration.start()

    # + plot the results in slice form
    # f_image.image = 1 - f_image.image
    # m_image.image = 1 - m_image.image
    # displacement = transformation.get_displacement()  # + get the matrix
    # warped_image = al.transformation.utils.warp_image(m_image, displacement)
    # ^ get sampled position and done (simple)
    # plt_a_slice(f_image, m_image, warped_image, [10, 45, 60])

    # + get transformation matrix
    transformation_matrix_from_this_method = transformation._compute_transformation_matrix().cpu().detach().numpy()
    trans_on_moving_images = get_matrix_we_use(transformation_matrix_from_this_method.copy(), m_image.size)

    # + vis resampled result of fixed and moving
    if vis_ops:
        fixed_verts, fixed_faces, fixed_normals, fixed_values = get_iso_surface(
            np.squeeze(f_image.image.cpu().detach().numpy()), 1000, 2000)
        moving_verts, moving_faces, moving_normals, moving_values = get_iso_surface(
            np.squeeze(m_image.image.cpu().detach().numpy()), 1000, 2000)
        # moving_verts, moving_faces, moving_normals, moving_values = get_iso_surface(moving_image_temp['images_ct'][30:100, 30:100, 30:100], 1000, 2000)

        fixed_mesh = to_show_surface(fixed_verts, fixed_faces, fixed_normals)
        moving_mesh = to_show_surface(moving_verts, moving_faces, moving_normals)

        fixed_mesh_t = copy.deepcopy(fixed_mesh)
        moving_mesh_t = copy.deepcopy(moving_mesh).transform(trans_on_moving_images)
        # print(trans_on_moving_images)

        fixed_mesh_t.paint_uniform_color([1, 0.706, 0])
        moving_mesh_t.paint_uniform_color([0, 0.706, 0])

        o3d.visualization.draw_geometries([fixed_mesh_t, moving_mesh_t])

    if ori_ops:
        new_matrix = np.diag(np.ones(4))
        new_matrix[0:3, :] = transformation_matrix_from_this_method
        # new_matrix = np.linalg.inv(new_matrix)
        return new_matrix

    return trans_on_moving_images

    # + save trans matrix
    # image_name = moving_path.split('/')[-2]
    # save_matrix_name = 'paired_trans_matrix/{}.npz'.format(image_name)
    # np.savez(save_matrix_name, pt_matrix=trans_on_moving_images)


def get_matrix_we_use(the_matrix, image_size):
    # + permute some axis (because the expansion of python matrix is z->y->x, so I need to permute axis)
    permute_target = [2, 1, 0]
    the_matrix[[0, 1, 2], :] = the_matrix[permute_target, :]
    the_matrix[:, [0, 1, 2]] = the_matrix[:, permute_target]
    # ^ 1. set2origin matrix
    set2origin_matrix = np.eye(4)
    set2origin_matrix[:3, 3] = - np.asarray(image_size) / 2
    # ^ 2. scale21 matrix
    scale21_matrix = np.eye(4)
    scale21_matrix[0, 0] = 2 / 128
    scale21_matrix[1, 1] = 2 / 128
    scale21_matrix[2, 2] = 2 / 128
    # ^ 3. apply the derived rotation matrix
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :] = the_matrix[:3, :]
    rotation_matrix = np.linalg.inv(rotation_matrix)
    # ^ 4 scale21 matrix reverse
    scale21_matrix_rev = np.eye(4)
    scale21_matrix_rev[0, 0] = 128 / 2
    scale21_matrix_rev[1, 1] = 128 / 2
    scale21_matrix_rev[2, 2] = 128 / 2
    # ^ 5 set2origin matrix reverse
    set2origin_matrix_rev = np.eye(4)
    set2origin_matrix_rev[:3, 3] = [128 / 2] * 3
    # ^ combine 2 final matrix
    colossal_matrix = np.matmul(set2origin_matrix_rev, np.matmul(scale21_matrix_rev,
                                np.matmul(rotation_matrix, np.matmul(scale21_matrix, set2origin_matrix))))
    return colossal_matrix


def map_back_matrix(colossal_matrix, image_size, reverse_ops=True):
    # ^ 2. set2origin matrix
    set2origin_matrix = np.eye(4)
    set2origin_matrix[:3, 3] = - np.asarray(image_size) / 2
    # ^ 1. scale21 matrix
    scale21_matrix = np.eye(4)
    scale21_matrix[0, 0] = 2 / 128
    scale21_matrix[1, 1] = 2 / 128
    scale21_matrix[2, 2] = 2 / 128
    # ^ 4 scale21 matrix reverse
    scale21_matrix_rev = np.eye(4)
    scale21_matrix_rev[0, 0] = 128 / 2
    scale21_matrix_rev[1, 1] = 128 / 2
    scale21_matrix_rev[2, 2] = 128 / 2
    # ^ 3 set2origin matrix reverse
    set2origin_matrix_rev = np.eye(4)
    set2origin_matrix_rev[:3, 3] = [128 / 2] * 3
    # ^ combine 2 final matrix
    the_matrix = np.matmul(np.linalg.inv(scale21_matrix_rev), np.matmul(np.linalg.inv(set2origin_matrix_rev),
                           np.matmul(colossal_matrix, np.matmul(np.linalg.inv(set2origin_matrix), np.linalg.inv(scale21_matrix)))))
    if reverse_ops:
        the_matrix = np.linalg.inv(the_matrix)
    # + permute some axis (because the expansion of python matrix is z->y->x, so I need to permute axis)
    permute_target = [2, 1, 0]
    the_matrix[[0, 1, 2], :] = the_matrix[permute_target, :]
    the_matrix[:, [0, 1, 2]] = the_matrix[:, permute_target]
    return the_matrix


def plt_a_slice(f_image, m_image, w_image, nums):
    for num in nums:
        plt.subplot(131)
        plt.imshow(f_image.numpy()[:, :, num], cmap='gray')
        plt.title('Fixed Image Slice')
        plt.subplot(132)
        plt.imshow(m_image.numpy()[:, :, num], cmap='gray')
        plt.title('Moving Image Slice')
        plt.subplot(133)
        plt.imshow(w_image.numpy()[:, :, num], cmap='gray')
        plt.title('Warped Moving Image Slice')
        plt.show()


if __name__ == '__main__':
    data_path = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/GpR_mhd_center_aligned_128_16_new'
    file_list = list(glob.glob(data_path + '/*/'))
    fixed_path = file_list[26]
    print('Fixed path is {}'.format(fixed_path))
    moving_path = file_list[58]
    print('Moving path is {}'.format(moving_path))

    fixed_image_temp = np.load(fixed_path + 'ct_3dra_scaled.npz')
    fixed_image = fixed_image_temp['images_ct']
    moving_image_temp = np.load(moving_path+'ct_3dra_scaled.npz')
    # moving_image = th.from_numpy(moving_image_temp['images_ct']).to(device=device)
    moving_image = fixed_image_temp['images_3dra']
    # moving_image = th.from_numpy(moving_image_temp['images_ct'][30:100, 30:100, 30:100]).to(device=device)

    get_paired_reg_matrix(fixed_image, moving_image, iter_num=200, init_3dra_matrix=np.diag(np.ones(4)), init_ct_matrix=np.diag(np.ones(4)))

    # # + fail 65

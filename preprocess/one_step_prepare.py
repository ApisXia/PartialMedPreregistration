import os
import numpy as np
import SimpleITK
import glob
from basic_utils.show_data import load_dcm
from scipy.ndimage import interpolation
from basic_utils.aff_reg_func_packs import get_paired_reg_matrix


def gen_mhd(path_dicom, save_path, case_class):
    os.makedirs(save_path, exist_ok=True)
    image_pixel, image_matrix = load_dcm(path_dicom)

    image_pixel = np.transpose(image_pixel, (2, 0, 1))
    image_matrix[[0, 1], :] = image_matrix[[1, 0], :]
    image_matrix[:, [0, 1]] = image_matrix[:, [1, 0]]

    sitk_img = SimpleITK.GetImageFromArray(image_pixel, isVector=False)
    SimpleITK.WriteImage(sitk_img, os.path.join(save_path, case_class + ".mhd"))
    np.save(os.path.join(save_path, case_class + "_transform_matrix.npy"), image_matrix)


def resample(image, old_spacing, new_spacing=[1, 1, 1]):
    resize_factor = old_spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = old_spacing / real_resize_factor

    image = interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def resample_limit_size(image, old_spacing, lim_size=128):
    real_shape = image.shape * old_spacing
    max_edge = np.max(real_shape)
    new_shape = np.round(real_shape / max_edge * lim_size)
    real_resize_factor = new_shape / image.shape
    # new_spacing = old_spacing / real_resize_factor

    image = interpolation.zoom(image, real_resize_factor, mode='nearest')

    im_x, im_y, im_z = image.shape
    fill_x = min(im_x, lim_size)
    fill_y = min(im_y, lim_size)
    filled_image = np.ones([lim_size] * 3) * -2048.
    if im_z <= lim_size:
        filled_image[:fill_x, :fill_y, (lim_size-im_z):] = image[:fill_x, :fill_y, :]
    else:
        filled_image[:fill_x, :fill_y, :] = image[:fill_x, :fill_y, (im_z-lim_size):]
    shift_z = lim_size - im_z

    return filled_image, lim_size/max_edge, shift_z


def resample_based_scale(image, old_spacing, lim_size, base_scale):
    resize_factor = old_spacing * base_scale
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    image = interpolation.zoom(image, real_resize_factor, mode='nearest')

    im_x, im_y, im_z = image.shape
    fill_x = min(im_x, lim_size)
    fill_y = min(im_y, lim_size)
    filled_image = np.ones([lim_size] * 3) * -2048.
    if im_z <= lim_size:
        filled_image[:fill_x, :fill_y, (lim_size-im_z):] = image[:fill_x, :fill_y, :]
    else:
        filled_image[:fill_x, :fill_y, :] = image[:fill_x, :fill_y, (im_z-lim_size):]
    shift_z = lim_size - im_z

    return filled_image, shift_z


def gen_bounding_box_with_minimum_size(shape, minimum_size=40):
    s_x, s_y, s_z = shape
    if s_x < minimum_size or s_y < minimum_size or s_z < minimum_size:
        raise ValueError
    x_start_p = np.random.randint(0, s_x-minimum_size+1)
    y_start_p = np.random.randint(0, s_y-minimum_size+1)
    z_start_p = np.random.randint(0, s_z-minimum_size+1)
    x_end_p = np.random.randint(x_start_p+minimum_size, s_x+1)
    y_end_p = np.random.randint(y_start_p+minimum_size, s_y+1)
    z_end_p = np.random.randint(z_start_p+minimum_size, s_z+1)
    return (x_start_p, y_start_p, z_start_p), (x_end_p, y_end_p, z_end_p)


def gen_fixed_bounding_box(shape, box_size=16):
    s_x, s_y, s_z = shape
    if s_x < box_size or s_y < box_size or s_z < box_size:
        raise ValueError
    x_start_p = np.random.randint(0, s_x-box_size+1)
    y_start_p = np.random.randint(0, s_y-box_size+1)
    z_start_p = np.random.randint(0, s_z-box_size+1)
    x_end_p = x_start_p + box_size
    y_end_p = y_start_p + box_size
    z_end_p = z_start_p + box_size
    return (x_start_p, y_start_p, z_start_p), (x_end_p, y_end_p, z_end_p)


def gen_save_mhd_based_scale(f_dir, lim_size, base_info):
    base_scale, base_image = base_info
    # f_dir = f_dir[1]
    case_name = f_dir.split('/')[-1]
    # print('visualize {}'.format(case_name))

    ct_folder = f_dir + '/CT'
    ra_re_folder = f_dir + '/3DRA/Registrated/3D RA'
    data_save_dir = save_dir.format(lim_size) + '/' + case_name
    os.makedirs(data_save_dir, exist_ok=True)

    # ^ read ct images and resampled to limited size
    ct_image_pixel, ct_image_matrix, _, _ = load_dcm(ct_folder)
    ct_old_spacing = ct_image_matrix.diagonal()[0:3]
    ct_old_translation = ct_image_matrix[0:3, 3]
    filled_ct, shift_z = resample_based_scale(ct_image_pixel, ct_old_spacing, lim_size=lim_size, base_scale=base_scale)

    # ^ load info of 3dra images
    ra_re_image_pixel, ra_re_image_matrix, ra_re_directions, ra_re_spacings = load_dcm(ra_re_folder)

    # ^ matrix of resampled registered 3dra
    ra_re_new_spacings = ra_re_spacings * base_scale
    ra_re_new_image_pixel, _ = resample(ra_re_image_pixel, ra_re_new_spacings)
    # ^ new matrix of transformation matrix for 3dra
    ra_re_new_image_matrix = np.eye(4)
    ra_re_new_image_matrix[:3, 0] = ra_re_directions[0]
    ra_re_new_image_matrix[:3, 1] = ra_re_directions[1]
    ra_re_new_image_matrix[:3, 2] = ra_re_directions[2]
    ra_re_new_image_matrix[:3, 3] = (ra_re_image_matrix[0:3, 3] - ct_old_translation) * base_scale
    ct_relative_shift_matrix = np.eye(4)
    ct_relative_shift_matrix[2, 3] = shift_z
    ra_re_new_image_matrix = np.matmul(ct_relative_shift_matrix, ra_re_new_image_matrix)
    # + load aligned matrix
    aligned_matrix = get_paired_reg_matrix(base_image, filled_ct, metric='mse', iter_num=700)

    # ^ save for scaled CT and 3DRA
    np.savez_compressed(data_save_dir + '/ct_3dra_scaled.npz',
                        images_ct=filled_ct,
                        scale=base_scale,
                        images_3dra=ra_re_new_image_pixel,
                        matrix_3dra=np.matmul(aligned_matrix, ra_re_new_image_matrix),
                        matrix_ct_aligned=aligned_matrix)

    # ^ randomly cut patches
    it = 0
    duplicate_holder = []
    # duplicate_counter = 0
    while it < 150:
        while True:
            rand_starts, rand_ends = gen_fixed_bounding_box(ra_re_new_image_pixel.shape, box_size=16)
            if (rand_starts, rand_ends) not in duplicate_holder:
                duplicate_holder.append((rand_starts, rand_ends))
                break
            print('duplicate!!!!!!!!!!!!!!')
            # duplicate_counter += 1
            # if duplicate_counter > 200:
            #     break
        # if duplicate_counter > 200:
        #     break
        ra_re_rand_image_pixel = ra_re_new_image_pixel[rand_starts[0]:rand_ends[0],
                                 rand_starts[1]:rand_ends[1],
                                 rand_starts[2]:rand_ends[2]]
        center_vector_format = np.ones([4, 1])
        center_vector_format[0:3, 0] = (np.asarray(rand_starts) + np.asarray(rand_ends) - 1) / 2
        # + this center also aligned
        ra_re_rand_image_center = np.matmul(aligned_matrix, np.matmul(ra_re_new_image_matrix, center_vector_format))
        ra_re_rand_image_center = ra_re_rand_image_center[0:3]
        if (ra_re_rand_image_center < 0).any() and (ra_re_rand_image_center > (lim_size - 1)).any():
            print(ra_re_rand_image_center)
            continue
        np.savez_compressed(data_save_dir + '/sampled_3dra_%02d.npz' % it, images_3dra=ra_re_rand_image_pixel,
                            center_3dra=ra_re_rand_image_center,
                            center_3dra_origin=(np.asarray(rand_starts) + np.asarray(rand_ends) - 1) / 2)
        it += 1

    print('Finished {}'.format(case_name))

    # # ^ vis resampled result of ct and 3dra (all checked to be correct)
    # ct_verts, ct_faces, ct_normals, ct_values = get_iso_surface(filled_ct, 1000, 2000)
    # ra_re_verts, ra_re_faces, ra_re_normals, ra_re_values = get_iso_surface(ra_re_new_image_pixel, 1000, 2000)
    #
    # ct_mesh = to_show_surface(ct_verts, ct_faces, ct_normals)
    # ra_re_mesh = to_show_surface(ra_re_verts, ra_re_faces, ra_re_normals)
    #
    # ct_mesh_t = copy.deepcopy(ct_mesh) # transform(ct_image_matrix)
    # # ra_re_image_matrix[2, 3] += 60
    # ra_re_mesh_t = copy.deepcopy(ra_re_mesh).transform(ra_re_new_image_matrix)
    #
    # ct_mesh_t.paint_uniform_color([1, 0.706, 0])
    # ra_re_mesh_t.paint_uniform_color([0, 0.706, 0])
    #
    # # o3d.visualization.draw_geometries([ra_re_mesh_t])
    # o3d.visualization.draw_geometries([ct_mesh_t, ra_re_mesh_t])


def gen_save_mhd_uniform_first(f_dir, lim_size):
    case_name = f_dir.split('/')[-1]
    print('{} is the preliminary case. Start preprocessing...'.format(case_name))

    ct_folder = f_dir + '/CT'
    ra_re_folder = f_dir + '/3DRA/Registrated/3D RA'
    data_save_dir = save_dir.format(lim_size) + '/' + case_name
    os.makedirs(data_save_dir, exist_ok=True)

    # ^ read ct images, resampled to limited size, get shift info on z axis
    ct_image_pixel, ct_image_matrix, _, _ = load_dcm(ct_folder)
    ct_old_spacing = ct_image_matrix.diagonal()[0:3]
    ct_old_translation = ct_image_matrix[0:3, 3]
    filled_ct, scale, shift_z = resample_limit_size(ct_image_pixel, ct_old_spacing, lim_size=lim_size)

    # ^ load info of 3dra images
    ra_re_image_pixel, ra_re_image_matrix, ra_re_directions, ra_re_spacings = load_dcm(ra_re_folder)

    # ^ matrix of resampled registered 3dra
    ra_re_new_spacings = ra_re_spacings * scale
    ra_re_new_image_pixel, _ = resample(ra_re_image_pixel, ra_re_new_spacings)
    # ^ new matrix of transformation matrix for 3dra
    ra_re_new_image_matrix = np.eye(4)
    ra_re_new_image_matrix[:3, 0] = ra_re_directions[0]
    ra_re_new_image_matrix[:3, 1] = ra_re_directions[1]
    ra_re_new_image_matrix[:3, 2] = ra_re_directions[2]
    ra_re_new_image_matrix[:3, 3] = (ra_re_image_matrix[0:3, 3] - ct_old_translation) * scale
    ct_relative_shift_matrix = np.eye(4)
    ct_relative_shift_matrix[2, 3] = shift_z
    ra_re_new_image_matrix = np.matmul(ct_relative_shift_matrix, ra_re_new_image_matrix)
    # ^ save for scaled CT and 3DRA
    np.savez_compressed(data_save_dir + '/ct_3dra_scaled.npz',
                        images_ct=filled_ct,
                        scale=scale,
                        images_3dra=ra_re_new_image_pixel,
                        matrix_3dra=ra_re_new_image_matrix,
                        matrix_ct_aligned=np.diag(np.ones(4)))

    # ^ randomly cut patches
    it = 0
    duplicate_holder = []
    # duplicate_counter = 0
    while it < 150:
        while True:
            rand_starts, rand_ends = gen_fixed_bounding_box(ra_re_new_image_pixel.shape, box_size=16)
            if (rand_starts, rand_ends) not in duplicate_holder:
                duplicate_holder.append((rand_starts, rand_ends))
                break
            print('duplicate!!!!!!!!!!!!!!')
            # duplicate_counter += 1
            # if duplicate_counter > 200:
            #     break
        # if duplicate_counter > 200:
        #     break
        ra_re_rand_image_pixel = ra_re_new_image_pixel[rand_starts[0]:rand_ends[0],
                                                       rand_starts[1]:rand_ends[1],
                                                       rand_starts[2]:rand_ends[2]]
        center_vector_format = np.ones([4, 1])
        center_vector_format[0:3, 0] = (np.asarray(rand_starts) + np.asarray(rand_ends) - 1) / 2
        ra_re_rand_image_center = np.matmul(ra_re_new_image_matrix, center_vector_format)
        ra_re_rand_image_center = ra_re_rand_image_center[0:3]
        if (ra_re_rand_image_center < 0).any() and (ra_re_rand_image_center > (lim_size-1)).any():
            print(ra_re_rand_image_center)
            continue
        np.savez_compressed(data_save_dir + '/sampled_3dra_%02d.npz' % it, images_3dra=ra_re_rand_image_pixel,
                            center_3dra=ra_re_rand_image_center,
                            center_3dra_origin=(np.asarray(rand_starts) + np.asarray(rand_ends) - 1) / 2)
        it += 1

    print('Finished {}'.format(case_name))
    return scale, filled_ct

    # ^ vis resampled result of ct and 3dra (all checked to be correct)
    # ct_verts, ct_faces, ct_normals, ct_values = get_iso_surface(filled_ct, 1000, 2000)
    # ra_re_verts, ra_re_faces, ra_re_normals, ra_re_values = get_iso_surface(ra_re_new_image_pixel, 1000, 2000)
    #
    # ct_mesh = to_show_surface(ct_verts, ct_faces, ct_normals)
    # ra_re_mesh = to_show_surface(ra_re_verts, ra_re_faces, ra_re_normals)
    #
    # ct_mesh_t = copy.deepcopy(ct_mesh) # transform(ct_image_matrix)
    # # ra_re_image_matrix[2, 3] += 60
    # ra_re_mesh_t = copy.deepcopy(ra_re_mesh).transform(ra_re_new_image_matrix)
    #
    # ct_mesh_t.paint_uniform_color([1, 0.706, 0])
    # ra_re_mesh_t.paint_uniform_color([0, 0.706, 0])
    #
    # # o3d.visualization.draw_geometries([ra_re_mesh_t])
    # o3d.visualization.draw_geometries([ct_mesh_t, ra_re_mesh_t])


if __name__ == '__main__':
    data_dir = '../data/our_better'
    save_dir = 'preprocessed_data'

    limit_size = 128
    preliminary_case = '../data/GpR33'

    case_list = list(glob.glob(data_dir + '/*'))
    # ^ Set preliminary ratio for CT Image
    preliminary_info = gen_save_mhd_uniform_first(preliminary_case, limit_size)

    # ^ define partial function
    for case_name in case_list:
        gen_save_mhd_based_scale(case_name, base_info=preliminary_info, lim_size=limit_size)




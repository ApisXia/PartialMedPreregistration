import copy
import open3d as o3d
from glob import glob
from basic_utils.show_data import load_dcm, get_iso_surface, to_show_surface
from exec_pipeline.post_DL_utils.rigid_matrix_from_predictions import matrix_diff_analysis

if __name__ == "__main__":
    ori_data_dir = 'registration_save/results_mse_samp_f2_1/ori'
    dl_data_dir = 'results_mse_samp_f2_1/DL_wf/{}'
    ori_case_list = list(glob(ori_data_dir + '/*'))
    gt_path = '/home2/reg/dataset/GpR/%s/3DRA/Registrated/*'

    # ori_index = [0, 4, 14]  # ori good
    # ori_index = [17, 16, 12, 11, 7, 2]  # DL good
    ori_case = ori_case_list[9]

    # ori_case = ori_case_list[0]

    case_name = ori_case.split('/')[-1]
    gt_folder = list(glob(gt_path % case_name))[0]
    or_pixels, or_matrix, _, _ = load_dcm(ori_case)
    gt_pixels, gt_matrix, _, _ = load_dcm(gt_folder)
    DL_pixels, DL_matrix, _, _ = load_dcm(dl_data_dir.format(case_name))

    or_trans_diff, or_rot_diff = matrix_diff_analysis(gt_matrix.copy(), or_matrix.copy())
    DL_trans_diff, DL_rot_diff = matrix_diff_analysis(gt_matrix.copy(), DL_matrix.copy())

    print('**** Case: {} ****'.format(case_name))
    print('Or trans diff: {}, rot diff: {}'.format(or_trans_diff, or_rot_diff))
    print('DL trans diff: {}, rot diff: {}'.format(DL_trans_diff, DL_rot_diff))

    gt_verts, gt_faces, gt_normals, gt_values = get_iso_surface(gt_pixels, 1000, 2000)
    toshow_verts, toshow_faces, toshow_normals, toshow_values = get_iso_surface(or_pixels, 1000, 2000)
    # toshow_verts, toshow_faces, toshow_normals, toshow_values = get_iso_surface(DL_pixels, 1000, 2000)

    gt_mesh = to_show_surface(gt_verts, gt_faces, gt_normals)
    toshow_mesh = to_show_surface(toshow_verts, toshow_faces, toshow_normals)

    gt_mesh_t = copy.deepcopy(gt_mesh).transform(gt_matrix)
    toshow_mesh_t = copy.deepcopy(toshow_mesh).transform(or_matrix)
    # toshow_mesh_t = copy.deepcopy(toshow_mesh).transform(DL_matrix)

    gt_mesh_t.paint_uniform_color([1, 0.706, 0])  # yellow
    toshow_mesh_t.paint_uniform_color([0, 0.706, 0])  # green

    # o3d.visualization.draw_geometries([ra_re_mesh_t])
    o3d.visualization.draw_geometries([gt_mesh_t, toshow_mesh_t])

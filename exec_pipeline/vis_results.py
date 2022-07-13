import copy
import open3d as o3d
from glob import glob
from basic_utils.show_data import load_dcm, get_iso_surface, to_show_surface
from exec_pipeline.post_DL_utils.rigid_matrix_from_predictions import lankmark_diff_analysis

if __name__ == "__main__":
    ori_data_dir = 'registration_Ori_save/GpR100'
    dl_data_dir = 'registration_DL_wf_save/GpR100'
    gt_data_dir = '../data_rel/our_better/GpR100/3DRA/Registrated/3D RA'

    or_pixels, or_matrix, _, _ = load_dcm(ori_data_dir)
    gt_pixels, gt_matrix, _, _ = load_dcm(gt_data_dir)
    DL_pixels, DL_matrix, _, _ = load_dcm(dl_data_dir)

    or_error = lankmark_diff_analysis(gt_matrix.copy(), or_matrix.copy(), gt_pixels.shape)
    DL_error = lankmark_diff_analysis(gt_matrix.copy(), DL_matrix.copy(), gt_pixels.shape)

    print('Or diff: {}'.format(or_error))
    print('DL diff: {}'.format(DL_error))

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


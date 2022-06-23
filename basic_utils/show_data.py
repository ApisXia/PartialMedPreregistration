import os
import numpy as np
import pydicom
from basic_utils.import_util_libs.dicom_numpy import combine_slices
import open3d as o3d
from skimage import measure


def load_dcm(dcm_path):
    dcm_image = []
    for filename in sorted(os.listdir(dcm_path)):
        if filename[:2] == '._':
            continue

        dcm_image.append(pydicom.dcmread(os.path.join(dcm_path, filename)))
    # + Util this line, the process is identical.
    pixels, t_matrix, directions, spacings = combine_slices(dcm_image)

    return pixels, t_matrix, directions, spacings


def load_dcm_more(dcm_path):
    dcm_image = []
    for filename in sorted(os.listdir(dcm_path)):
        if filename[:2] == '._':
            continue

        dcm_image.append(pydicom.dcmread(os.path.join(dcm_path, filename)))
    # + Util this line, the process is identical.
    pixels, t_matrix, directions, spacings = combine_slices(dcm_image)

    def _sort_by_slice_position(slice_datasets):
        slice_positions = _slice_positions(slice_datasets)
        return [d for (s, d) in sorted(zip(slice_positions, slice_datasets))]

    def _slice_positions(slice_datasets):
        image_orientation = slice_datasets[0].ImageOrientationPatient
        row_cosine, column_cosine, slice_cosine = _extract_cosines(image_orientation)
        return [np.dot(slice_cosine, d.ImagePositionPatient) for d in slice_datasets]

    def _extract_cosines(image_orientation):
        row_cosine = np.array(image_orientation[:3])  # + Former is row
        column_cosine = np.array(image_orientation[3:])  # + Latter is column
        slice_cosine = np.cross(row_cosine, column_cosine)  # + Cross to be slice
        return row_cosine, column_cosine, slice_cosine

    dcm_new_array = _sort_by_slice_position(dcm_image)

    return pixels, t_matrix, directions, spacings, dcm_new_array


def get_iso_surface(volume, lower_bound, upper_bound):
    iso_solid = np.logical_and(lower_bound <= volume, volume < upper_bound)

    if (iso_solid == 1).sum() == 0:
        verts, faces, normals, values = [], [], [], []
    else:
        verts, faces, normals, values = measure.marching_cubes(iso_solid, 0.)

    return verts, faces, normals, values


def to_show_points(points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def to_show_surface(verts, faces, normals):
    surf = o3d.geometry.TriangleMesh()
    surf.vertices = o3d.utility.Vector3dVector(verts)
    surf.triangles = o3d.utility.Vector3iVector(faces)
    surf.vertex_normals = o3d.utility.Vector3dVector(normals)
    return surf


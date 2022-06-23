from torch.utils.data import Dataset
import torch
import numpy as np
import traceback
import os


class SamplingLoader(Dataset):
    def __init__(self, matrix_3dra, batch_size, num_workers, patch_num=150, ct_align=None, ra_trans=None):
        self.ra_matrix = matrix_3dra

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patch_num = patch_num

        self._lower_bound = -500.
        self._upper_bound = 3000.
        assert self._lower_bound < self._upper_bound

        self._box_size = 16
        self._patch_loc_loader = []
        self._lim_not_find = 100

        self.ct_align = ct_align
        self.ra_trans = ra_trans

    def __len__(self):
        return self.patch_num

    def __getitem__(self, idx):
        not_find_counter = 0
        while True:
            # rand_starts, rand_ends = self._gen_fixed_bounding_box() # todo
            rand_starts, rand_ends = self._gen_fixed_bounding_box()
            if (rand_starts, rand_ends) in self._patch_loc_loader:
                not_find_counter += 1
                if not_find_counter > self._lim_not_find:
                    raise ValueError('3DRA matrix is too small to find new patch!')
                continue
            # get ori center
            gt_ori_center = (np.asarray(rand_starts) + np.asarray(rand_ends) - 1) / 2
            # # get pred center and check
            # center_vector_format = np.ones([4, 1])
            # center_vector_format[0:3, 0] = gt_ori_center
            # gt_trans_center = np.matmul(self.ct_align, np.matmul(self.ra_trans, center_vector_format))
            # gt_trans_center = gt_trans_center[0:3]
            # if (gt_trans_center < 0).any() and (gt_trans_center > (self._box_size - 1)).any():
            #     not_find_counter += 1
            #     if not_find_counter > self._lim_not_find:
            #         raise ValueError('3DRA matrix is too small to find new patch!')
            #     continue
            self._patch_loc_loader.append((rand_starts, rand_ends))
            ra_re_rand_image_pixel = self.ra_matrix[rand_starts[0]:rand_ends[0],
                                                    rand_starts[1]:rand_ends[1],
                                                    rand_starts[2]:rand_ends[2]]
            if self._check_patch_background_ratio(ra_re_rand_image_pixel) > 0.6:
                continue
            break

        # get pred center
        gt_trans_center = None
        if (self.ct_align is not None) and (self.ra_trans is not None):
            center_vector_format = np.ones([4, 1])
            center_vector_format[0:3, 0] = gt_ori_center.copy()
            gt_trans_center = np.matmul(self.ct_align, np.matmul(self.ra_trans, center_vector_format))
            gt_trans_center = gt_trans_center[0:3]
            return {'ra_sample_matrix': np.array(self.scale_input(ra_re_rand_image_pixel), dtype=np.float32),
                    'ra_sample_loc_origin': gt_ori_center,
                    'should_pred_loc': gt_trans_center}
        if np.min(ra_re_rand_image_pixel.shape) != 16:
            raise ValueError
        return {'ra_sample_matrix': np.array(self.scale_input(ra_re_rand_image_pixel), dtype=np.float32),
                'ra_sample_loc_origin': gt_ori_center}

    def _gen_fixed_bounding_box(self):
        s_x, s_y, s_z = self.ra_matrix.shape
        if s_x < self._box_size or s_y < self._box_size or s_z < self._box_size:
            raise ValueError
        x_start_p = np.random.randint(0, s_x - self._box_size + 1)
        y_start_p = np.random.randint(0, s_y - self._box_size + 1)
        z_start_p = np.random.randint(0, s_z - self._box_size + 1)
        x_end_p = x_start_p + self._box_size
        y_end_p = y_start_p + self._box_size
        z_end_p = z_start_p + self._box_size
        return (x_start_p, y_start_p, z_start_p), (x_end_p, y_end_p, z_end_p)

    def _gen_fixed_bounding_box_specialjustlayer(self):
        s_x, s_y, s_z = self.ra_matrix.shape
        if s_x < self._box_size or s_y < self._box_size or s_z < self._box_size:
            raise ValueError
        x_start_p = np.random.randint(0, s_x - self._box_size + 1)
        # y_start_p = np.random.randint(0, s_y - self._box_size + 1)
        y_start_p = int(s_y/2)-8
        z_start_p = np.random.randint(0, s_z - self._box_size + 1)
        x_end_p = x_start_p + self._box_size
        # y_end_p = y_start_p + self._box_size
        y_end_p = int(s_y/2)+8
        z_end_p = z_start_p + self._box_size
        return (x_start_p, y_start_p, z_start_p), (x_end_p, y_end_p, z_end_p)

    def _check_patch_background_ratio(self, patch):
        over_threshold_num = (patch > 0.).sum()
        over_threshold_ratio = over_threshold_num / self._box_size ** 3
        return over_threshold_ratio

    def clear_patch_holders(self):
        self._patch_loc_loader = []

    def scale_input(self, matrix):
        matrix[matrix < self._lower_bound] = self._lower_bound
        matrix[matrix > self._upper_bound] = self._upper_bound
        return (matrix - self._lower_bound) / (self._upper_bound - self._lower_bound) - 0.5

    def get_loader(self):
        self.clear_patch_holders()
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers)


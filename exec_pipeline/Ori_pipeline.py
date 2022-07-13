import os
import copy
import numpy as np
from glob import glob
from basic_utils.show_data import load_dcm_more, load_dcm
from exec_pipeline.pre_DL_utils.load_ins import resample, resample_based_scale
from exec_pipeline.post_DL_utils.aff_reg_func_packs import get_paired_reg_matrix


class OriginalMatch(object):
    def __init__(self, orifile_path, basic_scale=0.5813953391663639, lim_size=128):
        self._basic_scale = basic_scale
        self._lim_size = lim_size

        self.pixels, _, directions, self.spacings, self.dcm_array = load_dcm_more(orifile_path)

    def match(self, ctfile_path, save_folder_path, iter_num=500, metric='mse'):
        os.makedirs(save_folder_path, exist_ok=True)

        # ^ matrix of resampled registered 3dra
        new_spacings = self.spacings * self._basic_scale
        new_pixels, _ = resample(self.pixels, new_spacings)

        # ^ load corresponding CT
        ct_image_pixel, ct_image_matrix, _, _ = load_dcm(ctfile_path)
        ct_old_spacing = ct_image_matrix.diagonal()[0:3]
        ct_old_translation = ct_image_matrix[0:3, 3]
        ct_filled_pixels, ct_shift_z = resample_based_scale(ct_image_pixel, ct_old_spacing,
                                                            lim_size=self._lim_size,
                                                            base_scale=self._basic_scale)

        # ^ refine result
        refine_matrix = get_paired_reg_matrix(ct_filled_pixels, new_pixels, metric=metric, iter_num=iter_num)
        # refine_matrix = get_matrix_we_use(refine_matrix.copy(), new_pixels.shape)
        refine_matrix_save = refine_matrix.copy()

        # ^ save output2
        newOrientationSet = list(np.concatenate((refine_matrix[0:3, 0], refine_matrix[0:3, 1]), axis=0))
        translationSet = refine_matrix[0:3, 3].copy()
        translationSet[2] -= ct_shift_z
        translationSet /= self._basic_scale
        translationSet += ct_old_translation
        sliceOri = refine_matrix[0:3, 2]
        dcm_array = copy.deepcopy(self.dcm_array)
        for i in range(len(dcm_array)):
            dcm_array[i].ImageOrientationPatient = newOrientationSet
            dcm_array[i].ImagePositionPatient = list(translationSet + sliceOri * i * self.spacings[-1])
        # save
        save_path = save_folder_path + '/IM-%04d.dcm'
        for idx, f in enumerate(dcm_array):
            f.save_as(save_path % idx)


if __name__ == '__main__':
    data_dir = '../data_rel/our_better'

    case_list = list(glob(data_dir + '/*'))
    case_name = case_list[0]
    sa_p = 'registration_Ori_save/' + case_name.split('\\')[-1]
    os.makedirs(sa_p, exist_ok=True)
    or_p = list(glob(case_name + '/3DRA/Original/*'))[0]
    ct_p = case_name + '/CT'

    ori_matcher = OriginalMatch(or_p)
    ori_matcher.match(ct_p, sa_p, iter_num=500, metric='mse')

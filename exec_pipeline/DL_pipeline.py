import torch
import os
import copy
import numpy as np
from glob import glob
from basic_utils.show_data import load_dcm_more, load_dcm
from pre_DL_utils.load_ins import resample, resample_based_scale
from DL_utils.sampling_loader import SamplingLoader
from DL_utils.model import Simple_ConvMODEL
from DL_utils.eval_utils import Evaluator
from post_DL_utils.rigid_matrix_from_predictions import gen_rigid_matrix
from post_DL_utils.aff_reg_func_packs import get_paired_reg_matrix, map_back_matrix, get_matrix_we_use
from pred_optimizers.RANSAC import matrix_RANSAC_optimizer, matrix_RANSAC_filter


class DeepLearningMatch(object):
    def __init__(self,
                 orifile_path,
                 basic_scale=0.5813953391663639,
                 basic_shift_z=12,
                 basic_ct_translation=np.asarray([-109.94100, -104.31600, -643.50000]),
                 basic_ct_info_path='DL_utils/ct_3dra_scaled_basic.npz',
                 batch_size=8,
                 num_workers=12,
                 patch_num=100,
                 lim_size=128,
                 checkpoint_path='DL_utils/checkpoint_fold0.tar',
                 ct_align=None,
                 ra_trans=None,
                 expand_dims=0):
        self._basic_scale = basic_scale
        self._basic_shift_z = basic_shift_z
        self._basic_ct_translation = basic_ct_translation
        self._basic_ct_info_path = basic_ct_info_path
        self._lim_size = lim_size

        pixels, _, _, self.spacings, self.dcm_array = load_dcm_more(orifile_path)
        # ^ matrix of resampled registered 3dra
        new_spacings = self.spacings * self._basic_scale
        self.new_pixels, _ = resample(pixels, new_spacings)
        # ^ get patch loader
        patch_set = SamplingLoader(self.new_pixels, batch_size, num_workers, patch_num, ct_align, ra_trans)
        # ^ define model and eval utils
        net = Simple_ConvMODEL()
        self.evaluator = Evaluator(net,
                                   patch_set,
                                   device=torch.device("cuda"),
                                   checkpoint_path=checkpoint_path,
                                   expand_dims=expand_dims)

    def match_to_template(self, save_folder_path=None, save_ops=True):
        eval_pred, eval_ori = self.evaluator.eval_registration()
        # ^ get the rigid matrix
        # pred_matrix = gen_rigid_matrix(eval_ori, eval_pred)
        try:
            pred_matrix = matrix_RANSAC_optimizer(eval_ori, eval_pred)
        except ValueError:
            pred_matrix = matrix_RANSAC_optimizer(eval_ori, eval_pred, error_threshold=20)

        pred_matrix_save = pred_matrix.copy()
        if not save_ops:
            return pred_matrix_save

        os.makedirs(save_folder_path, exist_ok=True)
        # ^ save output1
        newOrientationSet = list(np.concatenate((pred_matrix[0:3, 0], pred_matrix[0:3, 1]), axis=0))
        translationSet = pred_matrix[0:3, 3]
        translationSet[2] -= self._basic_shift_z
        translationSet /= self._basic_scale
        translationSet += self._basic_ct_translation
        sliceOri = pred_matrix[0:3, 2]
        dcm_array = copy.deepcopy(self.dcm_array)
        for i in range(len(dcm_array)):
            dcm_array[i].ImageOrientationPatient = newOrientationSet
            dcm_array[i].ImagePositionPatient = list(translationSet + sliceOri * i * self.spacings[-1])
        # save
        save_path = save_folder_path + '/IM-%04d.dcm'
        for idx, f in enumerate(dcm_array):
            f.save_as(save_path % idx)

    def match_to_ori_wwo_refine(self, ctfile_path, save_wo_folder_path, save_wf_folder_path, iter_num=300, metric='mse'):
        os.makedirs(save_wo_folder_path, exist_ok=True)
        os.makedirs(save_wf_folder_path, exist_ok=True)

        pred_matrix_save = self.match_to_template(save_ops=False)
        # ^ load corresponding CT
        ct_image_pixel, ct_image_matrix, _, _ = load_dcm(ctfile_path)
        ct_old_spacing = ct_image_matrix.diagonal()[0:3]
        ct_old_translation = ct_image_matrix[0:3, 3]
        ct_filled_pixels, ct_shift_z = resample_based_scale(ct_image_pixel, ct_old_spacing,
                                                            lim_size=self._lim_size,
                                                            base_scale=self._basic_scale)
        # ^ load base CT file
        basic_ct_load = np.load(self._basic_ct_info_path)
        basic_ct_pixels = basic_ct_load['images_ct']
        # ^ get corresponding CT trans matrix and wo refine matrix
        ct_trans_matrix = get_paired_reg_matrix(basic_ct_pixels, ct_filled_pixels)
        wo_refine_matrix = np.matmul(np.linalg.inv(ct_trans_matrix), pred_matrix_save)

        # ^ save without output
        newOrientationSet = list(np.concatenate((wo_refine_matrix[0:3, 0], wo_refine_matrix[0:3, 1]), axis=0))
        translationSet = wo_refine_matrix[0:3, 3].copy()
        translationSet[2] -= ct_shift_z
        translationSet /= self._basic_scale
        translationSet += ct_old_translation
        sliceOri = wo_refine_matrix[0:3, 2]
        dcm_array = copy.deepcopy(self.dcm_array)
        for i in range(len(dcm_array)):
            dcm_array[i].ImageOrientationPatient = newOrientationSet
            dcm_array[i].ImagePositionPatient = list(translationSet + sliceOri * i * self.spacings[-1])
        # save
        save_path = save_wo_folder_path + '/IM-%04d.dcm'
        for idx, f in enumerate(dcm_array):
            f.save_as(save_path % idx)

        # ^ get corresponding CT to basic CT matrix
        ct_init_matrix = get_paired_reg_matrix(basic_ct_pixels, ct_filled_pixels, ori_ops=True)
        ct_init_matrix_inv = np.linalg.inv(ct_init_matrix.copy())
        mra_init_matrix = map_back_matrix(pred_matrix_save.copy(), self.new_pixels.shape)
        # ^ refine result
        refine_matrix = get_paired_reg_matrix(ct_filled_pixels, self.new_pixels,
                                              init_3dra_matrix=mra_init_matrix.copy(),
                                              init_ct_matrix=ct_init_matrix_inv.copy(),
                                              ori_ops=True,
                                              iter_num=iter_num,
                                              metric=metric)

        refine_matrix = get_matrix_we_use(refine_matrix.copy(), self.new_pixels.shape)
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
        save_path = save_wf_folder_path + '/IM-%04d.dcm'
        for idx, f in enumerate(dcm_array):
            f.save_as(save_path % idx)

    def match_to_ori_file(self, ctfile_path, save_folder_path, iter_num=300):
        os.makedirs(save_folder_path, exist_ok=True)

        pred_matrix_save = self.match_to_template(save_ops=False)
        # ^ load corresponding CT
        ct_image_pixel, ct_image_matrix, _, _ = load_dcm(ctfile_path)
        ct_old_spacing = ct_image_matrix.diagonal()[0:3]
        ct_old_translation = ct_image_matrix[0:3, 3]
        ct_filled_pixels, ct_shift_z = resample_based_scale(ct_image_pixel, ct_old_spacing,
                                                            lim_size=self._lim_size,
                                                            base_scale=self._basic_scale)
        # ^ load base CT file
        basic_ct_load = np.load(self._basic_ct_info_path)
        basic_ct_pixels = basic_ct_load['images_ct']
        # ^ get corresponding CT to basic CT matrix
        ct_init_matrix = get_paired_reg_matrix(basic_ct_pixels, ct_filled_pixels, ori_ops=True)
        ct_init_matrix_inv = np.linalg.inv(ct_init_matrix.copy())
        mra_init_matrix = map_back_matrix(pred_matrix_save.copy(), self.new_pixels.shape)
        # ^ refine result
        refine_matrix = get_paired_reg_matrix(ct_filled_pixels, self.new_pixels,
                                              init_3dra_matrix=mra_init_matrix.copy(),
                                              init_ct_matrix=ct_init_matrix_inv.copy(),
                                              ori_ops=True,
                                              metric='nmi',
                                              iter_num=iter_num)

        refine_matrix = get_matrix_we_use(refine_matrix.copy(), self.new_pixels.shape)
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

    def get_a_lot_patches(self):
        eval_pred, eval_ori = self.evaluator.eval_registration()
        wo_R_ps = (eval_pred.copy(), eval_ori.copy())
        # try:
        #     R_filter_pts = matrix_RANSAC_filter(eval_ori, eval_pred)
        # except ValueError:
        #     R_filter_pts = matrix_RANSAC_filter(eval_ori, eval_pred, error_threshold=20)
        return wo_R_ps #, R_filter_pts


if __name__ == '__main__':
    fold = 0

    split_file = '/home2/reg/dataset/CT_3DRA_split.npz'
    split_load = np.load(split_file)['split_{}'.format(fold)]
    case_name = split_load[0]
    sa_p = 'temp_save/' + case_name
    os.makedirs(sa_p, exist_ok=True)
    or_p = list(glob('/home2/reg/dataset/GpR/{}/3DRA/Original/*'.format(case_name)))[0]
    ct_p = '/home2/reg/dataset/GpR/{}/CT'.format(case_name)

    # gt_p = list(glob('/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/GpR_mhd_center_aligned_128_16_new/{}/ct_3dra_scaled.npz'.format(case_name)))[0]
    # gt_load = np.load(gt_p)
    # gt_ct_align = gt_load['matrix_ct_aligned']
    # gt_ra_trans = gt_load['matrix_3dra']

    # DL_matcher = DeepLearningMatch(or_p, expand_dims=32,
    #                                checkpoint_path='DL_utils/checkpoint_expand_fold0.tar')
    DL_matcher = DeepLearningMatch(or_p)
    # DL_matcher = DeepLearningMatch(or_p, ct_align=gt_ct_align, ra_trans=gt_ra_trans)
    DL_matcher.match_to_ori_file(ct_p, sa_p)
    c = 1
    # eval_matrix = DL_matcher.match_to_template(save_ops=False)
    # eval_pred, eval_ori, gt_loc = DL_matcher.evaluator.eval_registration_for_vis()


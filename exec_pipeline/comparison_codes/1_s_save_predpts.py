import numpy as np
from glob import glob
from Ori_pipeline import OriginalMatch
from DL_pipeline import DeepLearningMatch
import os
import itertools
import pandas as pd


def gen_all_kinds_predictions(fold, logger):
    split_file = '/home2/reg/dataset/CT_3DRA_split_no46.npz'
    split_load = list(np.load(split_file)['split_{}'.format(fold)])
    # split_load.remove('GpR19')


    for case_name in split_load:
        print('Now is working on {}.....'.format(case_name))

        s_file_name = 'pt_preds_vis/{}.xlsx'

        or_p = list(glob('/home2/reg/dataset/GpR/{}/3DRA/Registrated/*'.format(case_name)))[0]
        gt_info = '/home2/reg/dataset/GpR_mhd_center_aligned_128_16_new/{}/ct_3dra_scaled.npz'.format(case_name)
        gt_info_load = np.load(gt_info)
        gt_3dra_matrix = gt_info_load['matrix_3dra']

        DL_matcher = DeepLearningMatch(or_p,
                                       basic_ct_info_path='../DL_utils/ct_3dra_scaled_basic.npz',
                                       checkpoint_path='../DL_utils/checkpoint_fold{}.tar'.format(fold),
                                       patch_num=50)
        # DL_matcher.match_to_template(sa_p_dl_t)
        try:
            # DL_matcher.match_to_ori_file(ct_p, sa_p_dl)
            # DL_pred_set, RANSAC_idx = DL_matcher.get_a_lot_patches()
            DL_pred_set = DL_matcher.get_a_lot_patches()
        except ValueError:
            logger.write('{}/n'.format(case_name))
            pass

        DL_pred_pt, init_pt = DL_pred_set
        Real_pt = np.transpose(np.matmul(gt_3dra_matrix, np.concatenate((np.transpose(init_pt), np.ones((1, 50))), axis=0))[0:3, :])

        writer = pd.ExcelWriter(s_file_name.format(case_name), engine='xlsxwriter')
        pred_gt_df = pd.DataFrame({'pred_x': DL_pred_pt[:, 0],
                                   'pred_y': DL_pred_pt[:, 1],
                                   'pred_z': DL_pred_pt[:, 2],
                                   'gt_x':   Real_pt[:, 0],
                                   'gt_y':   Real_pt[:, 1],
                                   'gt_z':   Real_pt[:, 2],
                                   'ori_x':  init_pt[:, 0],
                                   'ori_y':  init_pt[:, 1],
                                   'ori_z':  init_pt[:, 2]})
        # R_id_df = pd.DataFrame({'RANSAC ID': RANSAC_idx})
        pred_gt_df.to_excel(writer, sheet_name='pred_gt', index=False)
        # R_id_df.to_excel(writer, sheet_name='R ID', index=False)
        writer.save()
        print('Finished case: {}'.format(case_name))


if __name__ == "__main__":
    folds = [0, 1, 2, 3, 4]
    prior_identifier = 'mse'
    post_identifier = 'noRANSAC'
    os.makedirs('pt_preds_vis', exist_ok=True)
    log = open('pt_preds_vis/bad_files.txt', 'w')
    for f in folds:
        gen_all_kinds_predictions(f, log)
    log.close()

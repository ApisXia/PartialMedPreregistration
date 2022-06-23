from glob import glob
from basic_utils.show_data import load_dcm
from post_DL_utils.rigid_matrix_from_predictions import matrix_diff_analysis, matrix_diff_analysis_2, lankmark_diff_analysis
import numpy as np
import pandas as pd
import itertools


def analyze_outputs_save(base_path):
    # base_path = 'results_samp_mse_f0_0_test3'
    dl_data_dir = base_path + '/DL_wf/'
    dl_wo_data_dir = base_path + '/DL_wo/{}'
    dl_t_data_dir = base_path + '/DL_t/{}'
    dl_case_list = list(glob(dl_data_dir + '/*'))
    gt_path = '/home2/reg/dataset/GpR/%s/3DRA/Registrated/*'

    case_result_save = {}
    # dataframe info
    name_holder = []
    DL_trans_diff_holder = []
    DL_rot_diff_holder = []
    DL_wo_trans_diff_holder = []
    DL_wo_rot_diff_holder = []

    DL_error_holder = []
    DL_wo_error_holder = []

    for i, dl_case in enumerate(dl_case_list):
        case_name = dl_case.split('/')[-1]
        gt_folder = list(glob(gt_path % case_name))[0]
        print('Now show the case {}: {}'.format(i, case_name))
        gt_pixel, gt_matrix, _, _ = load_dcm(gt_folder)
        _, DL_matrix, _, _ = load_dcm(dl_case)
        _, DL_wo_matrix, _, _ = load_dcm(dl_wo_data_dir.format(case_name))
        # _, DL_t_matrix, _, _ = load_dcm(dl_t_data_dir.format(case_name))

        DL_trans_diff, DL_rot_diff = matrix_diff_analysis_2(gt_matrix.copy(), DL_matrix.copy())
        DL_error = lankmark_diff_analysis(gt_matrix.copy(), DL_matrix.copy(), gt_pixel.shape)
        DL_wo_trans_diff, DL_wo_rot_diff = matrix_diff_analysis_2(gt_matrix.copy(), DL_wo_matrix.copy())
        DL_wo_error = lankmark_diff_analysis(gt_matrix.copy(), DL_wo_matrix.copy(), gt_pixel.shape)
        # DL_t_trans_diff, DL_t_rot_diff = matrix_diff_analysis_2(gt_matrix.copy(), DL_t_matrix.copy())

        print('**** Case: {} ****'.format(case_name))
        print('DL     trans diff: {}, rot diff: {}'.format(DL_trans_diff, DL_rot_diff))
        print('DL-wo  trans diff: {}, rot diff: {}'.format(DL_wo_trans_diff, DL_wo_rot_diff))
        # print('DL_t trans diff: {}, rot diff: {}'.format(DL_t_trans_diff, DL_t_rot_diff))

        # add in dataframe
        name_holder.append(case_name)
        DL_trans_diff_holder.append(DL_trans_diff)
        DL_rot_diff_holder.append(DL_rot_diff)
        DL_wo_trans_diff_holder.append(DL_wo_trans_diff)
        DL_wo_rot_diff_holder.append(DL_wo_rot_diff)

        DL_error_holder.append(DL_error)
        DL_wo_error_holder.append(DL_wo_error)

    stats_data = {'Data ID':          name_holder,
                  'DL Trans Diff':    DL_trans_diff_holder,
                  'DL-wo Trans Diff': DL_wo_trans_diff_holder,
                  'DL Rot Diff':      DL_rot_diff_holder,
                  'DL-wo Rot Diff':   DL_wo_rot_diff_holder,
                  'DL error':         DL_error_holder,
                  'DL-wo error':      DL_wo_error_holder}

    df = pd.DataFrame(stats_data)
    df.to_excel('{}.xlsx'.format(base_path), index=False)


if __name__ == "__main__":
    folds = [0, 1, 2, 3, 4]
    test_id = [0]
    for f, z in itertools.product(folds, test_id):
        b_path = 'results_STN_f{}_{}_Model0_extend_MSE_2'.format(f, z)
        analyze_outputs_save(b_path)

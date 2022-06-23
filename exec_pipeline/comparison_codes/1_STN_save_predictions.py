import numpy as np
from glob import glob
from Ori_pipeline import OriginalMatch
from DL_pipeline import DeepLearningMatch
import os
import itertools
from STN_pipline.STN_pipeline import STNMatch


def gen_all_kinds_predictions(fold, base_path):
    split_file = '/home2/reg/dataset/CT_3DRA_split_no46.npz'
    split_load = list(np.load(split_file)['split_{}'.format(fold)])
    # split_load.remove('GpR19')
    os.makedirs(base_path, exist_ok=True)
    f = open(base_path + '/bad_files.txt', 'w')

    for case_name in split_load:
        print('Now is working on {}.....'.format(case_name))

        # sa_p_or = base_path + '/ori/' + case_name
        # sa_p_dl_wf = base_path + '/DL_wf/' + case_name
        sa_STN_wf = base_path + '/DL_wf/' + case_name
        # sa_p_dl_wo = base_path + '/DL_wo/' + case_name
        sa_STN_wo = base_path + '/DL_wo/' + case_name
        # sa_p_dl_t = 'results/DL_t/' + case_name

        # or_p = list(glob('/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/GpR/{}/3DRA/Original/*'.format(case_name)))[0]
        or_p = list(glob('/home2/reg/dataset/GpR/{}/3DRA/Registrated/*'.format(case_name)))[0]
        ct_p = '/home2/reg/dataset/GpR/{}/CT'.format(case_name)

        # ori_matcher = OriginalMatch(or_p)
        # ori_matcher.match(ct_p, sa_p_or, iter_num=500)

        STN_matcher = STNMatch(or_p,
                               basic_ct_info_path='../DL_utils/ct_3dra_scaled_basic.npz',
                               checkpoint_path='../STN_pipline/DL_utils/ckpts_0_extend/checkpoint_f{}.tar'.format(fold))
        # DL_matcher.match_to_template(sa_p_dl_t)
        try:
            # DL_matcher.match_to_ori_file(ct_p, sa_p_dl)
            STN_matcher.match_to_ori_wwo_refine(ct_p, sa_STN_wo, sa_STN_wf, iter_num=500)
        except ValueError:
            f.write('{}/n'.format(case_name))
            pass
        # DL_matcher.match_to_ori_wo_refine(ct_p, sa_p_dl)

        print('Finished case: {}'.format(case_name))
    f.close()


if __name__ == "__main__":
    folds = [0, 1, 2, 3, 4]
    test_id = [0]
    prior_identifier = 'STN'
    post_identifier = 'Model0_extend_MSE_2'
    s_path_format = 'results_%s_f{}_{}_%s' % (prior_identifier, post_identifier)
    for f, z in itertools.product(folds, test_id):
        s_path = s_path_format.format(f, z)
        gen_all_kinds_predictions(f, s_path)

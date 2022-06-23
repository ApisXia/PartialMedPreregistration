import numpy as np
from glob import glob
from Ori_pipeline import OriginalMatch
from DL_pipeline import DeepLearningMatch
import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap


def gen_all_kinds_predictions(fold, logger):
    split_file = '/home2/reg/dataset/CT_3DRA_split_no46.npz'
    split_load = list(np.load(split_file)['split_{}'.format(fold)])
    # split_load.remove('GpR19')
    start_color = [69, 94, 255]
    end_color = [255, 84, 71]
    my_cm = np.vstack((np.linspace(start_color[0], end_color[0], 256)/256,
                       np.linspace(start_color[1], end_color[1], 256)/256,
                       np.linspace(start_color[2], end_color[2], 256)/256,
                       np.ones(256)))
    my_cm = np.transpose(my_cm)
    my_cm = ListedColormap(my_cm)

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])

    cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal',
                                   cmap=my_cm)

    plt.savefig('just_colorbar', bbox_inches='tight', dpi=600)

    for case_name in split_load:
        print('Now is working on {}.....'.format(case_name))

        s_file_name = 'pt_preds_vis/{}_s{}.png'

        or_p = list(glob('/home2/reg/dataset/GpR/{}/3DRA/Registrated/*'.format(case_name)))[0]
        gt_info = '/home2/reg/dataset/GpR_mhd_center_aligned_128_16_new/{}/ct_3dra_scaled.npz'.format(case_name)
        gt_info_load = np.load(gt_info)
        gt_3dra_matrix = gt_info_load['matrix_3dra']
        images_3dra = gt_info_load['images_3dra']
        x_s, y_s, z_s = images_3dra.shape
        images_3dra[images_3dra < -500.] = -500.
        images_3dra[images_3dra > 3000.] = 3000.

        the_slice = images_3dra[:, int(y_s/2), :]

        fig, ax = plt.subplots()
        ax.imshow(the_slice, cmap='gray')

        DL_matcher = DeepLearningMatch(or_p,
                                       basic_ct_info_path='../DL_utils/ct_3dra_scaled_basic.npz',
                                       checkpoint_path='../DL_utils/checkpoint_fold{}.tar'.format(fold),
                                       patch_num=100)
        # DL_matcher.match_to_template(sa_p_dl_t)
        try:
            # DL_matcher.match_to_ori_file(ct_p, sa_p_dl)
            # DL_pred_set, RANSAC_idx = DL_matcher.get_a_lot_patches()
            DL_pred_set = DL_matcher.get_a_lot_patches()
        except ValueError:
            logger.write('{}/n'.format(case_name))
            pass

        DL_pred_pt, init_pt = DL_pred_set
        Real_pt = np.transpose(np.matmul(gt_3dra_matrix, np.concatenate((np.transpose(init_pt), np.ones((1, 100))), axis=0))[0:3, :])

        loss = np.sqrt(np.sum((Real_pt - DL_pred_pt) ** 2, axis=1))

        ax.scatter(init_pt[:, 2], init_pt[:, 0], c=loss**2, cmap=my_cm)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        # plt.show()
        plt.savefig(s_file_name.format(case_name, images_3dra.shape), bbox_inches='tight', format='png', dpi=200)

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

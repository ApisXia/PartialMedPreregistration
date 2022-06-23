import numpy as np

import model as model
from dataloader import CT3DRA_dataset
import eval_utils
import torch
import os

net = model.Simple_ConvMODEL()

fold = 0
data_path = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/GpR_mhd_center_aligned_128_16_new'
split_file = '/media/apis/WDSSD/Igarashi_Lab_Projs/Proj_MRA_Reg/Dataset/Split_folder/CT_3DRA_split.npz'
eval_dataset = CT3DRA_dataset(fold, 'test', data_path, split_file, 32, 12, eval_opt=True)

exp_name = 'Simple_ConvMODEL_LScale_p16_f{}'.format(fold)

evaluator = eval_utils.Evaluator(net,
                                 eval_dataset,
                                 exp_name,
                                 device=torch.device("cuda"),
                                 checkpoint=294)

for i in range(eval_dataset.data_eval_len()):
    pred, gt, origin, name, ra_m, ct_m, ra_trans, ct_align = evaluator.eval_registration(i)
    save_dir = 'eval_result/p16_fold{}_new'.format(fold)
    os.makedirs(save_dir, exist_ok=True)
    np.savez(save_dir + '/{}.npz'.format(name),
             pred=pred,
             gt=gt,
             origin=origin,
             ra_m=ra_m,
             ct_m=ct_m,
             ra_trans=ra_trans,
             ct_align=ct_align)
    print('Finished {}'.format(name))


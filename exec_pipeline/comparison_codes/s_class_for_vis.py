import pandas as pd
import os
from shutil import copytree

file_indicator = 'results_mse_samp_f{}_1.xlsx'
folder_indicator = '/home2/reg/Send4Kinsensei/for_class/results_mse_samp_f{}_1/{}/{}'
target_folder = '/home2/reg/Send4Kinsensei/'

for i in range(5):
    file_name = file_indicator.format(i)
    file_dp = pd.read_excel(file_name)
    DL_trans_diff = list(file_dp['DL Trans Diff'])
    file_list = list(file_dp['Data ID'])

    for diff, f_name in zip(DL_trans_diff, file_list):
        if diff <= 1.:
            os.makedirs(target_folder + 'less-than-1mm/{}/'.format(f_name))
            t_format = target_folder + 'less-than-1mm/{}/{}'
            copytree(folder_indicator.format(i, 'ori', f_name), t_format.format(f_name, 'ori'))
            copytree(folder_indicator.format(i, 'DL_wf', f_name), t_format.format(f_name, 'DL_wf'))
            copytree(folder_indicator.format(i, 'DL_wo', f_name), t_format.format(f_name, 'DL_wo'))
        elif diff <= 3.:
            os.makedirs(target_folder + '1-3mm/{}/'.format(f_name))
            t_format = target_folder + '1-3mm/{}/{}'
            copytree(folder_indicator.format(i, 'ori', f_name), t_format.format(f_name, 'ori'))
            copytree(folder_indicator.format(i, 'DL_wf', f_name), t_format.format(f_name, 'DL_wf'))
            copytree(folder_indicator.format(i, 'DL_wo', f_name), t_format.format(f_name, 'DL_wo'))
        elif diff <= 5.:
            os.makedirs(target_folder + '3-5mm/{}/'.format(f_name))
            t_format = target_folder + '3-5mm/{}/{}'
            copytree(folder_indicator.format(i, 'ori', f_name), t_format.format(f_name, 'ori'))
            copytree(folder_indicator.format(i, 'DL_wf', f_name), t_format.format(f_name, 'DL_wf'))
            copytree(folder_indicator.format(i, 'DL_wo', f_name), t_format.format(f_name, 'DL_wo'))
        elif diff <= 10.:
            os.makedirs(target_folder + '5-10mm/{}/'.format(f_name))
            t_format = target_folder + '5-10mm/{}/{}'
            copytree(folder_indicator.format(i, 'ori', f_name), t_format.format(f_name, 'ori'))
            copytree(folder_indicator.format(i, 'DL_wf', f_name), t_format.format(f_name, 'DL_wf'))
            copytree(folder_indicator.format(i, 'DL_wo', f_name), t_format.format(f_name, 'DL_wo'))
        else:
            os.makedirs(target_folder + 'larger-than-10mm/{}/'.format(f_name))
            t_format = target_folder + 'larger-than-10mm/{}/{}'
            copytree(folder_indicator.format(i, 'ori', f_name), t_format.format(f_name, 'ori'))
            copytree(folder_indicator.format(i, 'DL_wf', f_name), t_format.format(f_name, 'DL_wf'))
            copytree(folder_indicator.format(i, 'DL_wo', f_name), t_format.format(f_name, 'DL_wo'))
